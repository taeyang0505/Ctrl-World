"""
Ctrl-World Table 1 재현용 지표 계산.

원본 저장소에는 지표 코드가 전혀 없어서(grep 0건) 직접 구현한다.

입력 규약 — 모든 함수가 동일하다:
    pred, gt : np.uint8, shape (N, T, H, W, 3), RGB, 값 범위 [0, 255]
        N = 클립 수, T = 프레임 수

논문(Table 1)이 명시하지 않아 우리가 정한 것들은 README에 근거와 함께 적어 두었다.
가장 중요한 두 가지:
    - LPIPS 백본: AlexNet (논문 미명시. VGG로 재면 값이 달라진다)
    - 조건 프레임 제외: 각 라운드의 첫 프레임은 입력으로 주어진 프레임이라
      예측이 아니다. 포함시키면 PSNR이 부풀려진다.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "to_tensor_bt",
    "compute_psnr_ssim",
    "compute_lpips",
    "compute_fid",
    "compute_fvd",
    "compute_all",
]


# ---------------------------------------------------------------------------
# 공통 변환
# ---------------------------------------------------------------------------

def to_tensor_bt(x: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """(N, T, H, W, 3) uint8  ->  (N*T, 3, H, W) float32 [0, 1]"""
    if x.dtype != np.uint8:
        raise TypeError(f"uint8이어야 합니다. 받은 dtype: {x.dtype}")
    if x.ndim != 5 or x.shape[-1] != 3:
        raise ValueError(f"(N,T,H,W,3) 형태여야 합니다. 받은 shape: {x.shape}")
    n, t = x.shape[:2]
    out = torch.from_numpy(x.reshape(n * t, *x.shape[2:])).to(device)
    return out.permute(0, 3, 1, 2).float().div_(255.0)


def _check_pair(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(f"shape 불일치: pred {pred.shape} vs gt {gt.shape}")


# ---------------------------------------------------------------------------
# PSNR / SSIM
# ---------------------------------------------------------------------------

def compute_psnr_ssim(
    pred: np.ndarray,
    gt: np.ndarray,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict:
    """프레임 단위로 계산한 뒤 전체 평균.

    PSNR은 data_range=1.0 기준. SSIM은 torchmetrics 기본값
    (gaussian kernel, kernel_size=11, sigma=1.5)을 쓴다 — skimage의 기본값
    (uniform 7x7)과 다르므로 다른 구현과 비교할 때 주의.
    """
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )

    _check_pair(pred, gt)
    p_all = to_tensor_bt(pred)
    g_all = to_tensor_bt(gt)

    psnrs, ssims = [], []
    for i in range(0, p_all.shape[0], batch_size):
        p = p_all[i : i + batch_size].to(device)
        g = g_all[i : i + batch_size].to(device)
        # PSNR은 프레임마다 따로 — 전체를 한 번에 넣으면 MSE가 뭉개진다
        for k in range(p.shape[0]):
            psnrs.append(
                peak_signal_noise_ratio(p[k : k + 1], g[k : k + 1], data_range=1.0).item()
            )
        ssims.append(
            structural_similarity_index_measure(p, g, data_range=1.0, reduction="none")
            .detach()
            .cpu()
        )

    ssim_all = torch.cat(ssims)
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(ssim_all.mean().item()),
        "psnr_std": float(np.std(psnrs)),
        "ssim_std": float(ssim_all.std().item()),
        "n_frames": int(p_all.shape[0]),
    }


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

def compute_lpips(
    pred: np.ndarray,
    gt: np.ndarray,
    net: str = "alex",
    device: str = "cuda",
    batch_size: int = 64,
) -> dict:
    """LPIPS. 입력을 [-1, 1]로 바꿔 넣는다.

    net='alex'가 관례이나 논문이 명시하지 않았다. VGG로 재면 값이 커진다
    (보통 alex < vgg). 비교할 때 반드시 백본을 함께 적을 것.
    """
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    _check_pair(pred, gt)
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type=net, normalize=False, reduction="mean"
    ).to(device)

    p_all = to_tensor_bt(pred)
    g_all = to_tensor_bt(gt)

    vals = []
    for i in range(0, p_all.shape[0], batch_size):
        p = p_all[i : i + batch_size].to(device) * 2 - 1  # [0,1] -> [-1,1]
        g = g_all[i : i + batch_size].to(device) * 2 - 1
        with torch.no_grad():
            vals.append(metric(p, g).item() * p.shape[0])

    total = float(sum(vals)) / p_all.shape[0]
    return {"lpips": total, "lpips_net": net, "n_frames": int(p_all.shape[0])}


# ---------------------------------------------------------------------------
# FID  (프레임 단위 InceptionV3)
# ---------------------------------------------------------------------------

def compute_fid(
    pred: np.ndarray,
    gt: np.ndarray,
    device: str = "cuda",
    batch_size: int = 64,
    feature: int = 2048,
) -> dict:
    """비디오 프레임을 이미지로 보고 계산하는 FID.

    주의: FID는 표본 수에 민감하다. 프레임이 적으면(수천 미만) 공분산 추정이
    불안정해 값이 과대평가된다. 256클립 x 50프레임 = 12,800장이면 최소한은 된다.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    _check_pair(pred, gt)
    metric = FrechetInceptionDistance(feature=feature, normalize=False).to(device)

    def _feed(arr: np.ndarray, real: bool) -> None:
        n, t = arr.shape[:2]
        flat = arr.reshape(n * t, *arr.shape[2:])
        for i in range(0, flat.shape[0], batch_size):
            chunk = torch.from_numpy(flat[i : i + batch_size]).permute(0, 3, 1, 2)
            metric.update(chunk.to(device), real=real)

    _feed(gt, real=True)
    _feed(pred, real=False)
    return {"fid": float(metric.compute().item())}


# ---------------------------------------------------------------------------
# FVD  (I3D — 별도 체크포인트 필요)
# ---------------------------------------------------------------------------

def compute_fvd(pred: np.ndarray, gt: np.ndarray, **kwargs) -> dict:
    """FVD는 I3D(Kinetics-400) 체크포인트가 필요해 별도 준비가 든다.

    구현체마다 값이 달라지는 것으로 악명이 높다(전처리·리사이즈 방식 차이).
    1차 재현에서는 PSNR/SSIM/LPIPS를 먼저 확정하고, FVD는 2차로 미룬다.
    붙일 때는 어느 구현을 썼는지 반드시 기록할 것.
    """
    raise NotImplementedError(
        "FVD는 아직 구현하지 않았습니다. eval/README.md의 '2차 작업' 항목 참고."
    )


# ---------------------------------------------------------------------------
# 한 번에
# ---------------------------------------------------------------------------

def compute_all(
    pred: np.ndarray,
    gt: np.ndarray,
    device: str = "cuda",
    with_fid: bool = True,
    lpips_net: str = "alex",
) -> dict:
    out = {}
    out.update(compute_psnr_ssim(pred, gt, device=device))
    out.update(compute_lpips(pred, gt, net=lpips_net, device=device))
    if with_fid:
        out.update(compute_fid(pred, gt, device=device))
    return out


# ---------------------------------------------------------------------------
# 자체 검증 — 지표가 제대로 동작하는지 확인
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    rng = np.random.default_rng(0)
    gt = rng.integers(0, 256, size=(2, 4, 192, 320, 3), dtype=np.uint8)

    print("\n[1] 동일 입력 — PSNR=inf, SSIM=1, LPIPS=0 이어야 함")
    r = compute_psnr_ssim(gt, gt, device=dev)
    print(f"    PSNR {r['psnr']:.2f}  SSIM {r['ssim']:.4f}")
    r = compute_lpips(gt, gt, device=dev)
    print(f"    LPIPS {r['lpips']:.6f}")

    print("\n[2] 노이즈를 키우면 단조적으로 나빠져야 함")
    for sigma in (5, 15, 40):
        noisy = np.clip(gt.astype(np.int16) + rng.normal(0, sigma, gt.shape), 0, 255).astype(np.uint8)
        a = compute_psnr_ssim(noisy, gt, device=dev)
        b = compute_lpips(noisy, gt, device=dev)
        print(f"    sigma={sigma:2d}  PSNR {a['psnr']:6.2f}  SSIM {a['ssim']:.4f}  LPIPS {b['lpips']:.4f}")
