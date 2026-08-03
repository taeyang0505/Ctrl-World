#!/usr/bin/env python3
"""
자기회귀 롤아웃의 시간에 따른 품질 저하(드리프트)를 측정한다.

논문 Table 1은 10초 구간 전체를 평균낸 값 하나만 보고한다. 그래서
"10초 동안 품질이 어떻게 떨어지는가"는 알 수 없다. 이 스크립트는
run_eval.py 가 남긴 캐시를 재사용해 **프레임별로** 지표를 계산한다.
롤아웃을 다시 돌리지 않으므로 몇 분이면 끝난다.

왜 중요한가: 월드모델은 자기 예측을 다시 입력으로 넣기 때문에 오차가
누적된다. 어느 시점부터 무너지는지 알면 "몇 초까지 믿고 쓸 수 있는가"를
말할 수 있고, 이는 접촉 인과가 언제 깨지는지 보는 후속 연구의 출발점이 된다.

사용법:
    python eval/analyze_drift.py --cache results/table1/cache --out results/drift
    python eval/analyze_drift.py --cache results/table1/cache --views 0 2 --limit 50
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

VIEW_NAMES = {0: "third_view", 1: "exterior_2", 2: "wrist_view"}
FPS = 5.0  # 전처리에서 15Hz -> 5Hz 로 줄였다 (rgb_skip=3)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="run_eval.py 가 만든 캐시 폴더")
    ap.add_argument("--out", default="results/drift")
    ap.add_argument("--views", nargs="*", type=int, default=[0, 2])
    ap.add_argument("--limit", type=int, default=None, help="앞 N개 클립만 (시험용)")
    ap.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    ap.add_argument("--no_lpips", action="store_true", help="LPIPS 생략 (빠름)")
    ap.add_argument("--round_size", type=int, default=4,
                    help="한 라운드가 만드는 새 프레임 수 (pred_step-1)")
    return ap.parse_args()


def per_frame_metrics(pred: np.ndarray, gt: np.ndarray, device: str,
                      lpips_metric=None) -> dict:
    """한 클립·한 뷰의 프레임별 지표.

    pred, gt : (T, H, W, 3) uint8
    반환: {'psnr': (T,), 'ssim': (T,), 'lpips': (T,) 또는 None}
    """
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )

    p = torch.from_numpy(pred).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    g = torch.from_numpy(gt).permute(0, 3, 1, 2).float().div_(255.0).to(device)

    psnr = np.array([
        peak_signal_noise_ratio(p[t:t + 1], g[t:t + 1], data_range=1.0).item()
        for t in range(p.shape[0])
    ])
    ssim = structural_similarity_index_measure(
        p, g, data_range=1.0, reduction="none").detach().cpu().numpy()

    lp = None
    if lpips_metric is not None:
        with torch.no_grad():
            # 프레임마다 따로 — 배치로 넣으면 평균만 나온다
            lp = np.array([
                lpips_metric(p[t:t + 1] * 2 - 1, g[t:t + 1] * 2 - 1).item()
                for t in range(p.shape[0])
            ])
    return {"psnr": psnr, "ssim": ssim, "lpips": lp}


def summarize(curve: np.ndarray, t_sec: np.ndarray, name: str,
              higher_is_better: bool) -> dict:
    """곡선에서 읽어낼 것들: 시작/끝, 기울기, 전반/후반 비교."""
    n = len(curve)
    half = n // 2
    slope = float(np.polyfit(t_sec, curve, 1)[0])          # 초당 변화
    slope_early = float(np.polyfit(t_sec[:half], curve[:half], 1)[0])
    slope_late = float(np.polyfit(t_sec[half:], curve[half:], 1)[0])
    drop = float(curve[0] - curve[-1])
    return {
        "metric": name,
        "first_frame": float(curve[0]),
        "last_frame": float(curve[-1]),
        "delta_first_to_last": drop if higher_is_better else -drop,
        "slope_per_sec": slope,
        "slope_per_sec_first_half": slope_early,
        "slope_per_sec_second_half": slope_late,
        "acceleration": slope_late - slope_early,  # 후반이 더 가파른가
    }


def main() -> None:
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.cache), "*.npz")))
    if not files:
        sys.exit(f"캐시를 찾을 수 없습니다: {a.cache}")
    if a.limit:
        files = files[: a.limit]
    print(f"캐시 클립 {len(files)}개, 뷰 {a.views}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_metric = None
    if not a.no_lpips:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type=a.lpips_net, normalize=False, reduction="mean").to(device)

    # view -> metric -> (n_clips, T)
    acc: dict = {v: {"psnr": [], "ssim": [], "lpips": []} for v in a.views}
    skipped = 0

    for i, f in enumerate(files, 1):
        try:
            z = np.load(f)
            pred, gt = z["pred"], z["gt"]     # (3, T, H, W, 3)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"  [skip] {os.path.basename(f)}: {type(exc).__name__}")
            continue

        for v in a.views:
            m = per_frame_metrics(pred[v], gt[v], device, lpips_metric)
            acc[v]["psnr"].append(m["psnr"])
            acc[v]["ssim"].append(m["ssim"])
            if m["lpips"] is not None:
                acc[v]["lpips"].append(m["lpips"])

        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    out: dict = {
        "n_clips": len(files) - skipped,
        "skipped": skipped,
        "fps": FPS,
        "lpips_net": None if a.no_lpips else a.lpips_net,
        "views": {},
    }

    for v in a.views:
        name = VIEW_NAMES.get(v, f"view{v}")
        psnr = np.stack(acc[v]["psnr"])          # (N, T)
        ssim = np.stack(acc[v]["ssim"])
        lpips = np.stack(acc[v]["lpips"]) if acc[v]["lpips"] else None

        T = psnr.shape[1]
        t_sec = (np.arange(T) + 1) / FPS         # 프레임 t 의 경과 시간(초)

        curves = {
            "psnr_mean": psnr.mean(0), "psnr_std": psnr.std(0),
            "ssim_mean": ssim.mean(0), "ssim_std": ssim.std(0),
        }
        if lpips is not None:
            curves["lpips_mean"] = lpips.mean(0)
            curves["lpips_std"] = lpips.std(0)

        # 라운드 단위 집계 — 자기회귀 한 번이 한 라운드다
        R = a.round_size
        n_round = T // R
        rounds = {
            "psnr": [float(psnr[:, r * R:(r + 1) * R].mean()) for r in range(n_round)],
            "ssim": [float(ssim[:, r * R:(r + 1) * R].mean()) for r in range(n_round)],
        }
        if lpips is not None:
            rounds["lpips"] = [float(lpips[:, r * R:(r + 1) * R].mean())
                               for r in range(n_round)]

        summ = [
            summarize(curves["psnr_mean"], t_sec, "psnr", True),
            summarize(curves["ssim_mean"], t_sec, "ssim", True),
        ]
        if lpips is not None:
            summ.append(summarize(curves["lpips_mean"], t_sec, "lpips", False))

        out["views"][name] = {
            "n_frames": int(T),
            "seconds": float(T / FPS),
            "curves": {k: [float(x) for x in vv] for k, vv in curves.items()},
            "t_sec": [float(x) for x in t_sec],
            "per_round": rounds,
            "summary": summ,
        }

        # 화면 출력
        print(f"\n=== {name} ===")
        print(f"{'시간(초)':>9s} {'PSNR':>8s} {'SSIM':>8s}" +
              (f" {'LPIPS':>8s}" if lpips is not None else ""))
        for t in range(0, T, max(1, T // 10)):
            line = f"{t_sec[t]:9.1f} {curves['psnr_mean'][t]:8.2f} {curves['ssim_mean'][t]:8.3f}"
            if lpips is not None:
                line += f" {curves['lpips_mean'][t]:8.3f}"
            print(line)
        for s in summ:
            print(f"  {s['metric']:5s} 첫 프레임 {s['first_frame']:.3f} → "
                  f"마지막 {s['last_frame']:.3f} "
                  f"(초당 {s['slope_per_sec']:+.4f}, "
                  f"전반 {s['slope_per_sec_first_half']:+.4f} / "
                  f"후반 {s['slope_per_sec_second_half']:+.4f})")

    # 저장
    jpath = os.path.join(a.out, "drift.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    cpath = os.path.join(a.out, "drift.csv")
    with open(cpath, "w", encoding="utf-8") as f:
        cols = ["view", "frame", "t_sec", "psnr", "ssim", "lpips"]
        f.write(",".join(cols) + "\n")
        for name, d in out["views"].items():
            for t in range(d["n_frames"]):
                lp = d["curves"].get("lpips_mean", [""] * d["n_frames"])[t]
                f.write(f"{name},{t},{d['t_sec'][t]:.2f},"
                        f"{d['curves']['psnr_mean'][t]:.4f},"
                        f"{d['curves']['ssim_mean'][t]:.5f},{lp}\n")

    print(f"\n저장: {jpath}")
    print(f"      {cpath}")

    # 그림 (matplotlib 있으면)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = ["psnr", "ssim"] + ([] if a.no_lpips else ["lpips"])
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        for ax, mkey in zip(axes, metrics):
            for name, d in out["views"].items():
                y = np.array(d["curves"][f"{mkey}_mean"])
                s = np.array(d["curves"][f"{mkey}_std"])
                x = np.array(d["t_sec"])
                ax.plot(x, y, label=name)
                ax.fill_between(x, y - s, y + s, alpha=0.15)
            ax.set_xlabel("rollout time (s)")
            ax.set_ylabel(mkey.upper())
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle(f"Ctrl-World autoregressive drift (n={out['n_clips']} clips)")
        fig.tight_layout()
        ppath = os.path.join(a.out, "drift.png")
        fig.savefig(ppath, dpi=150)
        print(f"      {ppath}")
    except ImportError:
        print("  (matplotlib 없음 — 그림 생략. drift.csv 로 직접 그리세요)")


if __name__ == "__main__":
    main()
