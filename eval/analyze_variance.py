#!/usr/bin/env python3
"""
클립별 품질 편차의 원인을 찾는다.

드리프트 분석에서 손목캠의 분산이 3인칭의 2~3배로 나타났다. 평균이
조금 나쁜 것이 아니라 **어떤 클립은 멀쩡하고 어떤 클립은 붕괴한다**는
뜻이다. 그 차이가 무엇인지 밝히는 것이 이 스크립트의 목적이다.

우리 연구 가설과의 연결: 접촉·조작이 활발한 장면에서 더 무너진다면,
"접촉 경계에서 월드모델이 약하다"는 문제 제기가 자체 실측으로 뒷받침된다.
아니라면 가설을 수정해야 한다.

측정하는 장면 특성 (전부 어노테이션과 정답 영상에서 직접 계산):
  - eef_path      : 엔드이펙터 이동 거리 총합 (m). 팔이 얼마나 움직였나
  - gripper_tv    : 그리퍼 개폐량 총합. 집고 놓는 동작이 얼마나 있었나
  - gt_motion     : 정답 영상의 프레임 간 변화량. 장면이 얼마나 역동적인가
  - z_min         : 엔드이펙터 최저 높이 (m). 낮을수록 테이블 접촉에 가깝다

사용법:
    python eval/analyze_variance.py \
        --cache results/table1/cache \
        --dataset ~/data/droid_val_processed \
        --out results/variance
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
FPS = 5.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", required=True, help="전처리된 데이터 폴더 (annotation/val/*.json)")
    ap.add_argument("--out", default="results/variance")
    ap.add_argument("--views", nargs="*", type=int, default=[0, 2])
    ap.add_argument("--rank_view", type=int, default=2, help="어느 뷰 기준으로 순위를 매길지")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lpips_net", default="alex")
    ap.add_argument("--tail", type=int, default=10, help="'후반' 으로 볼 마지막 프레임 수")
    ap.add_argument("--top_k", type=int, default=25, help="최악·최선 각각 몇 개를 뽑을지")
    ap.add_argument("--start_idx", type=int, default=8,
                    help="롤아웃 시작 프레임 (run_eval 과 같아야 함)")
    return ap.parse_args()


def clip_quality(pred: np.ndarray, gt: np.ndarray, device: str, lpips_metric,
                 tail: int) -> dict:
    """한 클립·한 뷰의 품질 요약."""
    from torchmetrics.functional.image import peak_signal_noise_ratio

    p = torch.from_numpy(pred).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    g = torch.from_numpy(gt).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    T = p.shape[0]

    psnr = np.array([peak_signal_noise_ratio(p[t:t+1], g[t:t+1], data_range=1.0).item()
                     for t in range(T)])
    with torch.no_grad():
        lp = np.array([lpips_metric(p[t:t+1]*2-1, g[t:t+1]*2-1).item() for t in range(T)])

    return {
        "psnr_mean": float(psnr.mean()),
        "psnr_head": float(psnr[:5].mean()),
        "psnr_tail": float(psnr[-tail:].mean()),
        "psnr_drop": float(psnr[:5].mean() - psnr[-tail:].mean()),
        "lpips_mean": float(lp.mean()),
        "lpips_head": float(lp[:5].mean()),
        "lpips_tail": float(lp[-tail:].mean()),
        "lpips_rise": float(lp[-tail:].mean() - lp[:5].mean()),
    }


def scene_stats(anno: dict, gt_third: np.ndarray, gt_wrist: np.ndarray,
                start: int, n: int) -> dict:
    """롤아웃 구간의 장면 특성. 어노테이션과 정답 영상에서 직접 계산한다."""
    states = np.asarray(anno["states"], dtype=np.float64)   # (T, 7)
    seg = states[start:start + n + 1]
    if len(seg) < 2:
        seg = states[-2:]

    xyz = seg[:, :3]
    eef_path = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    gripper = seg[:, 6]
    gripper_tv = float(np.abs(np.diff(gripper)).sum())      # 개폐량 총합
    z_min = float(xyz[:, 2].min())

    def motion(v: np.ndarray) -> float:
        d = np.abs(np.diff(v.astype(np.int16), axis=0)).mean()
        return float(d)

    return {
        "eef_path": eef_path,
        "gripper_tv": gripper_tv,
        "z_min": z_min,
        "gt_motion_third": motion(gt_third),
        "gt_motion_wrist": motion(gt_wrist),
        "instruction": (anno.get("texts") or [""])[0],
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """순위 상관. 비선형 관계도 잡고 이상치에 덜 민감하다."""
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x, y).statistic
        return float(r) if np.isfinite(r) else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def main() -> None:
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.cache), "*.npz")))
    if not files:
        sys.exit(f"캐시가 없습니다: {a.cache}")
    if a.limit:
        files = files[: a.limit]

    anno_dir = os.path.join(os.path.expanduser(a.dataset), "annotation", "val")
    if not os.path.isdir(anno_dir):
        sys.exit(f"어노테이션 폴더가 없습니다: {anno_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type=a.lpips_net, normalize=False, reduction="mean").to(device)

    print(f"클립 {len(files)}개 분석 중...")
    rows = []
    for i, f in enumerate(files, 1):
        cid = os.path.splitext(os.path.basename(f))[0]
        try:
            z = np.load(f)
            pred, gt = z["pred"], z["gt"]
        except Exception:  # noqa: BLE001
            continue

        apath = os.path.join(anno_dir, f"{cid}.json")
        if not os.path.exists(apath):
            continue
        with open(apath, encoding="utf-8") as fh:
            anno = json.load(fh)

        row = {"clip": cid}
        for v in a.views:
            q = clip_quality(pred[v], gt[v], device, lpips_metric, a.tail)
            row.update({f"{VIEW_NAMES.get(v, f'v{v}')}_{k}": val for k, val in q.items()})
        row.update(scene_stats(anno, gt[0], gt[2], a.start_idx, pred.shape[1]))
        rows.append(row)

        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    if not rows:
        sys.exit("분석할 클립이 없습니다.")

    rank_name = VIEW_NAMES.get(a.rank_view, f"v{a.rank_view}")
    key = f"{rank_name}_lpips_tail"
    rows.sort(key=lambda r: r[key])         # 낮을수록 좋음

    # ---- 상관 분석 ----
    factors = ["eef_path", "gripper_tv", "z_min", "gt_motion_third", "gt_motion_wrist"]
    targets = [f"{VIEW_NAMES.get(v, f'v{v}')}_{m}"
               for v in a.views for m in ("lpips_tail", "psnr_drop")]

    corr = {}
    for t in targets:
        y = np.array([r[t] for r in rows])
        corr[t] = {fa: spearman(np.array([r[fa] for r in rows]), y) for fa in factors}

    # ---- 최악 / 최선 ----
    k = min(a.top_k, len(rows) // 3)
    best, worst = rows[:k], rows[-k:]

    def group_mean(g, fld):
        return float(np.mean([r[fld] for r in g]))

    comparison = {}
    for fa in factors + [f"{rank_name}_lpips_tail", f"{rank_name}_psnr_tail"]:
        b, w = group_mean(best, fa), group_mean(worst, fa)
        comparison[fa] = {"best_k": b, "worst_k": w,
                          "ratio": (w / b) if b not in (0.0,) else float("nan"),
                          "diff": w - b}

    out = {
        "n_clips": len(rows),
        "rank_view": rank_name,
        "rank_key": key,
        "top_k": k,
        "spearman_correlation": corr,
        "best_vs_worst": comparison,
        "best_clips": [{"clip": r["clip"], key: r[key], "instruction": r["instruction"],
                        "gripper_tv": r["gripper_tv"], "eef_path": r["eef_path"]}
                       for r in best],
        "worst_clips": [{"clip": r["clip"], key: r[key], "instruction": r["instruction"],
                         "gripper_tv": r["gripper_tv"], "eef_path": r["eef_path"]}
                        for r in worst],
        "rows": rows,
    }

    with open(os.path.join(a.out, "variance.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    cols = ["clip"] + [c for c in rows[0] if c != "clip"]
    with open(os.path.join(a.out, "variance.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]).replace(",", ";") for c in cols) + "\n")

    # ---- 출력 ----
    print(f"\n{'='*70}")
    print(f"순위 기준: {key} (낮을수록 좋음), 클립 {len(rows)}개")
    print(f"{'='*70}")

    print(f"\n[1] 장면 특성과 품질 저하의 순위 상관 (Spearman)")
    print(f"{'':22s}" + "".join(f"{fa:>17s}" for fa in factors))
    for t in targets:
        print(f"{t:22s}" + "".join(f"{corr[t][fa]:17.3f}" for fa in factors))
    print("  * 양수 = 그 특성이 클수록 품질이 나쁨(lpips_tail↑ / psnr_drop↑)")

    print(f"\n[2] 최선 {k}개 vs 최악 {k}개")
    print(f"{'특성':22s} {'최선':>12s} {'최악':>12s} {'배율':>8s}")
    for fa, d in comparison.items():
        print(f"{fa:22s} {d['best_k']:12.4f} {d['worst_k']:12.4f} {d['ratio']:8.2f}")

    print(f"\n[3] 최악 클립 (위 5개)")
    for r in worst[-5:][::-1]:
        print(f"  {r['clip']:8s} {key}={r[key]:.3f} "
              f"grip_tv={r['gripper_tv']:.2f} path={r['eef_path']:.2f}m  "
              f"\"{r['instruction'][:50]}\"")
    print(f"\n[4] 최선 클립 (위 5개)")
    for r in best[:5]:
        print(f"  {r['clip']:8s} {key}={r[key]:.3f} "
              f"grip_tv={r['gripper_tv']:.2f} path={r['eef_path']:.2f}m  "
              f"\"{r['instruction'][:50]}\"")

    print(f"\n저장: {a.out}/variance.json, variance.csv")

    # ---- 산점도 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(factors), figsize=(4 * len(factors), 3.6))
        y = np.array([r[key] for r in rows])
        for ax, fa in zip(axes, factors):
            x = np.array([r[fa] for r in rows])
            ax.scatter(x, y, s=12, alpha=0.5)
            ax.set_xlabel(fa)
            ax.set_ylabel(key)
            ax.set_title(f"ρ = {corr[key][fa]:.3f}" if key in corr else fa)
            ax.grid(alpha=0.3)
        fig.suptitle(f"Scene factors vs quality degradation (n={len(rows)})")
        fig.tight_layout()
        p = os.path.join(a.out, "variance.png")
        fig.savefig(p, dpi=150)
        print(f"      {p}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
