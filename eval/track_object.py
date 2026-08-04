#!/usr/bin/env python3
"""
생성 영상에서 물체가 움직였는지 픽셀 단위로 판정한다.

near-miss 실험의 핵심 판정 도구다. "닿지 않았는데 물체가 반응했는가"를
사람 눈이 아니라 숫자로 답하기 위해 만들었다. 눈으로는 압축 아티팩트와
착각 때문에 1~2픽셀 흔들림을 '움직였다'로 오판하기 쉽다.

DROID 는 실제 녹화라 물체 위치 정답이 없다. 그래서 색으로 물체를 찾는다.
대상 물체가 배경과 색이 뚜렷이 구분될 때만 신뢰할 수 있다.

사용법:
    # 노란 머그 추적 (기본 프리셋)
    python eval/track_object.py --video rollout.mp4 --preset yellow

    # 색 범위를 직접 지정
    python eval/track_object.py --video rollout.mp4 \
        --rgb_min 110 110 0 --rgb_max 255 255 95

    # 여러 영상을 한 번에 비교 (near-miss 쌍 비교용)
    python eval/track_object.py --video contact.mp4 near_miss.mp4 --preset yellow
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

# 색 프리셋 — (rgb_min, rgb_max, 추가 조건)
PRESETS = {
    "yellow": ((110, 110, 0), (255, 255, 95), "r+g-2b>90"),
    "green": ((0, 90, 0), (110, 255, 110), "g-r>25"),
    "red": ((110, 0, 0), (255, 100, 100), "r-g>50"),
    "blue": ((0, 0, 100), (110, 130, 255), "b-r>40"),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", nargs="+", required=True)
    ap.add_argument("--preset", default="yellow", choices=list(PRESETS))
    ap.add_argument("--rgb_min", nargs=3, type=int, default=None)
    ap.add_argument("--rgb_max", nargs=3, type=int, default=None)
    ap.add_argument("--view", type=int, default=0,
                    help="가로로 이어붙은 뷰 중 몇 번째 (0=exterior_1)")
    ap.add_argument("--n_views", type=int, default=3)
    ap.add_argument("--pred_row", default="bottom", choices=["bottom", "top", "full"],
                    help="영상이 위=정답/아래=예측 구조인 경우 어느 쪽을 볼지")
    ap.add_argument("--min_pixels", type=int, default=20,
                    help="이 미만이면 검출 실패로 본다")
    ap.add_argument("--move_thresh", type=float, default=3.0,
                    help="이 픽셀 이상 움직이면 '반응했다'로 판정")
    ap.add_argument("--out", default=None, help="결과 JSON 경로")
    return ap.parse_args()


def read_frames(path: str, view: int, n_views: int, pred_row: str) -> np.ndarray:
    """영상에서 지정한 뷰·행만 잘라 (T, H, W, 3) uint8 로 읽는다."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    if probe.returncode != 0:
        sys.exit(f"영상을 읽을 수 없습니다: {path}")
    W, H = (int(x) for x in probe.stdout.strip().split(","))

    vw = W // n_views
    vh = H // 2 if pred_row in ("bottom", "top") else H
    vy = vh if pred_row == "bottom" else 0
    vx = view * vw

    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-vf", f"crop={vw}:{vh}:{vx}:{vy}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    if not raw:
        sys.exit(f"프레임을 뽑지 못했습니다: {path}")
    return np.frombuffer(raw, np.uint8).reshape(-1, vh, vw, 3)


def find_object(img: np.ndarray, lo, hi, extra: str | None) -> tuple:
    """색으로 물체를 찾아 (중심, 픽셀수) 반환."""
    r, g, b = (img[..., i].astype(np.int16) for i in range(3))
    m = ((r >= lo[0]) & (r <= hi[0]) &
         (g >= lo[1]) & (g <= hi[1]) &
         (b >= lo[2]) & (b <= hi[2]))
    if extra == "r+g-2b>90":
        m &= (r + g - 2 * b) > 90
    elif extra == "g-r>25":
        m &= (g - r) > 25
    elif extra == "r-g>50":
        m &= (r - g) > 50
    elif extra == "b-r>40":
        m &= (b - r) > 40

    n = int(m.sum())
    if n == 0:
        return None, 0
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean())), n


def track(path: str, a) -> dict:
    lo, hi, extra = PRESETS[a.preset]
    if a.rgb_min:
        lo, extra = tuple(a.rgb_min), None
    if a.rgb_max:
        hi = tuple(a.rgb_max)

    frames = read_frames(path, a.view, a.n_views, a.pred_row)
    pts, areas = [], []
    for i in range(len(frames)):
        c, n = find_object(frames[i], lo, hi, extra)
        pts.append(c if (c and n >= a.min_pixels) else None)
        areas.append(n)

    det = [(i, c) for i, c in enumerate(pts) if c]
    if len(det) < 2:
        return {"video": os.path.basename(path), "error": "물체 검출 실패",
                "n_frames": len(frames), "max_area": max(areas) if areas else 0}

    x0, y0 = det[0][1]
    disp = [float(np.hypot(c[0] - x0, c[1] - y0)) for _, c in det]
    max_disp = max(disp)

    return {
        "video": os.path.basename(path),
        "n_frames": len(frames),
        "n_detected": len(det),
        "start_xy": [round(x0, 2), round(y0, 2)],
        "end_xy": [round(det[-1][1][0], 2), round(det[-1][1][1], 2)],
        "max_disp_px": round(max_disp, 2),
        "end_disp_px": round(disp[-1], 2),
        "area_first": areas[det[0][0]],
        "area_last": areas[det[-1][0]],
        "moved": bool(max_disp >= a.move_thresh),
        "trajectory": [[i, round(c[0], 1), round(c[1], 1)] for i, c in det],
    }


def main() -> None:
    a = parse_args()
    results = [track(v, a) for v in a.video]

    print(f"{'영상':46s} {'검출':>8s} {'최대이동':>9s} {'끝이동':>8s} {'판정':>8s}")
    for r in results:
        if "error" in r:
            print(f"{r['video'][:46]:46s} {'실패':>8s}  ({r['error']}, 최대 픽셀 {r['max_area']})")
            continue
        verdict = "반응함" if r["moved"] else "정지"
        print(f"{r['video'][:46]:46s} {r['n_detected']:4d}/{r['n_frames']:<3d} "
              f"{r['max_disp_px']:9.2f} {r['end_disp_px']:8.2f} {verdict:>8s}")

    print(f"\n판정 기준: 최대 이동 {a.move_thresh}px 이상이면 '반응함'")
    print("주의: 색 기반 검출이라 대상 물체가 배경과 뚜렷이 구분될 때만 신뢰할 수 있다.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
