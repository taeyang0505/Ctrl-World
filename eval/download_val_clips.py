#!/usr/bin/env python3
"""
DROID에서 검증용 에피소드만 골라 받는다.

논문은 검증 클립 256개로 Table 1을 측정했다. DROID 전체는 370GB지만
HuggingFace에 에피소드 단위 파일로 올라가 있어서, 검증 분할에 해당하는
것만 받으면 된다.

  검증 분할 규칙: traj_id % 100 == 99   (extract_latent.py:42)
  전체 95,600개 중 956개가 검증
  에피소드 1개 = parquet 1 + mp4 3 ≈ 3.2MB
  → 256개면 약 0.8GB

사용법:
    python eval/download_val_clips.py --out ~/data/droid_val --n 256
    python eval/download_val_clips.py --out ~/data/droid_val --n 256 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ID = "cadene/droid_1.0.1"
CAMERAS = [
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
    "observation.images.wrist_left",
]


def episode_files(ep: int) -> list[str]:
    """에피소드 하나에 필요한 파일 경로들. chunk 크기는 1000 (meta/info.json)."""
    chunk = ep // 1000
    files = [f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"]
    files += [
        f"videos/chunk-{chunk:03d}/{cam}/episode_{ep:06d}.mp4" for cam in CAMERAS
    ]
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="저장 경로")
    ap.add_argument("--n", type=int, default=256, help="받을 검증 클립 수")
    ap.add_argument("--min-length", type=int, default=160,
                    help="너무 짧은 궤적 제외 (10초 롤아웃에 최소 프레임 필요)")
    ap.add_argument("--seed", type=int, default=0, help="256개를 고르는 시드")
    ap.add_argument("--dry-run", action="store_true", help="목록만 출력")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub이 필요합니다:  pip install huggingface_hub")

    out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out, exist_ok=True)

    # 1) 메타데이터 — 전체 목록과 길이를 여기서 얻는다 (8.7MB, 가볍다)
    print("meta/episodes.jsonl 내려받는 중...")
    meta_path = hf_hub_download(
        REPO_ID, "meta/episodes.jsonl", repo_type="dataset",
        local_dir=out, local_dir_use_symlinks=False,
    )
    hf_hub_download(
        REPO_ID, "meta/info.json", repo_type="dataset",
        local_dir=out, local_dir_use_symlinks=False,
    )

    episodes = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    print(f"  전체 에피소드: {len(episodes)}")

    # 2) 검증 분할 + 길이 필터
    val = [e for e in episodes if e["episode_index"] % 100 == 99]
    print(f"  검증 분할(%100==99): {len(val)}")

    long_enough = [e for e in val if e.get("length", 0) >= args.min_length]
    print(f"  길이 {args.min_length} 이상: {len(long_enough)}")

    if len(long_enough) < args.n:
        print(f"  ! 요청 {args.n}개보다 적습니다. 있는 만큼만 받습니다.")

    # 3) 고정 시드로 n개 선택 — 목록을 저장해 두어야 재현이 된다
    import random
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(long_enough, min(args.n, len(long_enough))),
                    key=lambda e: e["episode_index"])
    ids = [e["episode_index"] for e in chosen]

    sel_path = os.path.join(out, "selected_episodes.json")
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump({
            "repo_id": REPO_ID,
            "rule": "episode_index % 100 == 99",
            "min_length": args.min_length,
            "seed": args.seed,
            "n": len(ids),
            "episode_ids": ids,
            "lengths": {str(e["episode_index"]): e.get("length") for e in chosen},
        }, f, indent=2)
    print(f"  선택 목록 저장: {sel_path}")

    if args.dry_run:
        print(f"\n[dry-run] 받을 에피소드 {len(ids)}개, 파일 {len(ids)*4}개")
        print(f"  예상 용량: 약 {len(ids)*3.2/1024:.2f} GB")
        print(f"  앞 10개: {ids[:10]}")
        return

    # 4) 다운로드
    print(f"\n{len(ids)}개 에피소드 다운로드 시작 (파일 {len(ids)*4}개)")
    failed = []
    for i, ep in enumerate(ids, 1):
        for rel in episode_files(ep):
            try:
                hf_hub_download(REPO_ID, rel, repo_type="dataset",
                                local_dir=out, local_dir_use_symlinks=False)
            except Exception as exc:  # noqa: BLE001
                failed.append((ep, rel, str(exc)[:80]))
        if i % 10 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)}")

    if failed:
        print(f"\n실패 {len(failed)}건:")
        for ep, rel, err in failed[:10]:
            print(f"  ep {ep}  {rel}  — {err}")
    else:
        print("\n전부 성공")

    print(f"\n저장 위치: {out}")
    print("다음 단계: eval/patch_extract_latent.py 로 전처리 스크립트를 준비한 뒤")
    print("  accelerate launch repo/dataset_example/extract_latent_val.py \\")
    print(f"      --droid_hf_path {out} --droid_output_path <출력> --svd_path <SVD>")


if __name__ == "__main__":
    main()
