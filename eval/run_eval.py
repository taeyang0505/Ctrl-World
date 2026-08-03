#!/usr/bin/env python3
"""
Ctrl-World Table 1 재현 — 검증 클립을 롤아웃하며 예측/정답 프레임을 모아 지표를 계산한다.

원본 `scripts/rollout_replay_traj.py` 는 궤적 하나를 돌려 비교 영상 mp4를
저장할 뿐, 지표 계산에 필요한 배열을 남기지 않는다. 이 스크립트는 그 안의
`agent` 클래스를 그대로 재사용하면서 여러 클립을 돌고 결과를 누적한다.

논문 Table 1 (검증 클립 256개, 10초):
    PSNR 23.56 / SSIM 0.828 / LPIPS 0.091 / FID 25.00 / FVD 97.4

사용법:
    python eval/run_eval.py \
        --dataset_root_path ~/data/droid_val_processed \
        --dataset_names droid \
        --svd_model_path  $SVD --clip_model_path $CLIP --ckpt_path $CKPT \
        --episode_list ~/data/droid_val/selected_episodes.json \
        --out results/table1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


# ---------------------------------------------------------------------------
# 설계 결정 (근거는 eval/README.md 참고)
# ---------------------------------------------------------------------------
VIEW_THIRD = 0   # exterior_1_left — Table 1의 third-view
VIEW_WRIST = 2   # wrist_left      — Table 2의 wrist-view 비교용


def parse_args():
    ap = argparse.ArgumentParser()
    # 원본 스크립트와 같은 인자
    ap.add_argument("--svd_model_path", required=True)
    ap.add_argument("--clip_model_path", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--dataset_root_path", required=True)
    ap.add_argument("--dataset_meta_info_path", default=None,
                    help="기본값: repo/dataset_meta_info (정규화 통계는 반드시 원본을 쓸 것)")
    ap.add_argument("--dataset_names", default="droid")
    # 평가 설정
    ap.add_argument("--episode_list", default=None,
                    help="selected_episodes.json 경로. 없으면 --val_ids 사용")
    ap.add_argument("--val_ids", nargs="*", default=None, help="직접 지정할 클립 id")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개만 (시험용)")
    ap.add_argument("--start_idx", type=int, default=8,
                    help="궤적에서 롤아웃을 시작할 프레임 (원본 config 기본값 8)")
    ap.add_argument("--interact_num", type=int, default=13,
                    help="라운드 수. 13이면 52프레임 생성 → 앞 50프레임(10.0초)만 사용")
    ap.add_argument("--max_frames", type=int, default=50,
                    help="지표에 쓸 프레임 수. 5Hz 기준 50프레임 = 10.0초")
    ap.add_argument("--gt_source", choices=["raw", "vae"], default="raw",
                    help="정답 프레임 기준. raw=원본 mp4, vae=VAE 왕복본(모델 상한)")
    ap.add_argument("--views", nargs="*", type=int, default=[VIEW_THIRD],
                    help=f"평가할 카메라. 기본 {VIEW_THIRD}(third-view)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    ap.add_argument("--no_fid", action="store_true", help="FID 생략 (빠른 확인용)")
    ap.add_argument("--out", default="results/table1")
    ap.add_argument("--cache_dir", default=None,
                    help="클립별 예측/정답 배열 저장 위치. 기본 <out>/cache. "
                         "재실행 시 캐시가 있으면 롤아웃을 건너뛴다")
    ap.add_argument("--no_cache", action="store_true",
                    help="캐시를 쓰지 않는다 (디스크 절약, 재실행 시 전부 다시 롤아웃)")
    return ap.parse_args()


def build_agent(a):
    """원본 config + agent 를 그대로 사용한다."""
    from config import wm_args
    from rollout_replay_traj import agent as Agent

    args = wm_args(task_type="replay")
    args.svd_model_path = os.path.expanduser(a.svd_model_path)
    args.clip_model_path = os.path.expanduser(a.clip_model_path)
    args.ckpt_path = os.path.expanduser(a.ckpt_path)
    args.val_model_path = args.ckpt_path          # 원본은 이 이름으로 체크포인트를 읽는다
    args.dataset_root_path = os.path.expanduser(a.dataset_root_path)
    args.dataset_meta_info_path = os.path.expanduser(
        a.dataset_meta_info_path or os.path.join(REPO, "dataset_meta_info"))
    args.dataset_names = a.dataset_names
    args.val_dataset_dir = args.dataset_root_path
    args.interact_num = a.interact_num

    # config.py 의 data_stat_path 는 'dataset_meta_info/droid/stat.json' 같은 상대 경로라
    # 실행 위치에 따라 깨진다. 절대 경로로 바꿔 준다.
    # 이 통계(state_p01/p99)는 액션 정규화에 쓰이므로 반드시 원본 값을 써야 한다.
    args.data_stat_path = os.path.join(
        args.dataset_meta_info_path, a.dataset_names, "stat.json")
    if not os.path.exists(args.data_stat_path):
        sys.exit(f"정규화 통계를 찾을 수 없습니다: {args.data_stat_path}\n"
                 f"  --dataset_meta_info_path 와 --dataset_names 를 확인하세요.")
    print(f"정규화 통계: {args.data_stat_path}")

    return Agent(args), args


def rollout_one(Agent, args, val_id: str, start_idx: int, a) -> dict | None:
    """궤적 하나를 롤아웃하고 예측/정답 프레임을 모은다.

    원본 rollout_replay_traj.py 의 메인 루프와 같은 구조이나,
    영상을 저장하는 대신 배열을 반환한다.
    """
    pred_step = args.pred_step
    interact_num = args.interact_num

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    try:
        eef_gt, joint_gt, video_dict, video_latents, instruction = Agent.get_traj_info(
            val_id, start_idx=start_idx, steps=int(pred_step * interact_num + 8)
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {val_id}: {type(exc).__name__} {str(exc)[:80]}")
        return None

    # --- 히스토리 버퍼 초기화 (원본과 동일) ---
    his_cond, his_eef = [], []
    first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)
    for _ in range(args.num_history * 4):
        his_cond.append(first_latent)
        his_eef.append(eef_gt[0:1])

    preds, gts = [], []          # 라운드별 (3,T,H,W,3)
    video_dict_pred = None

    for i in range(interact_num):
        start_id = int(i * (pred_step - 1))
        end_id = start_id + pred_step
        video_latent_true = [v[start_id:end_id] for v in video_latents]
        cartesian_pose = eef_gt[start_id:end_id]

        history_idx = [0, 0, -8, -6, -4, -2]
        his_pose = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)
        action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
        his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
        current_latent = his_cond[-1]

        torch.manual_seed(a.seed * 100003 + i)   # 확산 노이즈 고정
        _, true_videos, video_dict_pred, predicted_latents = Agent.forward_wm(
            action_cond, video_latent_true, current_latent,
            his_cond=his_cond_input,
            text=instruction if args.text_cond else None,
        )

        # 정답 프레임: raw = 원본 mp4 디코드, vae = VAE 왕복본
        if a.gt_source == "raw":
            gt_round = np.stack([v[start_id:end_id] for v in video_dict], axis=0)
        else:
            gt_round = np.asarray(true_videos)

        pred_round = np.asarray(video_dict_pred)

        # 각 라운드의 첫 프레임은 조건으로 준 프레임이라 예측이 아니다 → 제외
        preds.append(pred_round[:, 1:])
        gts.append(gt_round[:, 1:])

        his_eef.append(cartesian_pose[pred_step - 1 : pred_step])
        his_cond.append(
            torch.cat([v[pred_step - 1] for v in predicted_latents], dim=1).unsqueeze(0)
        )

    pred = np.concatenate(preds, axis=1)   # (3, 4*interact_num, H, W, 3)
    gt = np.concatenate(gts, axis=1)

    if a.max_frames and pred.shape[1] > a.max_frames:
        pred = pred[:, : a.max_frames]
        gt = gt[:, : a.max_frames]

    return {"pred": pred, "gt": gt, "instruction": instruction, "n_frames": pred.shape[1]}


def main() -> None:
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)

    # 평가할 클립 목록
    if a.episode_list:
        with open(os.path.expanduser(a.episode_list), encoding="utf-8") as f:
            ids = [str(i) for i in json.load(f)["episode_ids"]]
    elif a.val_ids:
        ids = [str(i) for i in a.val_ids]
    else:
        sys.exit("--episode_list 또는 --val_ids 중 하나가 필요합니다")
    if a.limit:
        ids = ids[: a.limit]
    print(f"평가 클립 {len(ids)}개, 카메라 {a.views}, 정답기준={a.gt_source}, seed={a.seed}")

    Agent, args = build_agent(a)
    print(f"설정: pred_step={args.pred_step} interact_num={args.interact_num} "
          f"num_history={args.num_history} → 최대 {(args.pred_step-1)*args.interact_num}프레임 "
          f"({(args.pred_step-1)*args.interact_num*0.2:.1f}초)")

    # 롤아웃 결과는 클립마다 즉시 디스크에 저장한다.
    # 지표 계산에서 실패해도 몇 시간짜리 롤아웃을 잃지 않기 위해서다.
    cache_dir = None
    if not a.no_cache:
        cache_dir = os.path.expanduser(a.cache_dir or os.path.join(a.out, "cache"))
        os.makedirs(cache_dir, exist_ok=True)
        print(f"캐시: {cache_dir} (재실행 시 이미 있는 클립은 롤아웃을 건너뜁니다)")

    per_view = {v: {"pred": [], "gt": []} for v in a.views}
    ok, cached, t0 = 0, 0, time.time()

    for n, vid in enumerate(ids, 1):
        cpath = os.path.join(cache_dir, f"{vid}.npz") if cache_dir else None

        r = None
        if cpath and os.path.exists(cpath):
            try:
                z = np.load(cpath)
                r = {"pred": z["pred"], "gt": z["gt"], "n_frames": int(z["pred"].shape[1])}
                cached += 1
            except Exception:  # noqa: BLE001
                r = None  # 손상된 캐시는 무시하고 다시 롤아웃

        if r is None:
            r = rollout_one(Agent, args, vid, a.start_idx, a)
            if r is None:
                continue
            if cpath:
                np.savez_compressed(cpath, pred=r["pred"], gt=r["gt"])

        ok += 1
        for v in a.views:
            per_view[v]["pred"].append(r["pred"][v])
            per_view[v]["gt"].append(r["gt"][v])
        el = time.time() - t0
        tag = "캐시" if cpath and os.path.exists(cpath) and cached else ""
        print(f"  [{n}/{len(ids)}] {vid}  {r['n_frames']}프레임 {tag} "
              f"경과 {el/60:.1f}분  남은 예상 {(el/n)*(len(ids)-n)/60:.1f}분")

    if ok == 0:
        sys.exit("성공한 클립이 없습니다.")
    if cached:
        print(f"\n캐시에서 불러온 클립: {cached}/{ok}")

    from metrics import compute_psnr_ssim, compute_lpips, compute_fid

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_json = os.path.join(a.out, "table1.json")

    def save_partial(res: dict) -> None:
        """지표를 하나 계산할 때마다 저장한다. 뒤에서 실패해도 앞의 결과는 남는다."""
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"진행중": True, "results": res}, f, indent=2, ensure_ascii=False)

    results = {}
    for v in a.views:
        pred = np.stack(per_view[v]["pred"], axis=0)   # (N,T,H,W,3)
        gt = np.stack(per_view[v]["gt"], axis=0)
        name = {0: "third_view(exterior_1)", 1: "exterior_2", 2: "wrist_view"}.get(v, f"view{v}")
        print(f"\n지표 계산 — {name}  shape={pred.shape}")
        r: dict = {}

        # 지표마다 따로 감싼다. 하나가 실패해도 나머지는 살린다.
        for label, fn in (
            ("PSNR/SSIM", lambda: compute_psnr_ssim(pred, gt, device=dev)),
            ("LPIPS", lambda: compute_lpips(pred, gt, net=a.lpips_net, device=dev)),
        ):
            try:
                r.update(fn())
                print(f"  {label} 완료")
            except Exception as exc:  # noqa: BLE001
                r[f"{label}_error"] = f"{type(exc).__name__}: {exc}"
                print(f"  !! {label} 실패: {type(exc).__name__}: {str(exc)[:120]}")
            results[name] = r
            save_partial(results)

        if not a.no_fid:
            try:
                r.update(compute_fid(pred, gt, device=dev))
                print("  FID 완료")
            except Exception as exc:  # noqa: BLE001
                r["fid_error"] = f"{type(exc).__name__}: {exc}"
                print(f"  !! FID 실패: {type(exc).__name__}: {str(exc)[:160]}")
                print("     (torch-fidelity 가 필요하면: pip install torch-fidelity)")
            results[name] = r
            save_partial(results)

    summary = {
        "n_clips_requested": len(ids),
        "n_clips_ok": ok,
        "frames_per_clip": int(pred.shape[1]),
        "seconds_per_clip": round(pred.shape[1] * 0.2, 2),
        "gt_source": a.gt_source,
        "lpips_net": a.lpips_net,
        "seed": a.seed,
        "start_idx": a.start_idx,
        "interact_num": args.interact_num,
        "paper_table1": {"psnr": 23.56, "ssim": 0.828, "lpips": 0.091,
                         "fid": 25.00, "fvd": 97.4},
        "n_clips_from_cache": cached,
        "cache_dir": cache_dir,
        "results": results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    def fmt(x, w=7, p=2):
        return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) else f"{'—':>{w}s}"

    print("\n" + "=" * 62)
    print(f"{'':22s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s} {'FID':>7s}")
    for name, r in results.items():
        print(f"{name:22s} {fmt(r.get('psnr'))} {fmt(r.get('ssim'),7,3)} "
              f"{fmt(r.get('lpips'),7,3)} {fmt(r.get('fid'))}")
    p = summary["paper_table1"]
    print(f"{'논문 Table 1':22s} {p['psnr']:7.2f} {p['ssim']:7.3f} {p['lpips']:7.3f} {p['fid']:7.2f}")
    print("=" * 62)
    print(f"클립 {ok}/{len(ids)}개 · 클립당 {summary['frames_per_clip']}프레임 "
          f"({summary['seconds_per_clip']}초) · 저장: {out_json}")
    if cache_dir:
        print(f"캐시 유지: {cache_dir}  (같은 명령을 다시 실행하면 롤아웃 없이 지표만 재계산)")


if __name__ == "__main__":
    main()
