#!/usr/bin/env python3
"""
repo/scripts/rollout_replay_traj.py 로부터 near-miss 실험용 스크립트를 생성한다.

원본을 건드리지 않고 repo/scripts/rollout_near_miss.py 를 새로 만든다.
앵커 문자열이 하나라도 안 맞으면 즉시 실패하므로, 조용히 잘못 패치될 일은 없다.

사용법:
    python near_miss/make_rollout_near_miss.py
    python near_miss/make_rollout_near_miss.py --force   # 이미 있으면 덮어쓰기
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), "repo")
SRC = os.path.join(REPO, "scripts", "rollout_replay_traj.py")
DST = os.path.join(REPO, "scripts", "rollout_near_miss.py")


# --------------------------------------------------------------------------
# 삽입할 조각들
# --------------------------------------------------------------------------

HELPER = '''

# ===================== near-miss 실험용 추가 코드 =========================
def make_near_miss(eef, axis=2, delta=0.0, ramp=5, start=0):
    """엔드이펙터 절대 좌표 궤적을 한 축으로 평행이동한다.

    eef   : (T, 7) [x, y, z, roll, pitch, yaw, gripper]  단위 m / rad
    axis  : 0=x(앞뒤) 1=y(좌우) 2=z(상하)
    delta : m 단위 오프셋. 0.0이면 원본과 완전히 동일 (= 접촉 조건)
    ramp  : 오프셋을 서서히 적용할 프레임 수. 첫 프레임이 조건 이미지와
            어긋나서 순간이동을 명령하는 상황을 막는다.
    start : 램프를 시작할 프레임 인덱스

    6번 열(그리퍼)은 절대 건드리지 않는다. 두 조건에서 열고 닫는 타이밍이
    완전히 같아야 near-miss 비교가 성립하기 때문이다.
    """
    import numpy as _np
    eef = _np.asarray(eef, dtype=_np.float64).copy()
    if delta == 0.0:
        return eef
    T = eef.shape[0]
    w = _np.clip((_np.arange(T) - start) / max(ramp, 1), 0.0, 1.0)
    eef[:, axis] = eef[:, axis] + w * delta
    return eef
# =========================================================================


class agent():'''

CLI = """    parser.add_argument('--task_type', type=str, default='replay')
    # ---- near-miss 실험 인자 ----
    parser.add_argument('--nm_axis', type=int, default=2, help='0=x 1=y 2=z')
    parser.add_argument('--nm_delta', type=float, default=0.0, help='오프셋(m). 0.0이면 접촉 조건')
    parser.add_argument('--nm_ramp', type=int, default=5)
    parser.add_argument('--nm_start', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dump_frames', action='store_true', help='예측 프레임을 npy로 저장')
    parser.add_argument('--nm_start_idx', type=int, default=None,
                        help='롤아웃 시작 프레임 강제 지정. 접촉 사건 구간을 덮으려면 필요')
    parser.add_argument('--nm_only_ids', nargs='*', default=None, help='이 클립만 실행')
    args_new = parser.parse_args()"""

PERTURB = '''        # 접촉/near-miss 두 조건에서 확산 노이즈가 같아야 비교가 성립한다
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        eef_gt, joint_pos_gt, video_dict, video_latents, instruction = Agent.get_traj_info(val_id_i, start_idx=start_idx_i, steps=int(pred_step*interact_num+8))

        # ---- near-miss 주입: 두 조건의 유일한 차이점 ----
        eef_gt = make_near_miss(eef_gt, axis=args.nm_axis, delta=args.nm_delta,
                                ramp=args.nm_ramp, start=args.nm_start)
        print(f"[near-miss] axis={args.nm_axis} delta={args.nm_delta:+.3f}m "
              f"ramp={args.nm_ramp} seed={args.seed}")
        # ------------------------------------------------

        text_i = instruction'''

STEP_SEED = """            torch.manual_seed(args.seed * 100003 + i)  # 스텝별로도 노이즈 고정
            videos_cat, true_videos, video_dict_pred, predicted_latents = Agent.forward_wm("""

DUMP = '''            his_cond.append(torch.cat([v[pred_step-1] for v in predicted_latents], dim=1).unsqueeze(0))  # (1, 4, 72, 40)
            if args.dump_frames:
                _tag = f"{val_id_i}_ax{args.nm_axis}_d{args.nm_delta:+.3f}_seed{args.seed}"
                _dump = f"{args.save_dir}/{args.task_name}/frames/{_tag}"
                os.makedirs(_dump, exist_ok=True)
                np.save(f"{_dump}/step{i:02d}.npy", np.asarray(video_dict_pred))'''

FILENAME = '''        _nmtag = f"ax{args.nm_axis}_d{args.nm_delta:+.3f}_seed{args.seed}"
        filename_video = f"{args.save_dir}/{task_name}/video/{_nmtag}_traj_{val_id_i}_{start_idx_i}_{pred_step}_{text_id}.mp4"'''


# (앵커, 대체문자열) 목록. 앵커는 원본에 정확히 1회만 나와야 한다.
LOOP = """    # ---- near-miss 실험용 재정의 ----
    if args.nm_only_ids:
        keep = [i for i, v in enumerate(args.val_id) if str(v) in [str(x) for x in args.nm_only_ids]]
        if not keep:
            raise SystemExit(f'지정한 클립이 config 목록에 없습니다: {args.nm_only_ids} (가능: {args.val_id})')
        args.val_id = [args.val_id[i] for i in keep]
        args.instruction = [args.instruction[i] for i in keep]
        args.start_idx = [args.start_idx[i] for i in keep]
    if args.nm_start_idx is not None:
        args.start_idx = [args.nm_start_idx] * len(args.val_id)
    print(f'[near-miss] 클립 {args.val_id}, 시작 프레임 {args.start_idx}')
    # --------------------------------

    for val_id_i, text_i, start_idx_i in zip(args.val_id, args.instruction, args.start_idx):"""


EDITS = [
    ("\n\nclass agent():", HELPER),
    (
        "    parser.add_argument('--task_type', type=str, default='replay')\n    args_new = parser.parse_args()",
        CLI,
    ),
    (
        "        eef_gt, joint_pos_gt, video_dict, video_latents, instruction = "
        "Agent.get_traj_info(val_id_i, start_idx=start_idx_i, steps=int(pred_step*interact_num+8))\n"
        "        text_i = instruction",
        PERTURB,
    ),
    (
        "            videos_cat, true_videos, video_dict_pred, predicted_latents = Agent.forward_wm(",
        STEP_SEED,
    ),
    (
        "            his_cond.append(torch.cat([v[pred_step-1] for v in predicted_latents], dim=1).unsqueeze(0))  # (1, 4, 72, 40)",
        DUMP,
    ),
    (
        '        filename_video = f"{args.save_dir}/{task_name}/video/time_{uuid}_traj_{val_id_i}_{start_idx_i}_{pred_step}_{text_id}.mp4"',
        FILENAME,
    ),
    ("    for val_id_i, text_i, start_idx_i in zip(args.val_id, args.instruction, args.start_idx):", LOOP),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="이미 있으면 덮어쓰기")
    args = ap.parse_args()

    if not os.path.exists(SRC):
        sys.exit(f"원본을 찾을 수 없습니다: {SRC}")
    if os.path.exists(DST) and not args.force:
        sys.exit(f"이미 존재합니다: {DST}\n  덮어쓰려면 --force")

    src = open(SRC, encoding="utf-8").read()
    out = src

    for i, (anchor, repl) in enumerate(EDITS, 1):
        n = out.count(anchor)
        if n != 1:
            sys.exit(
                f"[실패] {i}번 앵커가 {n}번 발견됨 (1번이어야 함).\n"
                f"  원본이 바뀌었을 수 있습니다. 앵커:\n  {anchor[:120]}..."
            )
        out = out.replace(anchor, repl)
        print(f"  [{i}/{len(EDITS)}] 패치 적용")

    header = (
        "# 이 파일은 near_miss/make_rollout_near_miss.py 가 생성했습니다.\n"
        "# 원본: scripts/rollout_replay_traj.py (Ctrl-World, ICLR 2026)\n"
        "# 직접 수정하지 말고 생성 스크립트를 고친 뒤 --force 로 다시 만드세요.\n"
    )
    open(DST, "w", encoding="utf-8").write(header + out)

    # 문법 검사
    import py_compile

    try:
        py_compile.compile(DST, doraise=True)
    except py_compile.PyCompileError as e:
        sys.exit(f"[실패] 생성된 파일 문법 오류:\n{e}")

    print(f"\n생성 완료: {DST}")
    print("\n실행 예시 (접촉 조건 / near-miss 조건 한 쌍):")
    print("  cd repo")
    print("  # 접촉 (원본 궤적 그대로)")
    print("  CUDA_VISIBLE_DEVICES=0 python scripts/rollout_near_miss.py \\")
    print("    --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \\")
    print("    --dataset_names droid_subset --svd_model_path $SVD --clip_model_path $CLIP \\")
    print("    --ckpt_path $CKPT --nm_delta 0.0 --seed 0 --dump_frames")
    print("  # near-miss (z축 +4cm)")
    print("  CUDA_VISIBLE_DEVICES=0 python scripts/rollout_near_miss.py \\")
    print("    --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \\")
    print("    --dataset_names droid_subset --svd_model_path $SVD --clip_model_path $CLIP \\")
    print("    --ckpt_path $CKPT --nm_axis 2 --nm_delta 0.04 --seed 0 --dump_frames")


if __name__ == "__main__":
    main()
