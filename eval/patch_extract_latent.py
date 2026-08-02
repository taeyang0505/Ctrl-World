#!/usr/bin/env python3
"""
repo/dataset_example/extract_latent.py 를 검증셋 전처리용으로 고쳐서
extract_latent_val.py 를 새로 만든다. 원본은 건드리지 않는다.

고치는 것 두 가지:

1. **`joints` 필드 추가 (버그 수정)**
   원본 스크립트는 annotation JSON에 `joints`를 쓰지 않는데,
   롤아웃 스크립트가 그걸 읽는다:
       repo/scripts/rollout_replay_traj.py:107
           joint_pos = np.array(anno['joints'])
   그래서 새로 전처리한 데이터로 롤아웃하면 KeyError가 난다.
   저장소에 동봉된 예제 annotation에는 `joints`가 들어 있어서,
   공개된 스크립트가 그 데이터를 만든 버전과 어긋나 있음을 알 수 있다.
   값은 예제 데이터로 검증했다:
       joints == concat(joint_position(7), gripper_position(1))[::3]

2. **검증 에피소드만 처리**
   원본은 95,600개를 전부 순회한다. 부분 다운로드에서는 없는 파일이
   bare except로 조용히 넘어가긴 하지만, 시간 낭비이고 로그가 지저분해진다.

사용법:
    python eval/patch_extract_latent.py
    python eval/patch_extract_latent.py --force
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), "repo")
SRC = os.path.join(REPO, "dataset_example", "extract_latent.py")
DST = os.path.join(REPO, "dataset_example", "extract_latent_val.py")


# --- 1) 검증 에피소드만 남기는 필터 ---------------------------------------
FILTER_ANCHOR = "self.data = load_json_file(f'{old_path}/meta/episodes.jsonl')"
FILTER_NEW = """self.data = load_json_file(f'{old_path}/meta/episodes.jsonl')
        # [patched] 검증 분할만 처리 (traj_id % 100 == 99).
        # selected_episodes.json 이 있으면 그 목록으로 더 좁힌다.
        import os as _os, json as _json
        _sel = _os.path.join(old_path, 'selected_episodes.json')
        if _os.path.exists(_sel):
            with open(_sel) as _f:
                _ids = set(_json.load(_f)['episode_ids'])
            self.data = [d for d in self.data if d['episode_index'] in _ids]
            print(f'[patched] selected_episodes.json 적용: {len(self.data)}개')
        else:
            self.data = [d for d in self.data if d['episode_index'] % 100 == 99]
            print(f'[patched] 검증 분할만: {len(self.data)}개')"""


# --- 2) joints 계산 추가 ----------------------------------------------------
# 원본 123행에서 cartesian_states 를 만드는 방식 그대로 joints 를 만든다.
JOINTS_CALC_ANCHOR = (
    "        cartesian_states = np.concatenate((cartesian_pose, cartesian_gripper),"
    "axis=-1)[::rgb_skip].tolist()"
)
JOINTS_CALC_NEW = (
    JOINTS_CALC_ANCHOR
    + "\n        # [patched] rollout_replay_traj.py:107 이 읽는 필드인데 원본에 누락되어 있다."
    + "\n        #           예제 데이터로 검증: joints == concat(joint_position(7), gripper(1))[::3]"
    + "\n        joint_pose = np.array(traj_info['observation.state.joint_position'])"
    + "\n        joint_states = np.concatenate((joint_pose, cartesian_gripper),axis=-1)"
    "[::rgb_skip].tolist()"
)

JOINTS_DICT_ANCHOR = "            'states': cartesian_states,"
JOINTS_DICT_NEW = (
    JOINTS_DICT_ANCHOR + "\n            'joints': joint_states,  # [patched]"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(SRC):
        sys.exit(f"원본을 찾을 수 없습니다: {SRC}")
    if os.path.exists(DST) and not args.force:
        sys.exit(f"이미 존재합니다: {DST}\n  덮어쓰려면 --force")

    src = open(SRC, encoding="utf-8").read()
    out = src

    # 필터
    if out.count(FILTER_ANCHOR) != 1:
        sys.exit(f"[실패] 필터 앵커를 {out.count(FILTER_ANCHOR)}번 찾았습니다 (1번이어야 함)")
    out = out.replace(FILTER_ANCHOR, FILTER_NEW)
    print("  [1/2] 검증 분할 필터 적용")

    # joints — 계산부와 dict 등록부 두 군데
    for label, anchor, new in (
        ("계산", JOINTS_CALC_ANCHOR, JOINTS_CALC_NEW),
        ("등록", JOINTS_DICT_ANCHOR, JOINTS_DICT_NEW),
    ):
        if out.count(anchor) != 1:
            sys.exit(
                f"[실패] joints {label} 앵커를 {out.count(anchor)}번 찾았습니다 (1번이어야 함).\n"
                f"  앵커: {anchor.strip()[:90]}"
            )
        out = out.replace(anchor, new)
    print("  [2/2] joints 필드 추가 (계산 + dict 등록)")

    header = (
        "# 이 파일은 eval/patch_extract_latent.py 가 생성했습니다.\n"
        "# 원본: dataset_example/extract_latent.py (Ctrl-World, ICLR 2026)\n"
        "# 변경: (1) 검증 분할만 처리  (2) annotation 에 joints 필드 추가(원본 누락 버그)\n"
        "# 직접 수정하지 말고 생성 스크립트를 고친 뒤 --force 로 다시 만드세요.\n"
    )
    open(DST, "w", encoding="utf-8").write(header + out)

    import py_compile
    try:
        py_compile.compile(DST, doraise=True)
    except py_compile.PyCompileError as exc:
        sys.exit(f"[실패] 생성된 파일 문법 오류:\n{exc}")

    print(f"\n생성 완료: {DST}")
    print("\n실행 예시:")
    print("  cd repo")
    print("  accelerate launch dataset_example/extract_latent_val.py \\")
    print("    --droid_hf_path  ~/data/droid_val \\")
    print("    --droid_output_path ~/data/droid_val_processed \\")
    print("    --svd_path       $HOME/ckpt/ctrl-world/stable-video-diffusion-img2vid")
    print("\n주의: 정규화 통계(dataset_meta_info/droid/stat.json)는 재계산하지 말 것.")
    print("      256개로 다시 계산하면 논문과 다른 정규화가 되어 수치 비교가 무의미해집니다.")


if __name__ == "__main__":
    main()
