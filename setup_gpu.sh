#!/usr/bin/env bash
# Ctrl-World 환경 구성 — GPU 머신에서 실행 (Mac에서는 실행 불가: decord가 macOS arm64 휠 없음)
#
# 사용법:
#   bash setup_gpu.sh            # 환경 생성 + 패키지 설치 + 체크포인트 다운로드
#   bash setup_gpu.sh --no-ckpt  # 패키지까지만
#
# 전제: conda(또는 mamba), nvidia-smi 동작, 디스크 여유 20GB+

set -euo pipefail

ENV_NAME="${ENV_NAME:-ctrl-world}"
CKPT_DIR="${CKPT_DIR:-$HOME/ckpt/ctrl-world}"
DOWNLOAD_CKPT=1
[[ "${1:-}" == "--no-ckpt" ]] && DOWNLOAD_CKPT=0

echo "=============================================="
echo " Ctrl-World 환경 구성"
echo "  env      : $ENV_NAME"
echo "  ckpt dir : $CKPT_DIR"
echo "=============================================="

# ---------------------------------------------------------------
# 0. GPU 확인 — Blackwell(sm_120)이면 반드시 cu128 빌드가 필요
# ---------------------------------------------------------------
echo ""
echo "[0/5] GPU 확인"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || {
  echo "  !! nvidia-smi 실패. GPU 머신에서 실행하고 있는지 확인하세요."; exit 1; }
echo "  * RTX 5090 / RTX PRO 6000 Blackwell은 드라이버 570 이상 필요"

# ---------------------------------------------------------------
# 1. conda 환경
# ---------------------------------------------------------------
echo ""
echo "[1/5] conda 환경 생성 (python 3.11)"
eval "$(conda shell.bash hook)"
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "  이미 존재함 — 재사용"
else
  conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"
python -V

# ---------------------------------------------------------------
# 2. torch 먼저 (cu128) — 이 순서가 중요
#    기본 PyPI 휠은 cu126이고 sm_120 커널이 없어서 Blackwell에서
#    "no kernel image is available for execution on the device" 로 죽음
# ---------------------------------------------------------------
echo ""
echo "[2/5] torch 2.7.1 + cu128 설치"
pip install --upgrade pip
pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128

echo "  -- torch/CUDA 확인 --"
python - <<'PY'
import torch
print("  torch      :", torch.__version__)
print("  cuda avail :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device     :", torch.cuda.get_device_name(0))
    print("  capability :", torch.cuda.get_device_capability(0))
    print("  arch list  :", torch.cuda.get_arch_list())
    cap = torch.cuda.get_device_capability(0)
    if cap[0] >= 12 and 'sm_120' not in torch.cuda.get_arch_list():
        raise SystemExit("  !! Blackwell GPU인데 sm_120 커널이 없습니다. cu128 빌드가 맞는지 확인하세요.")
    # bf16 matmul 실검증
    a = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16)
    (a @ a).sum().item()
    print("  bf16 matmul: OK")
else:
    print("  !! CUDA 미인식 — 드라이버를 확인하세요.")
PY

# ---------------------------------------------------------------
# 3. 나머지 패키지
#    requirements.txt에 빠져 있지만 import 되는 것들 포함
# ---------------------------------------------------------------
echo ""
echo "[3/5] 의존 패키지 설치"
pip install \
  "diffusers==0.34.0" \
  "transformers==4.48.1" \
  "numpy==1.26.4" \
  "accelerate==1.3.0" \
  "swanlab==0.6.4" \
  mediapy wandb tqdm decord einops scipy pandas imageio imageio-ffmpeg opencv-python

# mediapy가 외부 ffmpeg 바이너리를 요구 (write_video에서 사용)
echo ""
echo "  -- ffmpeg 확인 --"
if command -v ffmpeg >/dev/null 2>&1; then
  echo "  ffmpeg: $(command -v ffmpeg)"
else
  echo "  ffmpeg 없음 → conda로 설치"
  conda install -y -c conda-forge ffmpeg
fi

echo ""
echo "  -- 저장소 모듈 import 확인 --"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/repo"
python - "$REPO_DIR" <<'PY'
import sys, os
sys.path.insert(0, sys.argv[1])
from models.ctrl_world import CrtlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.utils import key_board_control
print("  repo modules import: OK")
PY

# ---------------------------------------------------------------
# 4. 체크포인트 다운로드 (약 17GB)
# ---------------------------------------------------------------
if [[ "$DOWNLOAD_CKPT" == "1" ]]; then
  echo ""
  echo "[4/5] 체크포인트 다운로드 (~17GB)"
  pip install -q "huggingface_hub[cli]"
  mkdir -p "$CKPT_DIR"

  echo "  (1/3) CLIP  ~0.6GB"
  hf download openai/clip-vit-base-patch32 --local-dir "$CKPT_DIR/clip-vit-base-patch32"

  echo "  (2/3) SVD   ~8GB"
  hf download stabilityai/stable-video-diffusion-img2vid \
      --local-dir "$CKPT_DIR/stable-video-diffusion-img2vid"

  echo "  (3/3) Ctrl-World ~8GB"
  hf download yjguo/Ctrl-World --local-dir "$CKPT_DIR/Ctrl-World"

  echo ""
  echo "  다운로드 결과:"
  du -sh "$CKPT_DIR"/* 2>/dev/null || true
  echo ""
  echo "  Ctrl-World 체크포인트 파일 (아래에서 .pt 경로를 확인하세요):"
  find "$CKPT_DIR/Ctrl-World" -name "*.pt" -o -name "*.safetensors" | head
else
  echo ""
  echo "[4/5] 체크포인트 다운로드 건너뜀 (--no-ckpt)"
fi

# ---------------------------------------------------------------
# 5. 안내
# ---------------------------------------------------------------
echo ""
echo "[5/5] 완료"
cat <<EOF

다음 단계
---------
1) 체크포인트 경로를 확인하고 아래 명령의 \${...} 를 채우세요.
   CKPT_DIR = $CKPT_DIR

2) 원본 재생 롤아웃 1회 (동작 확인용):

   cd $(dirname "$0")/repo
   CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \\
     --dataset_root_path dataset_example \\
     --dataset_meta_info_path dataset_meta_info \\
     --dataset_names droid_subset \\
     --svd_model_path  $CKPT_DIR/stable-video-diffusion-img2vid \\
     --clip_model_path $CKPT_DIR/clip-vit-base-patch32 \\
     --ckpt_path       $CKPT_DIR/Ctrl-World/<체크포인트파일>.pt

3) near-miss 실험 스크립트 생성:

   python ../near_miss/make_rollout_near_miss.py

   자세한 실험 절차는 ../near_miss/README.md 참고.

참고
----
* openpi / JAX 는 설치하지 않았습니다. rollout_replay_traj.py 는 openpi import가
  주석 처리되어 있어 필요 없습니다. pi0.5 정책 실험을 할 때만 별도 설치하세요.
* DROID 전체 데이터셋(370GB)은 학습용입니다. 롤아웃은 repo/dataset_example 만으로 됩니다.
EOF
