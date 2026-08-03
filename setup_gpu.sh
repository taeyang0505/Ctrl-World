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
# HF 캐시가 홈에 중복으로 쌓이는 것을 막는다. 지정하지 않으면 체크포인트 옆에 둔다.
export HF_HOME="${HF_HOME:-$(dirname "$CKPT_DIR")/hf_cache}"
DOWNLOAD_CKPT=1
[[ "${1:-}" == "--no-ckpt" ]] && DOWNLOAD_CKPT=0

echo "=============================================="
echo " Ctrl-World 환경 구성"
echo "  env      : $ENV_NAME"
echo "  ckpt dir : $CKPT_DIR"
echo "  HF_HOME  : $HF_HOME"
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
# 0b. 디스크 여유 확인 — 공유 서버에서 루트 파티션을 채우면 전체 장애가 난다
# ---------------------------------------------------------------
if [[ "$DOWNLOAD_CKPT" == "1" ]]; then
  echo ""
  echo "[0b] 디스크 여유 확인 (체크포인트 약 17GB + 여유분)"
  TARGET_PARENT="$(dirname "$CKPT_DIR")"
  mkdir -p "$TARGET_PARENT"
  AVAIL_GB=$(df -BG --output=avail "$TARGET_PARENT" 2>/dev/null | tail -1 | tr -dc '0-9')
  USE_PCT=$(df --output=pcent "$TARGET_PARENT" 2>/dev/null | tail -1 | tr -dc '0-9')
  echo "  대상: $TARGET_PARENT  (여유 ${AVAIL_GB}G, 사용률 ${USE_PCT}%)"
  if [[ -n "$AVAIL_GB" && "$AVAIL_GB" -lt 30 ]]; then
    echo "  !! 여유가 30G 미만입니다. 중단합니다."
    echo "     다른 디스크를 쓰려면:  CKPT_DIR=/mnt/ssd/\$USER/ckpt bash $0"
    exit 1
  fi
  if [[ -n "$USE_PCT" && "$USE_PCT" -ge 90 ]]; then
    echo "  !! 파티션 사용률이 ${USE_PCT}% 입니다. 공유 서버라면 관리자와 상의하세요. 중단합니다."
    exit 1
  fi
fi

# ---------------------------------------------------------------
# 1. 파이썬 환경 — conda 가 있으면 conda, 없으면 venv
# ---------------------------------------------------------------
echo ""
echo "[1/5] 파이썬 환경 준비"
USING_CONDA=0
if command -v conda >/dev/null 2>&1; then
  echo "  conda 발견 → conda 환경 사용"
  USING_CONDA=1
  eval "$(conda shell.bash hook)"
  if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "  이미 존재함 — 재사용"
  else
    conda create -y -n "$ENV_NAME" python=3.11
  fi
  conda activate "$ENV_NAME"
else
  echo "  conda 없음 → venv 사용"
  VENV_DIR="${VENV_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv}"

  # torch 2.7.1 은 python 3.9~3.13 을 지원한다. 새 것부터 찾는다.
  PY_BIN=""
  for v in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$v" >/dev/null 2>&1; then
      ver=$("$v" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")
      case "$ver" in
        3.9|3.10|3.11|3.12|3.13) PY_BIN="$v"; break ;;
      esac
    fi
  done
  [[ -z "$PY_BIN" ]] && { echo "  !! 3.9~3.13 파이썬을 찾지 못했습니다."; exit 1; }
  echo "  사용할 파이썬: $PY_BIN ($($PY_BIN -V 2>&1))"

  # 이전 실행이 중간에 실패하면 bin/ 만 있고 activate 가 없는 껍데기가 남는다.
  # activate 존재 여부로 판단하고, 깨져 있으면 지우고 다시 만든다.
  if [[ -e "$VENV_DIR" && ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "  깨진 venv 발견 → 삭제 후 재생성: $VENV_DIR"
    rm -rf "$VENV_DIR"
  fi

  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    echo "  기존 venv 재사용: $VENV_DIR"
  else
    # Ubuntu 는 python3-venv 가 별도 패키지라 ensurepip 이 없는 경우가 많다.
    # 그때는 --without-pip 로 만들고 pip 를 따로 부트스트랩한다 (sudo 불필요).
    if "$PY_BIN" -m venv "$VENV_DIR" 2>/dev/null; then
      echo "  venv 생성: $VENV_DIR"
    else
      echo "  ensurepip 없음 → --without-pip 로 생성 후 pip 부트스트랩"
      rm -rf "$VENV_DIR"
      "$PY_BIN" -m venv --without-pip "$VENV_DIR" || {
        echo "  !! venv 생성 실패."
        echo "     대안 1: sudo apt install python3.10-venv"
        echo "     대안 2: Miniconda 를 홈에 설치 (sudo 불필요)"
        echo "       wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "       bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3"
        echo "       eval \"\$(\$HOME/miniconda3/bin/conda shell.bash hook)\" && bash $0 $*"
        exit 1; }
      echo "  venv 생성(pip 없이): $VENV_DIR"
    fi
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  # pip 확보
  if ! python -m pip --version >/dev/null 2>&1; then
    echo "  pip 부트스트랩 중..."
    GETPIP="$VENV_DIR/get-pip.py"
    if command -v curl >/dev/null 2>&1; then
      curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$GETPIP"
    elif command -v wget >/dev/null 2>&1; then
      wget -q https://bootstrap.pypa.io/get-pip.py -O "$GETPIP"
    else
      python - "$GETPIP" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", sys.argv[1])
PY
    fi
    python "$GETPIP" >/dev/null
    rm -f "$GETPIP"
    python -m pip --version || { echo "  !! pip 부트스트랩 실패"; exit 1; }
  fi
fi
python -V
echo "  python 경로: $(command -v python)"
echo "  pip        : $(python -m pip --version)"

# ---------------------------------------------------------------
# 2. torch 먼저 (cu128) — 이 순서가 중요
#    기본 PyPI 휠은 cu126이고 sm_120 커널이 없어서 Blackwell에서
#    "no kernel image is available for execution on the device" 로 죽음
# ---------------------------------------------------------------
echo ""
echo "[2/5] torch 2.7.1 + cu128 설치"
python -m pip install --upgrade pip
# torchvision 은 torchmetrics 의 LPIPS 가 요구한다. torch 2.7.1 의 짝은 0.22.1 이고
# 반드시 같은 cu128 인덱스에서 받아야 빌드가 어긋나지 않는다.
python -m pip install "torch==2.7.1" "torchvision==0.22.1" \
  --index-url https://download.pytorch.org/whl/cu128

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
python -m pip install \
  "diffusers==0.34.0" \
  "transformers==4.48.1" \
  "numpy==1.26.4" \
  "accelerate==1.3.0" \
  "swanlab==0.6.4" \
  mediapy wandb tqdm decord einops scipy pandas imageio imageio-ffmpeg opencv-python \
  pyarrow huggingface_hub torchmetrics torch-fidelity matplotlib

# mediapy가 외부 ffmpeg 바이너리를 PATH에서 찾는다 (write_video에서 사용).
# 없으면 롤아웃이 마지막 영상 저장 단계에서 실패한다.
echo ""
echo "  -- ffmpeg 확인 --"
if command -v ffmpeg >/dev/null 2>&1; then
  echo "  ffmpeg: $(command -v ffmpeg)"
elif [[ "$USING_CONDA" == "1" ]]; then
  echo "  ffmpeg 없음 → conda로 설치"
  conda install -y -c conda-forge ffmpeg
else
  # conda 가 없을 때는 imageio-ffmpeg 가 동봉한 정적 바이너리를 PATH 에 연결한다.
  # sudo 없이 해결되는 방법이다.
  echo "  ffmpeg 없음 → imageio-ffmpeg 동봉 바이너리를 연결"
  FFMPEG_SRC=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || echo "")
  if [[ -n "$FFMPEG_SRC" && -x "$FFMPEG_SRC" ]]; then
    BIN_DIR="$(dirname "$(command -v python)")"
    ln -sf "$FFMPEG_SRC" "$BIN_DIR/ffmpeg"
    echo "  연결 완료: $BIN_DIR/ffmpeg -> $FFMPEG_SRC"
    ffmpeg -version 2>/dev/null | head -1 || echo "  !! 연결했으나 실행 확인 실패"
  else
    echo "  !! ffmpeg 확보 실패. 롤아웃의 영상 저장 단계에서 실패할 수 있습니다."
    echo "     해결: sudo apt install ffmpeg  (또는 관리자에게 요청)"
  fi
fi

echo ""
echo "  -- 저장소 모듈 import 확인 --"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/repo"
if [[ ! -d "$REPO_DIR" ]]; then
  echo "  원본 저장소가 없습니다 → 지금 클론합니다: $REPO_DIR"
  git clone --depth 1 https://github.com/Robert-gyj/Ctrl-World.git "$REPO_DIR"
fi
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
  python -m pip install -q "huggingface_hub[cli]"
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

if [[ "$USING_CONDA" == "1" ]]; then
  ACTIVATE_CMD="conda activate $ENV_NAME"
else
  ACTIVATE_CMD="source $VENV_DIR/bin/activate"
fi

cat <<EOF

다음 단계
---------
0) 새 셸을 열 때마다 환경을 활성화하세요:

   $ACTIVATE_CMD

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
