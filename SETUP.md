# Ctrl-World 셋업 정리

논문: **Ctrl-World: A Controllable Generative World Model for Robot Manipulation** (ICLR 2026, arXiv 2510.10125)
저장소: https://github.com/Robert-gyj/Ctrl-World — `repo/`에 클론 완료 (commit `99fb206`, 원격 main과 동일)

이 논문을 고른 이유: 공개된 조작 월드모델 중 **코드와 가중치가 모두 나와 있고**, 액션 조건화
구조가 우리 연구 주제(접촉 인과)와 가장 가깝다. 예제 데이터가 저장소에 포함되어 있어
DROID 370GB 없이도 롤아웃이 돌아간다.

---

## 폴더 구성

```
ctrl-world/
├── SETUP.md                        이 문서
├── setup_gpu.sh                    GPU 머신용 환경 구성 + 체크포인트 다운로드
├── near_miss/
│   ├── README.md                   near-miss 실험 설계와 절차  ← 실험은 여기부터
│   └── make_rollout_near_miss.py   실험용 롤아웃 스크립트 생성기
└── repo/                           클론한 원본 (168MB, 예제 데이터 81MB 포함)
```

---

## 한 줄 요약

```bash
# GPU 머신(연구실 서버)에서 — 반드시 tmux 안에서
tmux new -s setup
bash setup_gpu.sh                          # 환경 + 체크포인트(~17GB)
python near_miss/make_rollout_near_miss.py # 실험 스크립트 생성
# 이후 near_miss/README.md 의 실행 절차대로
```

### 연구실 서버(apple, RTX 5090)에서 실행할 때

서버 실측 정보와 제약은 상위 [README.md](README.md)의 "실행 환경" 절 참고. 요점만:

- **tmux 필수** — Slurm이 없어서 SSH가 끊기면 프로세스가 죽는다
- **cu128 유지** — 드라이버가 CUDA 13.0이어도 cu130 휠은 torch 2.9부터만 있다.
  원본이 torch 2.7.1을 핀하므로 `setup_gpu.sh`의 cu128 설치가 그대로 정답
- **대용량은 `/mnt/ssd`로** (쓰기 권한 확보 후) — `/`가 85%라 홈에 쌓으면 위험하다:
  ```bash
  CKPT_DIR=/mnt/ssd/$USER/ckpt bash setup_gpu.sh
  export HF_HOME=/mnt/ssd/$USER/hf_cache
  ```
  권한이 나오기 전까지는 홈으로도 가능(총 ~20GB, 여유 263G 안)

---

## 중요한 사실 몇 가지 (코드 정독으로 확인)

### 1. Mac에서는 설치가 안 된다

`decord==0.6.0`은 macOS arm64 휠이 없고 소스 배포도 없다. **Mac은 편집·분석용으로만 쓰고,
설치와 실행은 GPU 머신에서** 한다. 저장소와 예제 데이터는 이미 Mac에 있으므로 궤적 JSON을
들여다보거나 스크립트를 고치는 건 Mac에서 해도 된다.

### 2. torch는 반드시 cu128 빌드

RTX 5090과 RTX PRO 6000 Blackwell은 compute capability **12.0 (sm_120)**이다.
`requirements.txt`대로 `pip install torch==2.7.1`을 하면 기본 PyPI 휠(cu126)이 깔리는데,
이 빌드의 아키텍처 목록에 12.0이 없어서 실행하는 순간

```
CUDA error: no kernel image is available for execution on the device
```

로 죽는다. `setup_gpu.sh`는 `--index-url https://download.pytorch.org/whl/cu128`로 설치하고,
설치 직후 `sm_120` 포함 여부와 bf16 행렬곱을 실제로 돌려서 검증한다.
호스트 드라이버는 570 이상이어야 한다.

### 3. openpi / JAX는 필요 없다

readme는 π0.5 정책 설치를 안내하지만, 우리가 쓸 `rollout_replay_traj.py`는 openpi import가
주석 처리되어 있다(파일 상단 1~3행). 정책 상호작용 실험을 할 때만 별도로 설치하면 된다.

### 4. requirements.txt에 빠진 패키지가 있다

`imageio`, `opencv-python` 등이 코드에서 import되지만 requirements에 없다. 또 `wandb`,
`swanlab`은 롤아웃에서 쓰지도 않으면서 모듈 최상단에서 import되므로 **설치는 해야 한다**.
`mediapy`는 영상 저장에 외부 `ffmpeg` 바이너리를 요구한다 — 없으면 롤아웃 맨 마지막
줄에서 실패한다. setup 스크립트가 다 처리한다.

### 5. config.py의 기본 경로는 전부 남의 서버 경로다

`config.py:11-13`의 기본값이 `/cephfs/...`로 되어 있다. CLI 인자
(`--svd_model_path`, `--clip_model_path`, `--ckpt_path`)로 넘기면 덮어써지므로
파일을 고칠 필요는 없다.

---

## 받아야 하는 체크포인트

| 이름 | HuggingFace | 용량 | 필요성 |
| --- | --- | --- | --- |
| CLIP 인코더 | `openai/clip-vit-base-patch32` | ~0.6GB | 필수 |
| SVD 베이스 | `stabilityai/stable-video-diffusion-img2vid` | ~8GB | 필수 |
| Ctrl-World | `yjguo/Ctrl-World` | ~8GB | 필수 |
| DROID 데이터셋 | `cadene/droid_1.0.1` | ~370GB | **학습용, 롤아웃엔 불필요** |
| π0.5 정책 | openpi 저장소 | — | 정책 실험 시에만 |

Mac 디스크 여유가 부족하므로(캐시 정리 전 기준 ~31GB) **GPU 머신에서 받는다.**
`setup_gpu.sh`가 `hf download`로 처리한다.

> 참고: Mac의 `~/.cache/huggingface`가 23GB, `~/Library/Caches`가 12GB를 쓰고 있다.
> 공간이 필요하면 여기부터 보면 된다. 다만 이전 프로젝트 모델들(dreamzero-so101-lora,
> so101-megamix 등)이 들어 있으니 지우기 전에 확인할 것.

---

## 예제 데이터 (다운로드 불필요)

`repo/dataset_example/` 81MB에 **롤아웃 가능한 궤적 56개**가 들어 있다.

| 경로 | 내용 |
| --- | --- |
| `droid_subset/` | val 4개 + train 7개. 원본 DROID 필드까지 포함 (49MB) |
| `droid_new_setup_full/` | 태스크별 45개 — pickplace 10, towel_fold 10, drawer·laptop·stack·tissue·wipe_table 각 5 |
| `droid_new_setup/` | 위 폴더의 부분집합(중복). 무시해도 됨 |
| `droid/` | 비어 있음. 전체 DROID를 받을 때 쓰는 자리 |

각 궤적 = 어노테이션 JSON 1개 + 320×192 mp4 3개(외부캠 2, 손목캠 1).
**전처리 불필요** — 롤아웃 스크립트가 mp4를 읽어 VAE 인코딩을 직접 한다.
(`latent_videos/*.pt`는 학습에만 쓰인다.)

어노테이션 주요 필드:

```
texts    : 지시문
states   : (T, 7)  [x, y, z, roll, pitch, yaw, gripper]   ← 액션/포즈 스트림
joints   : (T, 8)  관절 7 + 그리퍼
videos   : mp4 3개 경로
```

---

## 동작 확인 (환경 구성 후)

```bash
cd repo
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset \
  --svd_model_path  $HOME/ckpt/ctrl-world/stable-video-diffusion-img2vid \
  --clip_model_path $HOME/ckpt/ctrl-world/clip-vit-base-patch32 \
  --ckpt_path       $HOME/ckpt/ctrl-world/Ctrl-World/<체크포인트>.pt
```

기본 설정으로 궤적 3개(`899`, `18599`, `199`)를 12스텝씩 재생하고 mp4를 저장한다.
스텝당 H100 ~5초, A100 ~10초.

여기까지 되면 near-miss 실험으로 넘어간다 → `near_miss/README.md`
