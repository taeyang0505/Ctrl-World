# near-miss 실험 — 접촉하지 않았는데 물체가 반응하는가

## 목적

교수님 지적("문제가 진짜인지 실험으로 확인하라")에 대응하는 첫 실측 실험이다.
공개된 조작 월드모델(Ctrl-World, ICLR 2026)에 **같은 장면에서 접촉 궤적과 비접촉 궤적을
쌍으로 넣고**, 비접촉 쪽에서도 물체가 반응하는지를 센다.

결과가 어느 쪽으로 나와도 쓸모가 있다.

- 비접촉에서도 물체가 반응한다 → near-miss hallucination이 실재한다는 자체 증거
- 반응하지 않는다 → 이 모델은 접촉을 구분한다는 뜻이므로, 문제 주장의 범위를 좁혀야 한다
  (어떤 조건에서 깨지는지 다시 찾아야 함)

---

## 먼저 알아야 할 제약 — 시뮬 GT가 없다

원래 구상은 RoboTwin 시뮬레이터의 물체 좌표 GT로 반응 여부를 자동 판정하는 것이었다.
**Ctrl-World는 시뮬이 아니라 DROID(실제 로봇 녹화)로 학습·평가한다.** 따라서:

- 물체의 3D 위치 GT가 없다. 어노테이션에는 로봇 상태(`states`, `joints`)만 있다
- 반응 여부는 **영상에서 판정**해야 한다

그래서 1차 실험은 아래 순서로 간다.

1. **사람 판정**(N=20쌍): 두 영상을 나란히 보고 비접촉 쪽에서 물체가 움직였는지 표시.
   느리지만 확실하고, 첫 증거로는 이게 정직하다
2. 사람 판정으로 현상이 확인되면, 그때 자동 판정(점 추적기 CoTracker 등)을 붙여 규모를 키운다
3. 시뮬 GT 기반 자동 집계는 나중에 RoboTwin으로 옮길 때 (본 연구 3층 지표)

즉 **여기서 나오는 숫자는 near-miss FPR의 예비 추정치**이고, 정식 지표는 시뮬로 넘어가서
완성한다. 발표에서도 그렇게 구분해서 말하는 게 안전하다.

---

## 실험 설계

### 조작하는 것

액션은 **엔드이펙터 절대 좌표**다 (`annotation/*.json`의 `states`, 단위 m/rad):

```
[x, y, z, roll, pitch, yaw, gripper]
```

near-miss는 이 궤적 전체를 한 축으로 평행이동해서 만든다. `make_near_miss()`가 하는 일이
그것이고, **그리퍼 열림(6번 열)은 건드리지 않는다** — 열고 닫는 타이밍이 두 조건에서
같아야 "접촉 여부만 다르다"고 말할 수 있기 때문이다.

권장 오프셋 (정규화 기준 확인 완료: 정규화 범위가 축마다 달라 같은 cm도 입력값 변화가 다름)

| 축 | 오프셋 | 정규화 입력 변화 | 용도 |
| --- | --- | --- | --- |
| z (=2) | +0.04 m | 0.097 | 집기 태스크에서 물체 위를 스쳐 지나감 |
| z (=2) | +0.03 m | 0.073 | 더 아슬아슬한 near-miss |
| y (=1) | ±0.05 m | 0.114 | 밀기·닦기 태스크에서 옆으로 빗나감 |

너무 크게 밀면(10cm+) 로봇이 아예 딴짓을 하는 영상이 되어 "접촉을 구분했다"고 볼 수 없다.
3~5cm가 "닿을 뻔했는데 안 닿음"의 범위다.

### 통제해야 하는 것

생성 모델이라 같은 입력에도 매번 다른 영상이 나온다. 두 조건을 비교하려면 노이즈를 고정해야
한다. 생성 스크립트가 이미 처리한다:

- 궤적 로드 직전 `torch.manual_seed(seed)` — VAE 인코딩의 `latent_dist.sample()`이 확률적이라 필요
- 매 스텝 `torch.manual_seed(seed*100003 + i)` — 확산 초기 노이즈가 전역 RNG에서 나옴

같은 `--seed`로 접촉/비접촉을 각각 돌리면 **차이는 궤적뿐**이 된다.

---

## 실행

### 0. 준비

```bash
cd ~/Desktop/ctrl-world
bash setup_gpu.sh                      # GPU 머신에서
python near_miss/make_rollout_near_miss.py
```

### 1. 한 쌍 돌려보기

```bash
cd repo
export SVD=$HOME/ckpt/ctrl-world/stable-video-diffusion-img2vid
export CLIP=$HOME/ckpt/ctrl-world/clip-vit-base-patch32
export CKPT=$HOME/ckpt/ctrl-world/Ctrl-World/<체크포인트>.pt

COMMON="--dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \
        --dataset_names droid_subset --svd_model_path $SVD --clip_model_path $CLIP \
        --ckpt_path $CKPT --dump_frames"

# 접촉 조건 (원본 궤적)
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_near_miss.py $COMMON --nm_delta 0.0 --seed 0

# near-miss 조건 (z +4cm)
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_near_miss.py $COMMON --nm_axis 2 --nm_delta 0.04 --seed 0
```

출력 영상은 `<save_dir>/Rollouts_replay/video/ax2_d+0.040_seed0_traj_*.mp4` 형태로 저장되고,
`--dump_frames`를 주면 `frames/` 아래에 스텝별 예측 프레임(npy)도 남는다.

한 스텝에 H100 기준 약 5초, A100 약 10초. 기본 설정은 12스텝이므로 롤아웃 1회에 1~2분.
20쌍이면 40회 롤아웃, 대략 1~2시간.

### 2. 궤적 늘리기

`config.py`의 `replay` 분기가 어떤 궤적을 쓸지 정한다 (`config.py:__post_init__`):

```python
if self.task_type == "replay":
    self.val_dataset_dir = "dataset_example/droid_subset"
    self.val_id = ["899", "18599", "199"]
    self.start_idx = [8, 14, 8] * len(self.val_id)
```

여기를 바꾸면 다른 궤적을 쓸 수 있다. 쓸 수 있는 것:

- `dataset_example/droid_subset` — val 4개 (`199`, `899`, `1799`, `18599`)
- `dataset_example/droid_new_setup_full` — 태스크별 45개
  (pickplace 10, towel_fold 10, drawer/laptop/stack/tissue/wipe_table 각 5)

**접촉 실험에 적합한 것**: `pickplace`(집기 순간이 명확), `drawer`(손잡이 접촉),
`stack`(쌓는 순간). `towel_fold`나 `wipe_table`은 접촉이 넓게 퍼져 있어 판정이 애매하다.

> 주의: `droid_new_setup_full`은 `val_dataset_dir`를 태스크 폴더까지 지정해야 한다
> (예: `dataset_example/droid_new_setup_full/pickplace`), 그리고 어노테이션이 축약형이라
> `observation.*` 필드가 없다. 롤아웃은 `states`/`joints`만 쓰므로 문제없다.

### 3. 기록할 것

쌍마다 아래를 표에 남긴다. 이 표가 다음 발표의 "문제 증거 표"가 된다.

| traj | task | 오프셋 | seed | 접촉 조건 물체 반응 | near-miss 물체 반응 | 비고 |
| --- | --- | --- | --- | --- | --- | --- |
| 199 | pickplace | z+4cm | 0 | O | ? | |

- **near-miss에서 반응 = 오답**. 이 비율이 near-miss FPR의 예비 추정치다
- **접촉 조건에서 무반응도 함께 기록**한다. 아무것도 안 움직이는 모델은 FPR이 0이라
  좋아 보이지만 실은 쓸모없다 (contact response rate와 짝으로 봐야 하는 이유)
- 판정이 애매한 쌍은 "애매"로 따로 센다. 억지로 O/X로 밀어넣지 않는다

---

## 미리 알아둘 함정

1. **워크스페이스 클리핑** — `models/utils.py:107-109`에서 `x∈[0.3,0.8] y∈[-0.5,0.5] z∈[0.01,0.5]`로
   자른다. 오프셋을 크게 주면 여기 걸려서 의도한 만큼 안 밀릴 수 있다. 로그에 찍히는
   `cartesian space action` 값으로 실제 적용된 좌표를 확인할 것

2. **첫 프레임 불일치** — 조건 이미지(실제 첫 프레임)와 명령 좌표가 처음부터 어긋나면 모델이
   순간이동을 그린다. 그래서 `--nm_ramp`(기본 5프레임)로 서서히 밀어넣는다. 램프를 0으로
   두지 말 것

3. **자기회귀 드리프트** — 예측 프레임이 다시 입력으로 들어가므로 후반부는 두 조건 모두
   흐려진다. 판정은 **접촉이 일어나야 할 구간(보통 중반)** 에서 한다. 논문 자체가 20초
   일관성을 성과로 내세울 만큼 장기 롤아웃은 약하다

4. **π0.5 경로는 쓰지 말 것** — `rollout_interact_pi.py`는 정책이 모델의 예측 화면을 보고
   다시 계획하는 폐루프라 두 조건의 궤적을 같게 유지할 수 없다. 반드시 replay 계열을 쓴다

5. **torch 빌드** — RTX 5090 / RTX PRO 6000은 sm_120이라 기본 PyPI torch(cu126)로는
   커널이 없어서 죽는다. `setup_gpu.sh`가 cu128로 설치하고 검증까지 한다
