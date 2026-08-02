# Table 1 재현 — 직접 구현한 코드

논문 Table 1 (검증 클립 256개, 10초 롤아웃):

| | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD ↓ |
| --- | --- | --- | --- | --- | --- |
| **Ctrl-World** | 23.56 | 0.828 | 0.091 | 25.00 | 97.4 |
| Ctrl-World-Third-View | 21.27 | 0.793 | 0.110 | 23.47 | 127.5 |
| IRASim | 21.36 | 0.774 | 0.117 | 26.46 | 138.1 |
| WPE | 20.33 | 0.772 | 0.131 | 25.50 | 156.4 |

---

## 왜 직접 짜야 하는가

원본 저장소에 **지표 구현이 하나도 없다.** 확인:

```bash
grep -rniE 'psnr|ssim|lpips|fvd|fid' repo --include="*.py"   # 결과 0건
```

`requirements.txt`에도 관련 패키지가 없다. 즉 논문의 수치를 만들어 낸 평가 코드는
공개되지 않았고, 재현하려면 지표 정의부터 우리가 맞춰야 한다.

원본이 제공하는 것과 우리가 만드는 것:

| | 상태 |
| --- | --- |
| 롤아웃 자체 (`agent.get_traj_info`, `agent.forward_wm`) | 원본 재사용 |
| 자기회귀 버퍼·액션 조건화 | 원본 재사용 |
| 정규화 통계 `dataset_meta_info/droid/stat.json` | 원본 그대로 사용 (**재계산 금지**) |
| 검증 클립 선택 다운로드 | **직접 구현** |
| 전처리 스크립트의 `joints` 필드 누락 수정 | **직접 구현** |
| 여러 클립을 도는 평가 하네스 | **직접 구현** |
| PSNR / SSIM / LPIPS / FID / FVD | **직접 구현** |

---

## 파일

| 파일 | 역할 |
| --- | --- |
| `download_val_clips.py` | DROID에서 검증 에피소드만 선택 다운로드 (256개 ≈ 0.8GB) |
| `patch_extract_latent.py` | 전처리 스크립트 패치본 생성 (`joints` 누락 수정 + 검증셋 필터) |
| `metrics.py` | PSNR / SSIM / LPIPS / FID 구현 (FVD는 2차) |
| `run_eval.py` | 클립을 돌며 예측·정답 프레임을 모아 지표 계산 |

---

## 실행 순서

```bash
# 1) 검증 클립 받기 (목록 먼저 확인)
python eval/download_val_clips.py --out ~/data/droid_val --n 256 --dry-run
python eval/download_val_clips.py --out ~/data/droid_val --n 256

# 2) 전처리 스크립트 생성 후 실행
python eval/patch_extract_latent.py
cd repo && accelerate launch dataset_example/extract_latent_val.py \
    --droid_hf_path ~/data/droid_val \
    --droid_output_path ~/data/droid_val_processed \
    --svd_path $HOME/ckpt/ctrl-world/stable-video-diffusion-img2vid
cd ..

# 3) 소규모 확인 (클립 4개)
python eval/run_eval.py --limit 4 --no_fid \
    --dataset_root_path ~/data/droid_val_processed --dataset_names droid \
    --episode_list ~/data/droid_val/selected_episodes.json \
    --svd_model_path $SVD --clip_model_path $CLIP --ckpt_path $CKPT \
    --out results/smoke

# 4) 본 측정 (256개)
python eval/run_eval.py \
    --dataset_root_path ~/data/droid_val_processed --dataset_names droid \
    --episode_list ~/data/droid_val/selected_episodes.json \
    --svd_model_path $SVD --clip_model_path $CLIP --ckpt_path $CKPT \
    --out results/table1
```

`metrics.py`는 단독 실행하면 자체 검증을 돈다.

```bash
python eval/metrics.py
#  동일 입력 → PSNR inf, SSIM 1.0000, LPIPS 0.000000
#  노이즈 증가 → PSNR 34.20 / 24.89 / 16.89 로 단조 악화
```

---

## 논문이 명시하지 않아 우리가 정한 것

재현 결과를 논문과 비교할 때 반드시 함께 밝혀야 하는 항목들이다.

### 1. 조건 프레임 제외

각 라운드는 5프레임을 내놓지만 그중 **첫 프레임은 입력으로 준 프레임**이다
(라운드 간 1프레임 겹침, `rollout_replay_traj.py:276-277`의 `start_id = i*(pred_step-1)`).
이걸 지표에 포함하면 사실상 VAE 왕복 오차만 재게 되어 PSNR이 부풀려진다.
그래서 라운드마다 첫 프레임을 버리고 4프레임씩 쓴다.

### 2. 정답 프레임의 기준 — `--gt_source`

- `raw` (기본): 원본 mp4를 디코드한 프레임
- `vae`: 정답 프레임을 VAE로 인코딩했다가 디코딩한 것

모델은 VAE 잠재공간에서 예측하므로, VAE 왕복 오차는 모델이 아무리 완벽해도
넘을 수 없는 상한이다. **`vae`로 한 번 재보면 그 상한이 얼마인지 알 수 있고**,
`raw` 수치에서 그만큼은 모델 탓이 아니라는 해석이 가능해진다.
논문이 어느 쪽을 썼는지는 명시하지 않았다.

### 3. 카메라 선택

카메라 순서는 `extract_latent.py:68-71`에서 확정된다.

```
view 0 = exterior_1_left   (3인칭)
view 1 = exterior_2_left   (3인칭)
view 2 = wrist_left        (손목)
```

Table 1의 "Ctrl-World-Third-View"는 **3인칭만 쓰도록 학습한 모델**을 가리키는
행이지 평가 카메라를 뜻하는 게 아니다. Table 2가 Third-view / Wrist-view로
나뉘어 보고되는 것으로 보아 Table 1의 수치도 3인칭 기준일 가능성이 높다고 보고
기본값을 view 0으로 두었다. `--views 0 2`로 두 카메라를 함께 재면
논문 Table 2의 3인칭/손목 격차(23.56 vs 19.18)와 비교할 수 있다.

### 4. 롤아웃 길이 — 정확히 10초는 안 나온다

프레임률 계산:

```
DROID 원본 15Hz → rgb_skip=3 → 5Hz (Δt = 0.2초)
라운드당 5프레임 생성, 1프레임은 조건 → 새 프레임 4개 = 0.8초
전체 = 0.8 × interact_num 초
```

| interact_num | 새 프레임 | 시간 |
| --- | --- | --- |
| 12 (원본 기본값) | 48 | 9.6초 |
| **13** | 52 | 10.4초 → **앞 50프레임만 쓰면 정확히 10.0초** |

그래서 기본값을 `--interact_num 13 --max_frames 50`으로 두었다.
논문 본문의 "10 steps → 10초"는 원본 코드 기준으로는 8.0초에 해당해
서술과 코드가 어긋난다. 이 점도 결과에 적어 두는 게 좋다.

### 5. LPIPS 백본

`alex`를 기본으로 썼다(관례). 논문은 명시하지 않았다. `vgg`로 재면 값이
달라지므로(보통 alex < vgg) 비교 시 반드시 백본을 함께 적는다.

### 6. SSIM 구현

`torchmetrics` 기본값(가우시안 커널 11×11, sigma 1.5)을 쓴다.
`skimage`의 기본값(uniform 7×7)과 다르므로 다른 구현과 비교할 때 주의.

### 7. 정규화 통계는 재계산하지 않는다

액션은 `dataset_meta_info/droid/stat.json`의 `state_p01/p99`로 정규화된다
(`rollout_replay_traj.py:68-84`). 이 값은 **DROID 전체**로 계산된 것이고,
우리가 256개로 다시 계산하면 모델 입력 자체가 달라져 논문과 비교가 무의미해진다.
`create_meta_info.py`의 통계 계산부는 주석 처리되어 있어 어차피 재생성도 안 된다.

---

## 알려진 문제

### 전처리 스크립트의 `joints` 누락 (원본 버그)

`extract_latent.py`는 annotation JSON에 `joints`를 쓰지 않는데, 롤아웃이 그걸 읽는다.

```python
# repo/scripts/rollout_replay_traj.py:107
joint_pos = np.array(anno['joints'])
```

동봉된 예제 annotation에는 `joints`가 들어 있으므로, 공개된 전처리 스크립트가
그 데이터를 만든 버전과 어긋나 있다. 새로 전처리한 데이터로 롤아웃하면 KeyError가 난다.
`patch_extract_latent.py`가 이 필드를 복원한다. 값은 예제 데이터로 검증했다:

```
joints == concat(observation.state.joint_position(7), gripper_position(1))[::3]   # (T, 8)
```

### VAE 인코딩이 비결정적

`latent_dist.sample()`이 확률적이라 같은 입력도 실행마다 잠재가 달라진다
(`extract_latent.py`, `rollout_replay_traj.py:135`). `run_eval.py`는 클립마다
시드를 고정하고 라운드마다도 다시 고정해 실행 간 재현이 되게 했다.

---

## 2차 작업

- **FVD** — I3D(Kinetics-400) 체크포인트가 필요하고 구현체마다 값이 달라진다.
  PSNR/SSIM/LPIPS를 먼저 확정한 뒤 붙이고, 어느 구현을 썼는지 반드시 기록한다.
- **FID 표본 수** — 256클립 × 50프레임 = 12,800장. FID는 표본이 적으면 공분산
  추정이 불안정해 값이 커진다. 논문이 몇 장으로 쟀는지 알 수 없어 차이가 날 수 있다.
- **비교 대상(WPE·IRASim)** — 별도 모델이라 이번 재현 범위 밖이다.
