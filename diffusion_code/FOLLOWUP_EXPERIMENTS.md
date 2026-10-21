# DDPM 후속 진단과 평균 예측기를 고정한 확산 실험

## 구현한 조건

| 진입점 | 학습되는 부분 | 확인하려는 것 |
| --- | --- | --- |
| `matched_raw.py` | 기존 Raw TRT encoder + 직접 VA head | 현재 코드에서의 직접 MSE 대조군 |
| `diagnose.py --mode probe` | 기존 DDPM encoder는 고정, 새 직접 VA head만 학습 | DDPM이 학습한 표현에서 직접 회귀가 가능한지 |
| `diagnose.py --mode sampling` | 없음 | 같은 가중치의 표본 수/seed 민감도, 생성 분산, train/heldout 차이 |
| `train_mean_guided.py` | 사전 학습된 Raw TRT 전체는 고정, 새 denoiser만 학습 | 평균 예측을 보존한 표현 위에서 확산을 학습하는 후보 |

기존 DDPM과 Raw TRT의 예측 방식은 유지한다. Raw 학습에는 마지막
`trainer_state.json` 저장만 추가했다. 기존 평가 fold 재사용과 중복 유지 정책은
사용자가 지정한 원본 설정대로 이어진다. 데이터 출처를 모델 입력에 추가하지 않는다.

## 실행 순서

기존 환경에서 새 패키지는 필요 없다. 새 GPU라면 저장소 루트에서 기존
`python setup_environment.py --current-env`와 데이터 준비를 먼저 실행한다.
전체 환경/데이터 준비 명령은 [README](README.md#environment-and-data)에 있다.
아래 명령은 모두 **저장소 루트**에서 실행한다. `DDPM_RUN_NAME`에는 완료된
기존 실험의 실제 폴더 이름을 넣는다. 진단에는 서버의
`heldout_foldN/final_model/model.safetensors`가 필요하다. results-only ZIP에는
가중치가 없으므로 ZIP만으로 frozen probe나 표본 재생성은 할 수 없다.

### 1. 가중치를 바꾸지 않는 진단

```bash
python -u diffusion_code/diagnose.py \
  --run-name DDPM_RUN_NAME \
  --data-dir va_model_code/data/paper7_seed42 \
  --output-dir results/ddpm_diagnostics_seed42 \
  --mode all \
  --samples 16 64 \
  --sampling-seeds 2026 2027 2028 \
  --rows-per-dataset 48 \
  --memory-trt aligned zero \
  --batch-size 8 \
  --probe-epochs 30 \
  --probe-learning-rate 1e-3 \
  --probe-batch-size 256

python diffusion_code/package_diagnostics.py \
  --output-dir results/ddpm_diagnostics_seed42
```

- 데이터 파일의 SHA256과 원본 fold/필터/overlap 조건을 먼저 검증한다.
- 모든 N/seed 조건이 같은 행을 쓴다. 행 선택은 label을 보지 않는 해시로
  출처별 최대 48행씩, train과 heldout에서 각각 정한다.
- 원본 N/seed의 aligned 결과도 반드시 포함해 저장 예측과의 차이를 기록한다.
- 원래 구현의 common noise bank를 유지한다. N이 달라지면 난수 배열의 배치가
  달라져 N=16 경로가 N=64의 부분집합은 아니다. 여러 seed 평균/표준편차를
  보고, 가장 잘 나온 seed를 골라 성능으로 보고하지 않는다.
- `zero`는 denoiser에 추가로 들어가는 직접 TRT memory 값만 0으로 만든다.
  이미 계산한 TRT prefix, Qwen 표현, pooled context는 그대로다. 이 검사는
  추가 경로에 대한 추론 민감도이며, gaze 전체 제거 또는 재학습 ablation이 아니다.
- frozen probe는 원래 학습 fold 전체로만 업데이트한다. Qwen/ET2는 eval 모드와
  no_grad로 고정하고, head만 FP32로 학습한다. 원래 프로토콜대로 heldout MSE로
  head의 best epoch를 고른다. 이는 새로운 독립 test 결과가 아니다.
- `sampling_comparison.tsv`: 조건/partition별 MSE, MAE, Pearson, CCC, 생성 분산,
  `variance/N` 추정량. 마지막 값은 sampling 비용의 추정이지 관측 오차의 정확한 분해가 아니다.
- `sampling_seed_summary.tsv`: 미리 지정한 seed들의 평균과 표준편차.
- `probe_oof/`: 요청한 fold 전체의 직접 회귀 예측/메트릭.
- `probe_comparison.tsv`: 기존 DDPM과 frozen probe를 동일 heldout 행으로
  비교한 전체/fold/데이터셋별 표. 기존 DDPM 예측 사본도 결과 ZIP에 포함한다.
- 생성 분산을 사람의 정답 분산이나 보정된 uncertainty로 부르지 않는다.
- sampling은 subset 결과이고 probe OOF는 전체 요청 fold 결과다. ZIP에서도 구분한다.

`--mode sampling` 또는 `--mode probe`로 나눠 실행할 수 있다. 출력 폴더는
새 폴더여야 하며 원본 run 내부나 원본 run의 상위 폴더는 허용하지 않는다.
`--dry-run`은 입력/가중치 존재와 계약만 검사하며 모델을 로딩하지 않는다.

### 2. 현재 코드의 Raw TRT 대조군

```bash
python -u diffusion_code/matched_raw.py \
  --reference-run DDPM_RUN_NAME \
  --data-dir va_model_code/data/paper7_seed42 \
  --run-name raw_trt_matched_seed42

python va_model_code/package_results.py --run-name raw_trt_matched_seed42
```

기존 DDPM에서 데이터 해시, 필터, overlap, seed, BF16/FP32, batch/누적,
max length, 10 epoch, optimizer 설정, 모델/ET2 revision 등을 읽어 직접 MSE
모델에 적용한다. 해당 run이 no-IEMOCAP 전체 데이터면 여기서도 전체를 쓴다.
source가 base seed 42이면 이 실험도 seed 42 하나이며, heldout 1/2는 기존
정책대로 fold seed 42/43을 사용한다. 두 개의 독립 base seed 실험은 아니다.

새 학습은 사전학습 Qwen에서 시작한다. DDPM이 학습한 가중치에서 시작하는
조건이 아니다. `comparison_reference.json`에 비교의 기준과 현재 코드 해시를
저장한다. 현재 코드/환경과 역사적 DDPM 코드/환경 차이는 여전히 비교 한계다.

### 3. Raw 평균 예측기를 고정하고 잔차 확산 학습

```bash
python -u diffusion_code/train_mean_guided.py \
  --raw-run raw_trt_matched_seed42 \
  --data-dir va_model_code/data/paper7_seed42 \
  --run-name mean_guided_trt_seed42 \
  --conditioning token-trt \
  --epochs 10 \
  --head-learning-rate 1e-4 \
  --diffusion-steps 100 \
  --diffusion-samples 16 \
  --diffusion-width 128 \
  --diffusion-blocks 3 \
  --sampling-seed 2026

python diffusion_code/package_results.py --run-name mean_guided_trt_seed42
```

- Raw run의 두 fold가 완전히 끝난 뒤 실행한다. 두 fold의 final weights를
  실제로 읽으므로 앞 단계 Raw 가중치를 지우면 안 된다.
- 원본 Raw의 데이터/필터, tokenizer, max length, batch/누적, precision을 상속한다.
- 각 fold의 Raw 모델은 반대 fold로 학습된 해당 모델만 읽는다. 다른 fold의
  모델을 평균 예측기로 가져오지 않는다.
- Qwen, TRT projector, 기존 평균 head를 모두 `requires_grad=False` 및 `eval()`로
  고정한다. Dropout도 꺼져 있다. gradient checkpointing은 사용하지 않는다.
- denoiser만 LR `1e-4`로 학습한다. 원본 Raw의 `learning_rate`는 출처 정보로
  남지만 frozen encoder에 optimizer update를 적용하지 않는다.
- 각 batch에서 Qwen은 한 번만 실행한다. reverse T=100 동안 표현/keys/values를 재사용한다.
- 먼저 같은 precision으로 frozen Raw를 다시 평가한다. `frozen_mean_oof/`와
  `mean_comparison.tsv`에서 generated VA와 동일 행으로 직접 비교한다.
- best checkpoint는 generated VA의 MSE로 선택한다. loss 로그는 residual epsilon
  MSE이며, 기존 직접 VA MSE와 수치 크기를 그대로 비교하면 안 된다.
- 별도 family `conditional_mean_guided_ddpm_va` 및
  `mean_guided_architecture.json`을 사용한다. 기존 DDPM 로더로 잘못 읽지 못한다.
- 체크포인트에는 frozen Raw와 denoiser를 함께 저장해 원본 폴더 없이 재로딩할 수 있다.
  결과 ZIP은 가중치와 optimizer를 제외한다. 현재 새 mean-guided 진입점은
  fresh run만 지원하고 학습 중단 후 optimizer resume는 지원하지 않는다.
- `--conditioning pooled`는 추가 단어/TRT memory를 제외한 조건이다. Raw TRT
  prefix는 여전히 사용한다. token-trt와 비교하려면 별도 run 이름으로 실행한다.

## 새 조건의 수식과 한계

정규화 VA를 `y∈[0,1]`, frozen Raw 예측을 `mu(x)`라 두고:

```text
z       = 2*y - 1
mu_z    = 2*mu(x) - 1
r0      = z - mu_z
rt      = sqrt(alpha_bar[t])*r0 + sqrt(1-alpha_bar[t])*epsilon
loss    = MSE(epsilon_pred(rt,t,text,TRT,mu_z), epsilon)
y_draw  = clamp((r0_draw + mu_z + 1)/2, 0, 1)
```

새 denoiser는 평균 예측의 projection을 pooled condition에 더한다.
reverse 과정에서 clean residual의 올바른 범위는
`[-1-mu_z, 1-mu_z]`다. 기존 DDPM처럼 무조건 `[-1,1]`로 자르면
올바른 residual까지 잘리므로 별도 구현했다. 최종 step에서는 noise를 더하지 않는다.
Qwen에는 지정한 autocast를 쓰고, 평균 head/denoiser/loss/sampling은 FP32다.

**이는 개선 가설을 검증하는 새 조건이다.** 기존 DDPM에 확정된 수식 버그가
있다는 뜻이 아니다. Raw 표현을 확산의 초기 불안정한 업데이트로부터 보존하지만,
residual의 평균이 0이면 generated 평균은 Raw와 같고 sampling 오차만 추가될 수 있다.
또한 Raw가 학습 데이터에 과적합했다면 in-sample residual이 heldout residual을
잘 대표하지 않을 수 있다. 이 구현은 학습 데이터의 residual을 사용한다고 명시하며,
heldout label로 residual 학습을 하거나 사후 보정값을 추정하지 않는다.
먼저 진단/대조군 결과를 확인하고 장기 실험 확대 여부를 결정한다.

## 로컬 검증

```bash
python -m pytest -q diffusion_code/tests va_model_code/tests
```

검증 범위는 수식, 원본 가중치 보존, label 분리, optimizer coverage, FP32/BF16,
작은 실제 Trainer의 두-fold 저장/재로딩 및 결과 ZIP 검증이다. 실제 Qwen/ET2
가중치의 서버 실행 시간과 VA 성능 개선은 이 로컬 테스트로 확인되지 않는다.
