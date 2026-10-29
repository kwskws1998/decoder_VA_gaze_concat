# Train / validation / test 실험

기존 two-fold 실행기와 분리된 새 프로토콜이다. 세 모델을 학습하고 네 결과를 비교한다.

| 조건 | 학습 | 평가 |
|---|---|---|
| baseline | Qwen + frozen ET2의 raw TRT prefix concat, MSE | clean 입력 |
| companion | 같은 모델에 clean/noisy 두 view의 평균 MSE | clean 입력 |
| prediction mixture | 추가 학습 없음 | `(1-alpha)*baseline + alpha*companion` |
| gaze redistribution | baseline의 TRT projection 직전에 학습 가능한 비대칭 Gaussian 재분배 | 같은 재분배 적용 |

`alpha=0.307649721979222`를 문장과 V/A 모두에 공통으로 고정한다. 이번 실행에서는 계수를 탐색하거나 조정하지 않는다. Companion의 noisy 입력은 valid text embedding RMS의 1% 표준편차로 Gaussian noise를 더한다. Padding에는 더하지 않으며 frozen TRT는 clean/noisy 간 동일하다.

Redistribution은 과거 `legacy_code/va_model_code/decoder_va/redistribution.py`의 `AsymGaussianRedistributor`를 그대로 분리했다. Qwen token 위치의 valid TRT source를 valid destination으로 분배하며 source별 kernel mass를 1로 정규화한다. Padding/unmapped 위치는 제외하고 valid zero는 유지한다. 좌·우 폭은 전역 FP32 scalar 두 개이며 `exp(log_sigma)+1e-6`, 초기 폭은 각각 `1+1e-6`이다. 폭의 초기 LR은 `1e-3`, weight decay는 0이며 공통 warmup/decay를 따른다. Attention rotation을 추가하는 조건은 아니다.

## 데이터와 비교 조건

- 원본 두 fold의 SHA-256을 검증하고 합친다. IEMOCAP은 제외하며 나머지 53,791행은 유지한다.
- NFKC + casefold + 공백 정규화로 같은 텍스트를 묶는다. 같은 그룹은 출처가 다르더라도 분할을 넘지 않는다.
- 기본 목표는 train/val/test 80/10/10이다. VA label을 보지 않고 출처별 행 수를 맞추며, 그룹 크기 때문에 정확한 비율과 몇 행 차이가 날 수 있다.
- 원본의 빈 텍스트 3행도 한 그룹으로 유지한다. 공통 데이터셋의 EOS fallback을 사용한다.
- 세 모델 모두 같은 train 전체, 같은 validation, 같은 10 epochs, batch16/accum1, 매 epoch 같은 uniform permutation을 사용한다. Companion도 예전 A/B 내부 분할을 만들지 않는다. 두 view이므로 실제 backbone 처리량은 더 많다.
- 공통 seed42, max text length200, full tuning, native BF16 backbone + framework FP32 예외, FP32 head, AdamW LR `6e-6`, WD `.01`, warmup `.1`, linear decay, clip1, gradient checkpointing이다. Bias/norm와 Gaussian 폭에는 decay를 적용하지 않는다.
- **매 epoch 끝에 validation 전체를 한 번 평가한다.** 기본 10 epochs이므로 각 학습 모델마다 validation 10회다. Validation의 두 VA 축 mean MSE가 최소인 checkpoint를 선택한다. 동률이면 먼저 나온 checkpoint를 유지한다. 예산을 조기 종료하거나 늘리지 않는다.
- 세 선택 모델을 모두 seal한 후 test를 한 번씩 평가한다. 고정 예측 혼합은 그 예측 배열로 계산한다. Test로 checkpoint나 계수를 선택할 인터페이스는 없다.
- 기존 historical 실행과 달리 모든 조건에 같은 modern TRT alignment와 FP32 head 연산을 적용한다. 과거 결과의 수치 재현용 실행기는 `trt_pipeline/`이다.

원본 데이터는 이미 과거 개발 실험에 사용됐고 혼합 계수도 그 데이터에서 선택됐다. 따라서 이번 코드의 train/val/test 구분은 실행 내부의 분리이며, test를 역사적으로 전혀 보지 않은 데이터라고 주장할 수는 없다. 새 외부 데이터 평가와 구분한다.

## 예상 시간

동일 RTX4090 한 대의 이전 실측 baseline 6,210초, companion 5,227초를 update 수와 validation 행 수에 맞춰 환산한다. 기본 seed42 분할은 train **43,034** / validation **5,377** / test **5,380**이며 각 fit은 **26,900 updates**다. 환산 범위는 전체 **약 7.1~9.8시간**이고, 상한에 25% 여유를 더하고 올림한 **13시간**을 보수적인 계획으로 잡는다. 새 runner의 실데이터 GPU 실측은 아직 없다. 설치·다운로드·대기 시간은 제외한다.

`prepare`가 정확한 분할 행 수와 update 수로 `runtime_estimate.json`을 만든다. `estimate`로 다시 볼 수 있다. 이 추정으로 실행 도중 epoch나 설정을 바꾸지 않는다.

## Conda 환경과 requirements.txt

Conda 환경을 활성화한 뒤 **의존성은 `requirements.txt` 하나로 설치한다.** 별도 환경 설치 스크립트는 필수가 아니다.

```bash
conda create -n trt_va python=3.11 -y
conda activate trt_va
python -m pip install -r requirements.txt
```

이미 환경이 있으면 `conda create`를 생략하고 해당 환경을 활성화한다. `requirements.txt` 안에 PyPI 의존성과 공식 CUDA13.0 Torch wheel 소스, `torch==2.12.1+cu130`, FLA가 함께 지정돼 있다. 기존 RTX4090 실측 환경에 맞춘 GPU 프로필이다.

설치 후 검사는 다음 명령으로 할 수 있다. 이 검사는 실제 실험 학습을 시작하지 않는다.

```bash
python -m pip check
python run_experiments.py --stage check-runtime --gpu 0
```

`setup_environment.py --check-only`는 CUDA forward/backward와 kernel smoke까지 확인하는 선택적 보조 도구다. 기본 설치 경로는 위의 `pip install -r requirements.txt`이며, 보조 도구도 같은 파일 외에 별도 의존성 목록이나 CUDA 프로필을 만들지 않는다. GPU 실험은 `CONDA_PREFIX`, `sys.prefix`, `conda-meta`가 같은 conda 환경인지 확인한다.

## 네 조건을 한 번에 연속 실행

기본 `--stage run`은 아래 **네 조건 전체**를 처리한다. 개별 조건을 수동으로 네 번 실행할 필요가 없다.

1. **Baseline**: train 학습 + 매 epoch validation + 최적 checkpoint 저장.
2. **Companion**: 같은 split으로 clean/noisy 학습 + 매 epoch validation + 최적 checkpoint 저장.
3. **고정 예측 혼합**: 위 두 선택 모델과 고정 alpha를 기록한다. 별도 optimizer 학습은 없고 test를 아직 읽지 않는다.
4. **Gaze redistribution**: 같은 split으로 학습 + 매 epoch validation + 최적 checkpoint 저장.

세 학습 모델의 선택이 모두 끝나면 test를 baseline → companion → prediction mixture → redistribution 순서로 자동 평가한다. 혼합은 baseline/companion의 저장된 test 예측을 사용하며 두 모델을 다시 학습하거나 test inference를 반복하지 않는다. 네 조건의 개별 결과와 한 장의 비교 표를 저장한 뒤 CPU 검증까지 자동 실행한다.

`suite_status.json`은 네 조건의 진행·완료·실패를 구분한다. 한 조건이 실패하면 다음 조건과 test로 넘어가지 않고 오류를 보존한다. **네 결과와 최종 검증이 모두 완료돼야 전체 suite가 completed가 된다.**

이미 redistribution을 완료했다면 기존 명령에 `--conditions baseline companion prediction_mixture`를 지정한다. 이 경우 baseline과 companion 두 모델만 학습하고, 두 validation 선택이 모두 확정된 뒤 세 조건의 test 평가·고정 혼합·CPU 검증까지 자동 실행한다. Redistribution 모델이나 결과는 요구하거나 새로 생성하지 않는다. 선택한 세 조건은 준비 시 protocol lock과 상태·보고서에 기록된다. 기본값은 기존 네 조건이다. 코드가 바뀌기 전에 만든 protocol lock을 재사용하지 말고 새로운 output 경로를 사용한다.

## Python 실험 파일과 상세 인자

**직접 실행할 파일은 루트 `run_experiments.py`다.** 파일 상단에 `OUTPUT`, `INPUT_DATA`, `STAGE`, `TRAIN_ARM`, `GPU`, `EXPERIMENT_SETTINGS = Settings(...)`가 있으며 split 비율, seed, epochs, batch, LR, WD, warmup, clip, Gaussian 폭과 LR, 고정 혼합 계수, noise, pinned model IDs/revisions를 모두 명시했다. 실행 전에 Python 파일에서 값을 적어도 되고, 아래처럼 상세 인자를 주어도 된다. import만 하면 실행되지 않는다.

파일의 Python 설정을 그대로 실행:

```bash
python -u run_experiments.py
```

기본 네 조건을 상세하게 선언하는 전체 실행 명령:

```bash
python -u run_experiments.py \
  --stage run \
  --conditions baseline companion prediction_mixture redistribution \
  --gpu 0 \
  --data-dir data/paper7_seed42 \
  --output results/holdout_seed42 \
  --fractions 0.8 0.1 0.1 \
  --split-seed 42 \
  --seed 42 \
  --epochs 10 \
  --train-batch-size 16 \
  --eval-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --max-length 200 \
  --learning-rate 6e-6 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --max-grad-norm 1.0 \
  --precision bf16 \
  --loss mse \
  --validation-every-epochs 1 \
  --noise-std-fraction 0.01 \
  --prediction-alpha 0.307649721979222 \
  --redistribution-learning-rate 1e-3 \
  --sigma-left 1.0 \
  --sigma-right 1.0 \
  --min-sigma 1e-6
```

`python run_experiments.py --help`에 model/ET2 ID와 revision 인자도 있다. 이번 비교의 noise, 혼합 계수, accumulation1, BF16, MSE, epoch마다 validation은 명시적으로 고정돼 있다. Epoch, LR, batch, split 등 지원되는 설정은 새 실행을 만들기 전에 지정한다. Batch/길이/모델을 바꾸면 이전 RTX4090 시간 환산은 유효하지 않으므로 ETA를 숫자로 제시하지 않는다.

분할·추정부터 보고 단계별로 실행하려면 같은 Python 파일에서 stage만 지정한다:

```bash
python run_experiments.py --stage prepare --output results/holdout_seed42
python run_experiments.py --stage estimate --output results/holdout_seed42
python -u run_experiments.py --stage train --output results/holdout_seed42 --arm baseline --gpu 0
python -u run_experiments.py --stage train --output results/holdout_seed42 --arm companion --gpu 0
python -u run_experiments.py --stage train --output results/holdout_seed42 --arm redistribution --gpu 0
python run_experiments.py --stage seal --output results/holdout_seed42
python -u run_experiments.py --stage evaluate --output results/holdout_seed42 --gpu 0
python run_experiments.py --stage verify --output results/holdout_seed42
```

`run`은 새 폴더 생성부터 전체 실험과 결과 검증까지 실행한다. `prepare`한 폴더에는 단계별 명령을 사용한다. `prepare` 이후에는 이미 고정된 설정을 읽으며 다른 값으로 바꾸면 거부한다. Source hash도 고정하므로 준비 후에는 Python 파일을 수정하지 말고 `--stage`와 `--output`으로 단계를 선택한다. 변경된 실험은 새 output에서 시작한다.

이미 시작한 arm을 덮어쓰거나 자동 재시작하지 않는다. 실패 기록과 선택 checkpoint는 보존하며 checkpoint resume는 지원하지 않는다. Test가 시작된 실행에서는 학습을 추가할 수 없다. GPU 학습은 활성 conda 소유권을 확인한 뒤 시작한다. 기본 10 epochs 중 성능에 따라 예산을 늘리거나 줄이는 경로는 없다.

GitHub에 필요한 코드는 `run_experiments.py`, `setup_environment.py`, `environment_runtime.py`, `requirements.txt`, `holdout_experiments/`, `trt_pipeline/`다. 기존 모듈 실행기는 호환성을 위해 유지하지만 새 상세 실행의 기본 진입점은 `python run_experiments.py`다.

## 결과와 검증

```text
results/holdout_seed42/
  protocol_lock.json           분할·코드·설정·원본 SHA
  suite_status.json            네 조건의 연속 실행 상태와 이벤트
  runtime_estimate.json         실측을 바탕으로 환산한 시간 범위
  splits/{train,val,test}.tsv   고정된 행 순서와 labels
  {baseline,companion,redistribution}/
    model/                     선택된 전체 safetensors + tokenizer + reconstruction manifest
    validation_history.json    epoch별 validation 지표, step, 시간, 선택 여부
    optimizer_audit.json        첫 update 후 모든 live AdamW moment dtype
    final_optimizer_audit.json  최종 update 후 같은 검사
    strict_reload.json         keys/shapes/dtypes와 validation probe 저장·복원 검사
    train_complete.json        실제 update 수, 선택 epoch/step, 시간, VRAM peak
  prediction_mixture/preparation.json  혼합 계수와 두 선택 모델 SHA, 추가 학습 0회
  {네 조건}/condition_complete.json    각 조건의 test 지표와 예측 SHA
  selection_lock.json          test 접근 전 고정한 세 모델과 alpha
  test_results/
    {baseline,companion,redistribution,prediction_mixture}.npz
    report.json                각 모델 지표와 baseline 대비 차이, 시간, 선택 증거
    metrics.tsv                V/A Pearson, RMSE, MAE, MSE, CCC와 기록된 평균·행 수
    verification.json          CPU에서 행/label/SHA/혼합/지표 재계산
  complete.json                최종 결과 SHA
```

모델은 `holdout_experiments.model.load_saved`로 복원한다. Custom full-model wrapper이므로 일반 `AutoModel.from_pretrained`에 바로 전달하지 않는다. 이전 10개 백업 모델을 초기 가중치로 쓰지 않고 pinned pretrained Qwen에서 세 번 새로 시작한다. CPU 검증은 큰 모델을 추론한 증거가 아니며 GPU strict reload 검사와 별도로 기록한다.

```bash
python -m pytest -q tests/test_holdout.py tests/test_launchers.py tests/test_suite.py
```

테스트는 모델 다운로드와 GPU 학습 없이 split 누수 방지, 정확한 noisy/redistribution 연산·gradient, validation 선택, 저장·복원과 test gate를 검증한다.
