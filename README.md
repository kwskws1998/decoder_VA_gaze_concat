# TRT VA prediction

새 train/validation/test 4조건 실험은 별도 [holdout_experiments/README.md](holdout_experiments/README.md)에 있다. Baseline, companion, 고정 예측 혼합, gaze redistribution을 같은 분할로 비교한다.

아래 `trt_pipeline/`은 기존 two-fold **baseline, companion, 예측 혼합** 실행을 보존한다.

- **Baseline**: historical native BF16 Qwen+TRT prefix concat, 10 epochs, batch16/accum1, grouped sampling. Seed는 `42 + heldout_fold - 1`이며, held-out fold의 최소 mean MSE checkpoint를 선택한다.
- **Companion**: 별도로 초기화한 modern native BF16 Qwen+TRT 모델. Clean/noisy 두 입력의 MSE 평균으로 학습한다. 잡음은 텍스트 임베딩 RMS의 1%이고 TRT는 그대로다. Seed1, uniform batch16/accum1, 16,820 updates이며 학습 쪽 Bvalidation 최소 MSE로 선택한다. 추론은 clean 입력 하나를 쓴다.
- **예측 혼합**: 같은 행의 출력에 `0.692350278020778 * baseline + 0.307649721979222 * companion`을 적용한다. V/A와 모든 문장에 같은 계수를 쓴다. 계수는 이전 Fold1에서 선택한 값으로 고정돼 있다.

```text
trt_pipeline/              세 조건의 학습·추론·혼합과 필수 공통 구현
  core/                    Qwen/TRT, 데이터셋, 지표, Trainer 지원
  historical/snapshot/     원본 해시를 유지한 historical 연산
  companion.py             companion 학습·선택·저장 검사
  two_view.py              clean/noisy 입력
  mixing.py                행 정렬 검사와 고정 예측 혼합
  cli.py                   단일 실행 진입점
tests/                     정리 전후 연산·실제 예측 회귀검사
data/paper7_seed42/        원본 prepared fold 데이터
results/                   실행 결과
server_backup_20261028/    검증 완료 모델 10개와 의존 자산·증거
artifacts/                 이전 결과와 정리 전 복구 ZIP
```

## 환경

실제 GPU 학습은 검증 당시의 Torch2.12.1, Transformers5.14.1 및 native BF16 환경을 요구한다. 새 환경용 의존성은 `requirements.txt`, 서버의 전체 패키지 목록은 `server_backup_20261028/pip_freeze.txt`에 있다. 로컬 환경을 자동으로 변경하지 않는다.

```bash
conda create -n trt_va python=3.11 -y
conda activate trt_va
python -m pip install -r requirements.txt
python run_experiments.py --help
```

Conda 환경 안에서 의존성은 `requirements.txt` 하나로 설치한다. 새 실험은 `run_experiments.py`의 Python 설정 또는 상세 인자로 실행하며, 기본 `--stage run` 한 번으로 baseline, companion, 고정 예측 혼합, gaze redistribution 네 조건을 연속 처리한다. 이 정리 작업에서는 환경 설치나 GPU 재학습을 실행하지 않았다.

## 학습

Fold1은 Fold2 학습→Fold1 평가, Fold2는 Fold1 학습→Fold2 평가다. 기존 결과 디렉터리를 덮어쓰거나 자동 재시작하지 않는다. 다른 실행 이름을 쓰려면 baseline의 `--run-name`, companion의 `--output results/새이름`을 지정한다.

```bash
python -m trt_pipeline baseline --fold 1
python -m trt_pipeline baseline --fold 2
python -m trt_pipeline companion --fold 1
python -m trt_pipeline companion --fold 2
```

Companion의 두 업데이트 사전검증은 다음과 같다. 학습 쪽 데이터만 사용하며 본 학습이나 성능 증거에 합산하지 않는다.

```bash
python -m trt_pipeline companion --fold 1 --preflight
```

원본 데이터 3개 파일의 SHA, 두 방향 A/B/Bfit/Bvalidation 분할의 행 수와 fingerprint를 학습 전에 확인한다. Baseline/companion의 원래 alignment와 head precision 차이를 각각 보존했다.

## 예측 혼합

각 학습의 최종 selected-model 출력은 `predictions.npz`다. 같은 fold의 baseline과 companion만 섞는다. Unicode 행 ID·순서·정답이 다르면 중단한다.

```bash
python -m trt_pipeline mix \
  --baseline results/baseline_fold1/heldout_fold1/predictions.npz \
  --companion results/companion_fold1/predictions.npz \
  --output results/prediction_mixing_fold1.npz
```

기존 백업 모델을 다시 평가할 때는 다음 진입점을 쓴다. `--kind`는 baseline 또는 companion이다.

```bash
HF_HUB_CACHE="$PWD/server_backup_20261028/hf_hub" HF_HUB_OFFLINE=1 \
python -m trt_pipeline predict --kind baseline --fold 1 \
  --model-dir server_backup_20261028/models/fold1_baseline \
  --output results/baseline_fold1_reloaded
```

예측 혼합은 두 모델을 각각 실행해야 한다. 양방향 요약은 각 지표를 fold별로 계산한 뒤 `(Fold1 + Fold2) / 2`로 평균한다. Fold1에서 계수를 선택했고 baseline도 held-out checkpoint 선택을 유지하므로 development 비교이며 독립 test 결과로 해석하지 않는다.

## 보존 및 검증

기존 모델·검증 결과·원본 데이터는 보존했다. 정리 전 소스, 사용자 수정 사항과 이전 실험 파일 1,334개는 `artifacts/cleanup_20261029/original_workspace.zip`에 원래 경로와 SHA manifest를 포함해 보존했다. 최초 git diff는 같은 폴더의 `initial_worktree.patch`에 있다. 이전 실험별 artifacts는 `artifacts/experiment_history/`로 옮겼다.

```bash
python -m pytest -q
```

회귀검사는 실제 두 fold 전체 예측의 혼합값과 6개 지표, 원래 데이터 분할, clean/noisy forward·MSE·gradient, historical snapshot 해시와 활성 import 경계를 확인한다. GPU 점수 재현 검증과 구분한다.
