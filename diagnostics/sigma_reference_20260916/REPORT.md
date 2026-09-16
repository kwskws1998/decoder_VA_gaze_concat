# 참조 reward-model 코드와 현재 VA 실험의 sigma 차이

작성일: 2026-09-16

## 결론

현재 VA 커널이 큰 비대칭을 표현하거나 학습하지 못한다는 증거는 없다. 같은 입력과 올바른 마스크를 주면 첨부 커널과 현재 커널의 출력·log-sigma gradient가 수치적으로 일치한다. 현재 커널도 오른쪽 폭이 왼쪽의 약 8배인 합성 목표를 학습했다.

두 시스템 사이에서 가장 먼저 통제해야 할 구체적인 차이는 **sigma learning rate와 schedule**이다. 첨부 `main.py` 기본값은 `0.05`이고 이번 VA 실행은 `0.001`이다. 참조 스케줄은 높은 학습률을 유지하며, accumulation을 스케줄 총 step에서 빠뜨려 기본 accumulation 8에서는 감쇠가 거의 진행되지 않는 경로도 확인했다. 추가로 참조 기본 LoRA 경로는 gaze projector를 동결하지만 이번 VA full fine-tuning에서는 projector와 backbone도 함께 학습한다.

다만 **과거 극단적인 오른쪽 sigma 실행의 실제 옵션은 미확인**이다. 사용자는 당시 learning rate를 기억하지 못한다고 답했다. 이 ZIP에는 과거 실행 로그·학습 설정·가중치가 없으므로, 아래 참조 기본값을 과거 실행의 확정 설정으로 간주하지 않는다. 학습률 차이는 변화량을 설명할 후보이지, 오른쪽이라는 방향을 단독으로 설명하지 않는다.

## 1. 자료와 재현 범위

- 참조: `34755_Leveraging_Psychophysica_Supplementary Material (2).zip`, SHA-256 `6d8fef30b7f8a0ad9cd2f6d81220a7d5b16c73ea1f3523896762de2be5228e61`.
- 원본 소스 14개를 변경 없이 `reference_source/`에 보관했다. [추출 manifest](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_reference_20260916/source_manifest.json)와 각 파일 hash를 수치 검증 실행 시 다시 대조했다.
- 현재 실행: `sigma_sentence_only_seed42_results_only.zip`, SHA-256 `eb5ddd199076fefb4141e6da6b1455051c0c04737180d66e361262dc9d56b9e5`. [기존 실제 결과 분석](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_20260916/REPORT.md)의 검증된 수치를 사용했다.
- 새 수치 검증: CPU / PyTorch 2.8.0 / FP32. LLM·ET 가중치를 내려받거나 downstream 학습을 재실행하지 않았다. BF16 전체 모델 동치 검증도 아니다.
- [재현 Python 코드](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_reference_20260916/compare_kernels.py), [수치 결과 JSON](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_reference_20260916/comparison.json).

## 2. 첫 번째 차이: sigma LR은 50배, 감소 방식도 다르다

첨부 [main.py:193](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_reference_20260916/reference_source/main.py:193) 원문:

```python
parser.add_argument("--init_sigma_left", type=float, default=1.0)
parser.add_argument("--init_sigma_right", type=float, default=1.0)
parser.add_argument("--sigma_lr", type=float, default=5e-2)
parser.add_argument("--sigma_lr_scheduler_type", type=str, default="cosine_with_min_lr")
```

이 값이 [생성자 호출:287](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_reference_20260916/reference_source/main.py:287)에 전달된다. `RewardTrainerConstructorGeneral.__init__`의 `5e-3`은 `main.py` 기본 실행에서 덮어써지므로 실제 진입점 기본값으로 쓰면 안 된다.

| 항목 | 첨부 main.py 기본 경로 | 이번 VA 실행에서 확인 |
|---|---|---|
| 초기 left/right | 1 / 1 | 1 / 1 |
| 최적화 변수 | log-sigma | log-sigma |
| sigma 최대 LR | 0.05 | 0.001 |
| sigma weight decay | 0 | 0 |
| 스케줄 | custom cosine, 최저 배율 0.7 | 10% warmup 후 linear decay to 0 |
| accumulation | 8 | 1 |
| sigma와 함께 학습 | 기본 LoRA 경로, 아래 4절 참조 | full backbone + projector + VA head |

참조 optimizer는 `sigma_left`/`sigma_right`로 이름이 끝나고 `requires_grad=True`인 파라미터를 별도 그룹에 넣는다. `log_sigma_left/right`도 이 suffix에 해당한다. 현재 코드도 sigma 두 개를 별도 LR 그룹에 넣고 weight decay 0을 적용한다. 따라서 현재의 `weight_decay=0.01`이 sigma를 1로 끌어당겨 움직이지 못하게 했다는 설명은 맞지 않는다.

### Accumulation을 빠뜨린 참조 스케줄

첨부 [reward_trainer_general.py:570](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_reference_20260916/reference_source/trainers/reward_trainer_general.py:570) 원문:

```python
num_training_steps = num_samples // self.batch_size * self.train_epochs

num_warmup_steps = int(num_training_steps * 0.01)
```

감쇠 바닥값을 만드는 [같은 파일:580](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_reference_20260916/reference_source/trainers/reward_trainer_general.py:580) 원문:

```python
cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
return (1 - self.min_lr_ratio) * cosine_decay + self.min_lr_ratio
```

`main.py`에서 `min_lr_ratio=0.7`, `gradient_acum_steps=8`이다. 그런데 첫 식에는 accumulation이 없다. ZIP이 지정한 Transformers 4.40.0은 optimizer update 후 scheduler를 한 번 진행한다. 따라서 단일 GPU, 기본 accumulation 8, 정상 update를 가정하면 실제 학습 종료까지 스케줄 총 길이의 약 1/8만 진행된다. `TrainingArguments`의 `warmup_ratio=0.02`도 이 별도 전달된 custom scheduler의 실제 warmup을 바꾸지 않는다. [Transformers 4.40.0 Trainer 원문](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py#L2057-L2065)

아래는 참조 optimizer 함수를 AST로 그대로 추출해 실행한 계산이다. **두 방식 모두 실제 optimizer update가 4,490번이라고 가정한 비교이며 과거 학습 로그가 아니다.** 단일 GPU, 건너뛴 update 없음, epoch 경계 나머지 문제 없음으로 구성했다.

| 방식 | 마지막 update 직후 sigma LR | 적용된 LR 합 / 현재 LR 합 |
|---|---:|---:|
| 현재: 0.001, 10% warmup + linear | 0 | 1 |
| 참조: 0.05, accumulation 1 | 0.035 | 84.65 |
| 참조: 0.05, accumulation 8 | 0.049506 | 95.69 |

LR 합은 움직일 수 있는 규모를 비교하는 수치이지 실제 파라미터 이동량이 아니다. Adam moment, gradient 방향, 전체 모델 학습 경로가 바뀌므로 sigma가 96배 더 움직인다고 예측할 수 없다. 다만 두 실험이 비슷한 최적화 조건이었다고 볼 수 없다는 근거는 명확하다.

## 3. Gaussian 식을 옮기면서 비대칭 학습이 사라졌는가?

아니다. 유효 위치의 간격과 마스크가 같다면 양쪽 모두 다음 연산이다.

\[
K_{ij}=m_i m_j\exp\left[-\frac{(i-j)^2}{2\sigma_{ij}^2}\right],\qquad
W_{ij}=\frac{K_{ij}}{\sum_k K_{kj}},\qquad
y_i=\sum_jW_{ij}x_j,
\]

\(i<j\)이면 left, 나머지에는 right를 쓴다. 양쪽 모두 \(\sigma=\exp(\theta)+10^{-6}\). 유효 source에는 자기 위치의 kernel 값 1이 있으므로 현재 구현의 denominator clamp 1과 참조의 clamp `1e-8`은 정상적인 유효 source에서 같은 결과를 준다. masked source는 output 기여가 0이다. 현재의 log-distance 안정화 역시 시험한 폭에서 방향을 바꾸지 않는다.

### 동일 입력·동일 마스크 대조

길이 1/2/8/31/200, dense/right-padded/interior-gap/all-masked 4종 마스크, 대칭·비대칭·극단 폭 5종을 조합한 100개 조건을 확인했다. signed FP32 입력을 사용하고, 공통 무작위 선형 readout의 loss로 log-sigma gradient를 비교했다.

| 검증 | 최대 절대 차이 |
|---|---:|
| 재분배 output | 4.7684e-7 |
| log-sigma gradient | 2.9803e-8 |

각 구현의 masked output=0과 유효 source 질량 보존도 검증했다. 이는 해당 CPU FP32 조건들의 수치적 일치이며 모든 dtype·입력·전체 pipeline의 완전한 동치를 뜻하지 않는다.

### 오른쪽 폭이 8배인 합성 목표를 실제 학습

같은 TRT 입력에 참조 커널의 목표 폭 `0.5 / 4.0`을 적용해 목표 output을 만들었다. 양쪽을 `1 / 1`에서 시작해 동일 AdamW, weight decay 0, constant LR로 각각 학습했다.

| LR / step | 참조 커널 left / right | 현재 커널 left / right |
|---|---|---|
| 0.001 / 1,000 | 0.507135 / 2.487308 | 0.507135 / 2.487308 |
| 0.001 / 3,000 | 0.492223 / 3.982818 | 0.492223 / 3.982818 |
| 0.05 / 500 | 0.500001 / 4.000001 | 0.500001 / 4.000000 |

이 대조는 두 가지를 보여준다.

1. 현재 구현에는 큰 오른쪽 비대칭을 못 배우게 하는 구조적 제한이 없다.
2. LR 0.001도 일관된 학습 신호가 있으면 큰 비대칭을 학습한다. 따라서 실제 VA 결과를 작은 LR 하나만으로 설명할 수 없다.

이 합성 loss에는 gaze projector, decoder, V/A label, clipping, 실제 warmup/decay가 없다. VA에서도 이 폭으로 수렴하거나 성능이 좋아진다는 증거로 사용하지 않는다.

## 4. 두 번째 차이: 참조 기본 LoRA 경로에서 projector는 동결된다

첨부 [reward_trainer_general.py:113](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_reference_20260916/reference_source/trainers/reward_trainer_general.py:113) 원문:

```python
if bnb_config is not None:
    model = prepare_model_for_kbit_training(model)

if peft_config is not None:
    model = get_peft_model(model, peft_config)
```

첨부 [LoRA 설정:420](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_reference_20260916/reference_source/trainers/reward_trainer_general.py:420) 원문:

```python
task_type=TaskType.SEQ_CLS,
```

```python
modules_to_save=["asym_gaussian_redistributor"],
```

참조가 지정한 PEFT 0.10.0에서 k-bit 준비 함수는 기존 파라미터를 동결한다. LoRA 경로는 adapter 및 지정된 modules-to-save를 학습하도록 처리하고, sequence-classification wrapper는 `classifier`/`score`를 추가한다. 이 ZIP은 `fixations_embedding_projector`와 `norm_layer_fix`를 학습 대상으로 추가하거나 다시 unfreeze하지 않는다. 따라서 기본 신규 학습 경로를 이 의존성 구현과 결합하면 **projector와 해당 LayerNorm은 동결되는 구성**이다. 이는 pinned 코드 경로 분석이며 과거 실제 실행의 trainable-parameter dump를 확인한 것은 아니다. [PEFT k-bit 준비 함수](https://github.com/huggingface/peft/blob/v0.10.0/src/peft/utils/other.py#L90-L100), [LoRA 동결 처리](https://github.com/huggingface/peft/blob/v0.10.0/src/peft/tuners/lora/model.py#L231-L248), [분류 head 학습 처리](https://github.com/huggingface/peft/blob/v0.10.0/src/peft/peft_model.py#L823-L835)

현재 VA는 full fine-tuning이고 [backbone.requires_grad_(True)](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/model.py:517)를 적용하며, 별도 생성한 gaze projector와 VA head도 학습한다. gaze를 해석하는 가중치가 고정된 경우와 함께 변하는 경우에는 sigma가 받는 gradient가 같지 않다. VA 모델이 다른 가중치를 조정해 loss를 줄일 수 있다는 것은 작은 sigma 이동의 가능한 설명이다. 그러나 동결하면 반드시 sigma가 더 크거나 오른쪽으로 간다는 결론은 대조 실험 없이 내릴 수 없다.

## 5. LayerNorm 가설은 두 시스템의 차이로 사용할 수 없다

첨부 [reward_model_base.py:101](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_reference_20260916/reference_source/models/reward_model_base.py:101) 원문 일부:

```python
nn.Linear(1, 128),
```

```python
nn.LayerNorm(128),
nn.ReLU(),
```

```python
self.norm_layer_fix = nn.LayerNorm(hidden_size)
```

현재 [projector:185](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/model.py:185)도 `Linear → LayerNorm → 활성화 → Linear → LayerNorm` 구조다. ReLU/GELU 차이는 있지만 LayerNorm의 존재와 위치는 공통이다. 참조 ET1은 scalar 입력이고, ET2에서는 선택된 feature 수가 입력 차원이 된다. 참조 `main.py` 기본은 ET1이므로 과거 실험을 5-feature ET2로 단정해서도 안 된다.

**이전 분석에서 LayerNorm을 출력 민감도가 작아지는 후보로 든 것은 유지할 수 있지만, 이것만으로 참조의 큰 sigma와 현재의 작은 sigma 차이를 설명하는 것은 수정해야 한다.** 현재 전체 pipeline의 낮은 민감도 관찰과 LayerNorm이 그 원인이라는 확정은 서로 다른 주장이다.

## 6. 세 번째 차이: 학습 목표와 sigma의 거리 단위

### Reward ranking과 VA 회귀는 gradient가 다르다

참조는 chosen/rejected 응답 쌍을 TRL `RewardTrainer`에 넣는다. ZIP이 지정한 TRL 0.9.6의 margin 없는 loss 원문:

```python
loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

두 응답의 reward 차이를 키우는 목적이다. 현재는 두 출력 V/A의 0–1 예측에 MSE를 적용한다. 어느 방향으로 TRT를 퍼뜨리면 loss가 줄어드는지는 이 목표와 모델·입력에 의해 결정되므로 참조에서 right가 커졌다는 사실이 VA에서도 right가 커져야 한다는 조건이 되지 않는다. [TRL 0.9.6 RewardTrainer](https://raw.githubusercontent.com/huggingface/trl/v0.9.6/trl/trainer/reward_trainer.py)

이번 실제 로그에는 이미 이 차이를 뒷받침하는 제한적인 증거가 있다.

- 학습형에서 기록된 LR>0 update 178개 모두 양쪽 log-sigma가 실제로 변했다. sigma의 optimizer 연결이 끊어진 상황이 아니다.
- 선택 sigma는 fold 1 `1.054003 / 0.936260`, fold 2 `1.061833 / 0.967911`. right/left 비율은 약 0.888/0.912이다. 현재 결과는 극단적인 비대칭은 아니지만 정확히 1:0.96으로 고정된 것도 아니다.
- fold 2는 epoch 3에서 `1.088320 / 0.921527`까지 벌어진 뒤 다시 가까워졌다. 일관된 한쪽 방향의 이동만 누적되지 않았다.
- 선택 모델에서 sigma를 `0.5 / 2`로 바꿨을 때 TRT 상대 변화는 34–45%였지만 V/A 예측 RMS 변화는 약 0.0028/0.0079였다. 이것은 fold마다 같은 훈련 문장 2개에 대한 probe이고 전체 데이터 추정치가 아니다.
- fold 2의 한 문장에서는 left log-sigma gradient가 V loss에서 `+2.0433e-4`, A loss에서 `-2.6354e-4`였다. 두 과제가 공유 sigma를 반대 방향으로 밀 수 있다. 모든 문장에 대한 상쇄 비율은 미확인이다.

즉 현재는 **sigma를 학습할 수는 있지만, 관측된 VA 신호가 큰 한쪽 비대칭을 지속해서 요구하지 않았다**는 해석이 자료에 맞다. 최적 sigma가 1 부근이라고 증명된 것은 아니다.

### 양쪽 숫자 1이 같은 길이를 의미하지 않는다

참조 concat 경로는 [remap=False](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_reference_20260916/reference_source/models/reward_model_general_sp.py:161)로 가져온 predictor fixation 배열에 재분배를 적용한다. 현재는 ET2 word feature를 [Qwen 첫 subword 위치에 매핑](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/gaze.py:459)하고 원래 Qwen token 간격을 유지한 채 kernel을 계산한다. 이후 유효 gaze prefix를 만든다.

같은 인접 단어여도 Qwen 위치가 0과 3이면 거리 3이다. σ=1에서 정규화 전 가중치는 거리 1의 `exp(-0.5)=0.6065`에서 거리 3의 `exp(-4.5)=0.0111`로 줄어든다. 이것은 마스크 누락이 아닌 좌표계 차이다. 따라서 sigma 절대값을 읽기 폭이나 단어 개수로 직접 비교하면 안 된다. 첨부 ZIP 밖 predictor 내부 tokenizer 구현 전체는 이번 비교에서 재현하지 않았으므로 참조의 모든 fixation 위치를 일괄적으로 '단어 단위'라고 단정하지 않는다.

## 7. 이미 알려진 masking 문제의 처리

사용자가 알려준 문제를 두 실험 차이의 단독 원인으로 재사용하지 않았다. 새 수치 검증에서는 동일한 올바른 source/target mask를 양쪽에 직접 주어 이 혼선을 통제했다.

이번 ZIP의 kernel은 `weights = weights * src_mask * tgt_mask`를 수행하고, `process_fixations`도 mask를 전달한다. 그렇다고 전체 pipeline에 mask 문제가 없다는 뜻은 아니다. 예를 들어 [per-sequence 호출:302](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_reference_20260916/reference_source/models/reward_model_base.py:302)에서는 padding을 제거한 한 문장의 `torch_seq`와 원래 batch의 `attention_mask`를 함께 넘기며, ET2 경로에서는 이를 predictor로 전달한다. 실제 문제가 어떻게 나타나는지는 predictor 버전·batch·cache에 따라 달라진다. 과거 실행의 정확한 mask 오류 효과는 이 ZIP과 가중치 없는 결과만으로 분리할 수 없다.

올바른 마스크에서도 현재 커널이 큰 비대칭을 학습한다는 사실은 확인했다. 반대로 과거의 극단 sigma가 masking 오류와 무관한 과제 최적값이었다고 확인한 것은 아니다.

## 8. 왜 short sentence만 써도 작은가? 다음 실험은 무엇인가?

문장을 짧게 해도 sigma LR/schedule, full fine-tuning, TRT-only projector, V/A 공동 MSE, Qwen token 간격은 그대로다. 그래서 short-sentence 필터만으로 큰 비대칭이 나타나야 할 이유는 없다. 길이 자체가 원인이었는지는 길이 외 조건을 맞춘 별도 결과가 필요하다.

**다음 대조는 현재 learned 조건에서 sigma LR만 `1e-3 → 5e-2`로 바꾸는 한 조건을 우선한다.** 동일 seed·fold·데이터·mask·full fine-tuning·warmup/decay·진단을 유지한다. 목적은 참조의 극단 sigma를 억지로 만드는 것이 아니라 현재 실험에서 sigma 이동이 LR에 얼마나 제한됐는지 확인하는 것이다. 이 한 변경은 참조의 schedule까지 재현하지 않는다.

| 결과 | 해석 |
|---|---|
| sigma 이동도 커지고 검증 성능도 좋아짐 | 현재 LR이 sigma 적응을 제한했다는 근거가 강화됨 |
| sigma만 커지고 성능은 비슷하거나 악화 | 큰 sigma 자체는 유용성의 증거가 아님 |
| 더 큰 LR에서도 다시 작은 비대칭으로 돌아옴 | 현재 과제 신호·공동 학습·입력 표현의 역할이 더 중요할 가능성 |
| 흔들리거나 불안정해짐 | 0.05가 현재 전체 학습과 맞지 않음; 참조 기본값의 직접 이식 실패 |

오른쪽으로 반드시 커질 것이라고 예측하지 않는다. 현재 update 방향을 보면 left가 더 커질 수도 있다. 실제 과거 옵션이 없으므로 원인을 하나로 확정하는 데에는 한계가 있지만, 이제 확인할 대조는 구체적이다.

## 9. 실행

이 파일은 서버 훈련용 wrapper가 아니라 두 커널과 schedule을 비교한 로컬 검증 코드다. 기존 학습 명령 형식은 바꾸지 않았다.

```text
python3 /Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_reference_20260916/compare_kernels.py
```

가중치 다운로드와 GPU가 필요 없으며 `comparison.json`을 다시 생성한다. 훈련 코드 수정·재학습·GitHub 업로드는 수행하지 않았다.
