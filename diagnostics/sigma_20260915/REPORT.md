# Sentence-only TRT redistribution: sigma 진단

검토일: 2026-09-15. 현재 소스 commit: `0cab59552207b1dd71b5a21c66d62c1439d18cdf`.

## 결론

**저장된 sentence-only 실험에서 sigma는 실제로 업데이트됐다. 문제는 완전한 동결이 아니라, 초기 폭 1 부근에서 움직이고 좌우 차이가 제한적이라는 것이다.** 전용 학습률 누락, FP32 sigma의 BF16 반올림, 캐시 내부의 redistribution 고정, 좌우 초기값 동일에 따른 강제 대칭을 주원인으로 볼 근거는 없다.

현재 증거로 우선 검토할 설명은 **VA 손실이 두 개의 전역 sigma를 강하게 구별하지 못하는 것**이다. 그 구체적 경로로는 TRT-only projection의 LayerNorm에 의한 크기 정보 약화, downstream에서 위치별 차이를 적게 사용하는 경우의 gradient 상쇄, 샘플마다 다른 최적 방향이 하나의 전역 파라미터에 합산되는 점이 있다. **이들은 코드와 수치 실험으로 성립 가능한 메커니즘을 확인한 가설이며, 실제 학습에서 어느 것이 지배적인지는 가중치와 gradient 기록 없이 확정할 수 없다.**

학습 코드는 변경하지 않았다. 진단 스크립트, 원본 로그 발췌, 계산 결과와 그림만 추가했다.

## 1. 사용한 실험 증거

- [원본 결과 ZIP](/Users/wansookim/Downloads/qwen3.5-0.8b_full_gaze_TRT_redistribution_asym-gaussian_sentence_only_no_iemocap_seed42_results_only.zip)
- ZIP SHA-256: `247feee3f9f154c43957c8e01c70bbe60c633f830ed020486cd32be5b7b951f2`.
- 17개 항목 CRC 검사 통과. 학습 가중치와 optimizer state는 포함되지 않았다.
- Qwen3.5-0.8B-Base, full fine-tuning, BF16, batch 16, MSE, seed 42.
- TRT-only `[0, 0, 0, 1, 0]`, gaze prefix, trainable parameter 752,533,828개 중 redistribution 2개.
- Backbone LR `6e-6`, **sigma 전용 LR `1e-3`, sigma weight decay `0`**.
- 두 fold 모두 선택된 checkpoint는 step 2,245, epoch 5. 설정은 10 epochs / max_steps 4,490이다.
- ZIP의 checkpoint 로그는 epoch 5까지만 있다. 6–10 epoch의 sigma 궤적은 이 ZIP으로 판단하지 않았다.

### 실제 변화

| 구간 | Fold 1 left | Fold 1 right | Fold 2 left | Fold 2 right |
|---|---:|---:|---:|---:|
| 초기값 | 1.000001 | 1.000001 | 1.000001 | 1.000001 |
| epoch 1 | 1.022537 | 0.962209 | 1.011300 | 1.008096 |
| epoch 2 | 1.043954 | 0.952133 | 1.032825 | 1.002858 |
| epoch 3 | 1.046790 | 0.944450 | 1.055685 | 0.997564 |
| epoch 4 | 1.022434 | 0.978982 | 1.054454 | 1.003894 |
| 선택된 epoch 5 | **1.025631** | **0.981501** | **1.073959** | **0.984700** |

Fold 1의 step 1,200에서는 `left=1.0521786212921143`, `right=0.9404751658439636`였다. 이후 초기값 방향으로 되돌아왔다. 저장 지점 사이 log-sigma 이동거리 대비 최종 순변화는 fold 1 left 약 16.7%, right 약 11.8%다. 이는 관측 지점 사이의 왕복 운동을 뜻하며, 기록되지 않은 개별 batch gradient의 부호나 손실 곡면을 증명하지는 않는다.

원본 로그의 학습률도 epoch 1에서 `0.001`, epoch 5에서 `0.0005555555555555556`으로 변한다. 전용 parameter group이 단순히 설정 파일에만 존재한 것이 아니다.

**로그 해석 주의:** `training_parameters.json` 및 architecture manifest의 `redistribution_sigma_left/right` 또는 `init_sigma_left/right`는 초기 설정이다. 거기에 `1.0`이 유지되는 것은 학습 실패의 증거가 아니다. 학습값은 `trainer_state.json`의 `log_history`와 가중치의 `log_sigma_left/right`를 봐야 한다.

![Sigma trajectory and kernel response](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_20260915/sigma_trajectory.png)

## 2. 코드 경로에서 확인한 사항

실제 경로는 아래와 같다.

```text
frozen ET2 / raw TRT cache
→ FP32 TRT + explicit first-subword gaze mask
→ trainable asymmetric redistribution
→ Linear(1, 128) → LayerNorm → GELU → Dropout(0.1)
→ Linear(128, hidden) → Dropout(0.3) → LayerNorm
→ eye_start, gaze prefix, eye_end, text
→ causal decoder의 마지막 유효 text token
→ regression head → hard sigmoid → VA MSE
```

| 의심 지점 | 확인 결과 | 근거 |
|---|---|---|
| sigma가 optimizer에서 누락 | 현재 코드에서 두 파라미터를 정확히 별도 group으로 선택; 실제 실험 로그도 변화 | [trainer.py:49](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/trainer.py:49) |
| LLM용 작은 LR을 그대로 사용 | sentence-only 실험은 실제 `1e-3` 사용 | 원본 manifest 및 로그 |
| sigma weight decay가 1로 끌어당김 | 해당 실험에서 sigma decay는 `0` | [trainer.py:91](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/trainer.py:91) |
| sigma가 BF16이라 미세 업데이트가 사라짐 | 파라미터 초기화와 kernel 계산 모두 FP32; 로그 변화 확인 | [redistribution.py:169](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/redistribution.py:169), [redistribution.py:220](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/redistribution.py:220) |
| ET2의 inference/cache가 sigma까지 동결 | 캐시에는 raw gaze를 저장하고 redistribution은 매 forward에서 수행 | [gaze.py:626](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/gaze.py:626), [model.py:307](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/model.py:307) |
| padding으로 TRT가 새어 나감 | source와 target를 둘 다 gaze mask로 제한하고 source별 정규화 | [redistribution.py:241](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/redistribution.py:241) |
| 좌우 sigma가 하나로 묶임 | 독립적인 scalar Parameter 두 개; 동일 초기값에서 gradient도 다를 수 있음 | [redistribution.py:169](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/redistribution.py:169) |
| min_sigma/clamp로 sigma=1 근처 gradient가 잘림 | 하한 `1e-6`이며 log distance의 극단 영역만 제한; sigma=1을 고정하는 clamp 없음 | [redistribution.py:235](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/redistribution.py:235) |
| causal mask 때문에 gaze에 접근 못함 | gaze는 text 앞에 있고 마지막 text 위치에서 pooling | [model.py:321](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/model.py:321), [model.py:375](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/model.py:375) |

### 커널이 비대칭을 학습할 수 있는지 직접 확인

현재 production `AsymGaussianRedistributor`에 synthetic TRT와 알려진 정답 커널을 사용했다. 초기값 `(1,1)`, AdamW LR `1e-3`, decay `0`, 3,000 steps, 정답 `(0.5,2.0)`이다.

- 결과: `(0.5002774, 2.0004466)`.
- MSE: `0.2811633 → 3.2658e-8`.
- 동일 초기값 `(1,1)`에서 위치를 구별하는 synthetic loss의 gradient: left `-1.2654141`, right `+0.2470188`.

따라서 동일 초기화 자체가 좌우 분리를 막지는 않는다. 이 실험은 **커널에 직접 정답을 주었을 때의 학습 가능성**을 확인한 것이며, 실제 VA 데이터에서 정답 sigma가 `(0.5,2)`라는 뜻은 아니다.

## 3. 원인 후보와 우선순위

### A. 우선 확인할 구조: scalar TRT 직후의 LayerNorm

[model.py:185](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/model.py:185)의 실제 코드:

```python
self.gaze_projector = nn.Sequential(
    nn.Linear(self.gaze_feature_count, self.gaze_projection_dim),
    nn.LayerNorm(self.gaze_projection_dim),
    nn.GELU(),
    nn.Dropout(first_dropout),
    nn.Linear(self.gaze_projection_dim, self.hidden_size),
    nn.Dropout(second_dropout),
    nn.LayerNorm(self.hidden_size),
)
```

TRT-only에서는 첫 입력이 scalar `x` 하나이므로 첫 Linear 출력은 `wx+b`다. LayerNorm은 이 벡터의 평균을 빼고 표준편차로 나눈다. `b=0`, `x>0`, epsilon을 무시하는 한계에서는

\[
\operatorname{LN}(wx)=\operatorname{LN}(w).
\]

즉 **TRT의 크기를 바꿔도 거의 같은 벡터가 나올 수 있다.** 현재 Linear에는 bias가 있으므로 완전한 불변성이 항상 성립하는 것은 아니다. 다만 `wx`가 bias보다 큰 영역에서는 같은 현상에 가까워진다. 최종 LayerNorm도 크기 차이를 줄일 수 있다.

같은 production projection 구조를 작은 hidden=128 모델에서 새로 초기화하여 확인했다. 아래 값은 실제 ET2 TRT 분포나 학습된 projection의 측정값이 아니다.

| synthetic TRT 변경 | projection 출력의 상대 L2 변화 |
|---|---:|
| 1 → 2 | 24.21% |
| 10 → 20 | 5.18% |
| 100 → 200 | 0.537% |
| 1000 → 2000 | 0.0538% |

첫 Linear의 bias를 0으로 놓는 메커니즘 확인용 조건에서는 `1 → 2` 변화가 약 `3.1e-6`의 상대 변화만 만들었다.

**왜 short sentence에서도 남는가:** 이 정규화는 문장 길이와 무관하게 각 gaze token의 feature 차원에서 작동한다. 짧은 문장만 남겨도 사라지지 않는다. 단, 실제 TRT가 bias의 영향이 큰 범위에 있다면 약화가 크지 않을 수 있으므로 현재 자료로 주원인 확정은 불가능하다.

### B. 총량을 보존하는 redistribution과 위치 구별이 약한 downstream

코드는 source `j`의 TRT를 destination `i`들로 나눈다.

\[
r_i=\sum_j P_{ij}(\sigma_L,\sigma_R)x_j,
\qquad \sum_iP_{ij}=1.
\]

따라서 유효 위치의 TRT 총합은 sigma와 관계없이 보존된다. downstream이 총합이나 평균 위주로 반응한다면 sigma를 바꿔도 손실이 거의 변하지 않을 수 있다. 실제 decoder는 비선형이고 위치를 사용하므로 총합 불변성이 곧 모델 출력 불변성은 아니다.

현재 커널의 synthetic probe에서:

- 출력 총합을 loss로 쓰면 left/right gradient는 FP32 오차 수준인 `1.29e-7 / 1.11e-7`.
- 같은 입력에 위치별로 다른 readout을 주면 `-1.2654 / +0.2470`.

더 정확히, `u_i = ∂L/∂r_i`라고 하면 각 source의 kernel gradient에는 **destination마다 다른 `u_i`**가 필요하다. `u_i`가 모든 destination에서 같으면 source별 gradient가 상쇄된다. 아래 Appendix에 식을 적었다.

**왜 short sentence에서도 남는가:** 유효 이웃 수를 늘리는 것과 decoder가 그 이웃의 TRT 차이를 VA 예측에 사용하는 것은 별개다. 또한 텍스트 경로와 약 7.5억 개의 trainable parameters가 VA 손실을 낮출 수 있으므로, sigma를 크게 바꾸어야 하는 압력이 약할 가능성이 있다. 실제 gaze 의존도가 낮다는 결론은 아직 측정하지 않았다.

### C. 모든 문장·위치·V/A가 공유하는 sigma 두 개

현재 sigma는 문장별/단어별 함수가 아니라 모델 전체에서 공유하는 scalar 두 개다. 학습 신호는 모든 위치와 샘플, valence 및 arousal에서 합산된다.

어떤 샘플에서는 left 폭을 늘리는 것이 유리하고 다른 샘플에서는 줄이는 것이 유리하면 평균 신호는 작아질 수 있다. V와 A의 선호 방향도 같다고 보장되지 않는다. 실제 로그의 왕복 변화와 양립하는 설명이다. **하지만 로그에는 개별 gradient가 없으므로 실제 상쇄 비율은 측정하지 못했다.**

현재 loss는 VA MSE이며, 좌우 폭을 다르게 만들라는 별도 정답은 없다. 최적의 전역 폭이 대칭에 가까울 가능성도 열어 둬야 한다. 폭의 큰 차이 자체를 성공 기준으로 삼을 근거는 없다.

### D. first-subword mask와 Qwen token 좌표 사이 간격

ET2 예측은 [gaze.py:459](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/gaze.py:459)에서 각 단어의 **첫 subword 위치에만** 배치된다. Redistribution 거리는 단어 순번이 아니라 원래 Qwen token index다. 중간 subword를 destination으로 쓰지 않으면서 좌표 간격은 유지한다.

유효 위치가 두 개이고 sigma=1인 synthetic 예:

| 두 유효 위치 간 token 간격 | 다른 위치로 이동하는 source mass | 그 mass의 right log-sigma 미분 |
|---|---:|---:|
| 1 | 37.7541% | 0.2350 |
| 2 | 11.9203% | 0.4200 |
| 3 | 1.0987% | 0.0978 |
| 4 | 0.0335% | 0.00536 |
| 5 | 0.000373% | 0.0000932 |

인접 단어라도 token 간격이 충분히 크면 혼합량과 gradient가 작아질 수 있다. **간격 2에서는 gradient가 간격 1보다 크므로, 간격이 증가할 때마다 gradient가 단조 감소한다고 해석하면 안 된다.** 실제 sentence-only 데이터의 유효 mask와 token 간격 분포는 저장되지 않아 영향의 크기는 미확인이다.

### E. 보조 후보: clipping·dropout·mixed precision

- 초기 로그의 `grad_norm`은 fold 1 step 200에서 `427.72`, fold 2에서 `1475.36`으로 크다. 현재 training 코드는 `max_grad_norm`을 따로 지정하지 않으며, 로컬 Transformers의 기본값은 `1.0`이고 전체 모델을 함께 clipping한다. 따라서 초기 clipping이 sigma gradient에도 영향을 줄 가능성은 있다.
- 다만 원 실행의 optimizer state, sigma별 gradient, 정확한 server Trainer 소스는 없고 로컬 Transformers 버전도 다르다. **이를 원인으로 확정하거나, 전체 norm만으로 sigma update가 1/1475가 된다고 계산하면 안 된다.** Adam은 1차·2차 moment로 gradient 크기를 정규화하므로 일정한 공통 scaling은 상당 부분 상쇄될 수 있다. 시간에 따라 달라지는 clipping과 epsilon 효과는 별도 측정이 필요하다.
- Gaze projection dropout은 `0.1 / 0.3`이다. 작은 sigma 효과와 stochastic noise를 구별하려면 같은 batch에서 dropout을 끈 sensitivity 측정이 필요하다.
- Sigma parameter와 kernel이 FP32여도 이후 projection/decoder 계산에 mixed precision이 사용될 수 있다. 작은 효과의 해상도 문제는 남지만, 현재 로그와 CPU BF16 통합 검사는 “BF16 때문에 sigma가 전혀 업데이트되지 않는다”는 설명을 지지하지 않는다.
- 저장된 OOF 예측에서 V의 정확한 0/1 값은 0개, A는 71/14,352개다. Hard-sigmoid가 전체 예측을 포화시킨 모습은 아니다. 학습 중 특정 batch의 포화까지 배제하는 자료는 아니다.

## 4. Short sentence 결과가 배제하는 것과 남기는 것

`--sentence-only`는 [filters.py:20](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/filters.py:20)의 세 source를 선택하며, 별도 길이 cutoff를 적용하지 않는다. **이번 길이 통계는 로컬 fold 파일이 아니라 해당 실험 ZIP의 OOF text 14,352개에서 직접 계산했다.** 로컬 fold SHA-256은 원 실험 manifest와 달라 실험 데이터의 대체물로 사용하지 않았다.

| source | 수 | 공백 기준 단어 수 중앙값 | 최대 |
|---|---:|---:|---:|
| EmoTales sentences | 1,395 | 10 | 90 |
| Emobank | 10,062 | 12 | 116 |
| fb | 2,895 | 11 | 148 |

공백 기준 1단어 이하는 396개(2.76%)다. ET2 segment 규칙을 적용한 뒤 정렬 전 segment가 1개 이하인 경우는 59개(0.41%)다. **두 수치 모두 Qwen 정렬 후 유효 gaze token 수와는 다르다.**

단일 유효 gaze token에서는 kernel이 정확한 identity가 되고 두 sigma gradient가 0인 것을 수치로 확인했다. 그러나 sentence-only 코퍼스의 일반적인 문장은 여러 단어이므로, “거의 모두 단어 하나여서 재분배할 대상이 없다”는 설명으로 이번 결과를 포괄할 수 없다. Projection의 크기 불변성, downstream의 약한 위치 의존성, 전역 sigma 신호의 상쇄, token 간격 문제는 모두 짧은 문장에서도 남는다.

## 5. 작은 sigma 변화와 redistribution 효과를 구분

문장 내부의 연속적인 유효 위치를 가정한 단일 source impulse의 분배량이다. 경계와 sparse mask에서 값은 달라진다.

| 조건 | 앞쪽 token들 | 자기 위치 | 뒤쪽 token들 |
|---|---:|---:|---:|
| 초기 sigma 1/1 | 30.05% | 39.89% | 30.05% |
| Fold 1 선택값 | 31.22% | 39.75% | 29.02% |
| Fold 2 선택값 | 32.79% | 38.76% | 28.45% |

초기 kernel 대비 learned kernel의 total variation distance는 fold 1 `0.01170`, fold 2 `0.02736`이다. 이 조건에서는 정규화된 분배량의 약 1.17% 또는 2.74%가 추가로 이동한 정도다.

그러나 **sigma=1이 raw TRT는 아니다.** 초기부터 약 60.1%를 이웃으로 분배하는 kernel이다. 따라서 현재의 raw-vs-learned 비교에는 대칭 smoothing 효과와 좌우 비대칭 학습 효과가 함께 섞여 있다. 비대칭 학습의 추가 효과를 보려면 고정된 `(1,1)` Gaussian control이 필요하다.

## 6. 다음 학습에서 최소한 기록할 것

가중치 없이 완료한 이번 조사로 원인을 단정해서 모델 구조를 바꾸지는 않았다. 다음 학습에서는 아래 측정이 먼저 필요하다.

1. **실제 입력·mask:** raw TRT의 분위수/부호, 유효 gaze token 수, 유효 위치 간 Qwen token 간격. Projection 포화 및 sparse mask 가설을 판별한다.
2. **Gradient와 실제 update:** sigma별 clipping 전/후 gradient, optimizer step 전/후 log-sigma 차이, sigma별 Adam moments. `.grad`는 optimizer가 zero_grad하기 전에 기록해야 한다. 매 200step의 sigma 값만으로는 0-gradient와 상쇄를 구분할 수 없다.
3. **손실 민감도:** 같은 모델·batch에서 dropout을 끄고 `(sigma_L,sigma_R)`만 고정해 `(1,1)`, 학습값, `(0.5,2)`, `(2,0.5)`의 TRT/projection/prediction/loss 차이를 측정한다. 모델의 다른 가중치는 고정하고 값을 복원한다. Transformer 자체를 `no_grad`로 감싸면 sigma backward 진단이 끊기므로 gradient 진단에서는 사용하지 않는다.
4. **상쇄 확인:** V-only와 A-only, 그리고 일부 개별 샘플의 sigma gradient를 따로 측정한다. `abs(mean(g)) / mean(abs(g))`가 작으면 방향 상쇄의 증거가 된다. 모든 `g=0`인 경우는 별도 분리한다.
5. **대조 실험:** 동일한 seed·초기화·fold·step 조건에서 raw TRT / 고정 대칭 Gaussian `(1,1)` / learned asym-Gaussian을 비교한다. 기존 CLI에는 고정 Gaussian 모드가 없으므로 이 조건은 후속 구현 대상이다. 별도 sigma LR은 이미 적용됐으므로 다시 “LR을 분리한 실험”으로 취급하면 안 된다.

판정 기준:

- TRT는 달라지는데 projection은 거의 안 바뀜 → projection의 입력 민감도를 우선 수정/검증.
- Projection은 달라지는데 VA 예측·loss는 거의 안 바뀜 → downstream의 gaze 의존도 및 sigma 식별성이 약함.
- 개별 샘플 gradient는 큰데 평균만 작음 → 전역 공유 sigma에서 방향 상쇄.
- Raw gradient는 있는데 실제 step이 거의 0 → optimizer/precision/clipping 경로 검토.
- 시작값을 달리했을 때 동일한 sigma 근처로 돌아옴 → 최적점 가능성의 추가 증거. 시작값 근처에 남으면서 loss도 같으면 평평한 손실 곡면 가능성. 현재 로그만으로는 둘을 구별할 수 없다.

큰 좌우 차이를 만들어내는 것보다, VA 성능 차이가 어디서 발생하거나 사라지는지 분리하는 것이 재학습 목적에 맞다.

## 7. 검증과 재실행

- 관련 기존 검사: **138 passed, 3 skipped**. Kernel gradient/finite difference, mask, cache 비변조, optimizer group/scheduler, 작은 native causal decoder의 checkpointing 및 FP32/BF16 full training/reload 등을 포함한다.
- Skip: CUDA 검사 1개, PEFT가 없는 환경의 LoRA 검사 2개.
- 로컬 환경: PyTorch `2.12.1`, Transformers `4.49.0`, CPU. 원 실험: Transformers `5.16.1`, CUDA. 실제 대형 Qwen 및 GPU 학습을 재현한 결과는 아니다.
- 진단 script 실행 완료. 모든 JSON 값 finite. 학습 소스 변경 없음.

```bash
cd /Users/wansookim/Documents/decoder_based_va_prediction
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  va_model_code/tests/test_redistribution.py \
  va_model_code/tests/test_redistribution_model.py \
  va_model_code/tests/test_redistribution_transformer.py

PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/private/tmp/sigma_audit_mpl \
XDG_CACHE_HOME=/private/tmp/sigma_audit_cache \
python diagnostics/sigma_20260915/diagnose_sigma.py \
  --archive /Users/wansookim/Downloads/qwen3.5-0.8b_full_gaze_TRT_redistribution_asym-gaussian_sentence_only_no_iemocap_seed42_results_only.zip
```

산출물:

- [diagnose_sigma.py](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_20260915/diagnose_sigma.py)
- [diagnosis.json](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_20260915/diagnosis.json)
- [Fold 1 원본 trainer state](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_20260915/heldout_fold1_trainer_state.json)
- [Fold 2 원본 trainer state](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_20260915/heldout_fold2_trainer_state.json)

## Appendix: source 정규화와 gradient 상쇄

유효 source/destination 집합 안에서 `theta_s=log_sigma_s`, `sigma_s=exp(theta_s)+epsilon`이고, `d_ij=i-j`다. `s=L`은 `d_ij<0`, `s=R`은 `d_ij>0`에서 작용한다. 중심에서는 거리 0이므로 미분도 0이다. 수치 안정화용 극단 clamp가 활성화되지 않는 일반 영역에서

\[
q_{ij}=\exp[-d_{ij}^2/(2\sigma_{s(i,j)}^2)],\qquad
P_{ij}=q_{ij}/\sum_k q_{kj}.
\]

\[
a^s_{ij}=\frac{\partial\log q_{ij}}{\partial\theta_s}
=\mathbf{1}_{s(i,j)=s}\frac{d_{ij}^2}{\sigma_s^2}
\frac{\exp\theta_s}{\sigma_s}.
\]

정규화 미분은

\[
\frac{\partial P_{ij}}{\partial\theta_s}
=P_{ij}\left(a^s_{ij}-\sum_k P_{kj}a^s_{kj}\right).
\]

`r_i=sum_j P_ij x_j`, `u_i=∂L/∂r_i`를 대입하면

\[
\frac{\partial L}{\partial\theta_s}
=\sum_j x_j\sum_i P_{ij}a^s_{ij}
\left(u_i-\sum_kP_{kj}u_k\right).
\]

따라서 `u_i`가 한 source의 모든 유효 destination에서 동일하면 해당 source의 sigma gradient는 0이다. 이것이 “TRT 총량에는 반응하지만 위치 변화에는 거의 반응하지 않을 때 sigma 신호가 약하다”는 설명의 정확한 조건이다. 실제 decoder가 그 조건을 만족하는지는 추가 측정이 필요하다.
