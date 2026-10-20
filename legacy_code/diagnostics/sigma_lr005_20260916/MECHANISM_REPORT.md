# 왜 LR 0.05에서 좁은 폭 또는 왼쪽 비대칭이 나왔는가

분석일: 2026-09-16. 기존 네 조건 비교에 더해 optimizer 기록과 현재 커널을 다시 검증했다.

## 판단 요약

1. **Fold 1:** 두 sigma가 토큰 간격보다 훨씬 작아지면서 재분배와 그 gradient가 거의 사라졌다. 높은 LR에서 크게 움직인 뒤 이런 영역에 들어갔다는 학습 경로는 확인된다. 그 위치가 최적해인지, 큰 update로 좁은 영역에 진입한 뒤 되돌아오기 어려워진 것인지는 구분되지 않는다.
2. **Fold 2:** gradient와 update가 계속 살아 있었으며 학습 중 선택된 폭은 left > right다. 현재 VA loss가 학습 경로상 이 방향을 만들었다는 것은 확인되지만, 왜 왼쪽 방향이 유리했는지는 고정 방향 대조 없이 단정할 수 없다.
3. **공통:** 이 sigma는 ET2가 예측한 TRT를 변환하는 전역 파라미터다. 사람의 fixation 주변 정보 획득 범위를 직접 추정하지 않는다. 입력 좌표계·목적함수·prefix 구조·전체 모델 공동 학습이 그 해석을 바꾼다.
4. 평균 성능 개선이 최종 sigma의 방향 때문에 생겼다는 인과관계는 확인되지 않았다. 학습 경로와 전체 가중치가 함께 달라졌기 때문이다.

[재현 코드](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_lr005_20260916/diagnose_mechanism.py), [상세 진단 수치](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_lr005_20260916/mechanism_diagnostics.json), [기존 성능·sigma 비교](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_lr005_20260916/REPORT.md).

## 1. Fold 1: 작은 sigma에서는 실제 학습 신호도 사라진다

새 실행의 epoch별 폭:

| 시점 | σ left | σ right |
|---|---:|---:|
| 초기 | 1.000001 | 1.000001 |
| epoch 1 | 1.500862 | 0.364902 |
| epoch 2 | 0.247894 | 0.253632 |
| epoch 3 | 0.253567 | 0.271953 |
| epoch 4 | 0.227875 | 0.173729 |
| 선택 epoch 6 | 0.219749 | 0.173731 |
| epoch 10 | 0.213889 | 0.173724 |

증가는 단조롭지 않았다. 예를 들어 기록된 step 400 직전 left는 약 2.754까지 올라갔다. 이후 양쪽 폭이 작아졌고 right는 0.1737 부근에서 거의 유지됐다. 기록 간격이 50 step이므로 이 시점들이 전체 궤적의 실제 극값이라고 단정하지 않는다.

### 토큰 거리 1과 Gaussian 폭 0.17의 관계

정규화 전 Gaussian은

\[
k(d,\sigma)=\exp\left[-\frac{d^2}{2\sigma^2}\right].
\]

현재 \(\sigma=\exp(\theta)+\epsilon\), \(\epsilon=10^{-6}\)이므로 정확한 미분은

\[
\frac{\partial k}{\partial\theta}
=k\frac{d^2}{\sigma^2}\left(1-\frac{\epsilon}{\sigma}\right).
\]

자기 위치는 d=0이라 k=1이고 폭에 대한 미분이 0이다. 다른 token과는 최소 d=1이므로 sigma가 작아지면 이웃의 k와 미분이 지수적으로 작아진다. 정규화 후에도 자기 위치 비중이 거의 1이 되고 변화가 작아진다.

| σ | d=1 가중치 | d=1 가중치의 log-sigma 미분, 약 |
|---|---:|---:|
| 1.0 | 0.6065 | 0.6065 |
| 0.5 | 0.1353 | 0.5413 |
| 0.3 | 0.003866 | 0.04295 |
| 0.219749 | 3.186e-5 | 6.597e-4 |
| 0.173731 | 6.390e-8 | 2.117e-6 |

유효 위치 사이에 subword 간격이 있어 d=2 이상이면 더 작아진다. 실제 sigma=0.173731은 코드의 최소값 1e-6에 붙은 것이 아니다. **코드상 하한과 무관하게 이산 token 간격 때문에 실질적으로 재분배가 꺼질 수 있다.** Raw TRT feature 자체는 계속 gaze prefix에 들어간다. 재분배가 약해졌다는 사실을 gaze 전체를 사용하지 않는다는 뜻으로 해석하면 안 된다.

### 실제 training gradient가 이를 뒷받침한다

아래는 epoch 4 초과부터 10까지, 50 step마다 기록된 54개 training update의 통계다. 고정 문장 2개 probe의 통계가 아니다.

| Fold 1 right | LR 0.001 | LR 0.05 |
|---|---:|---:|
| clipping 전 gradient 절댓값 중앙값 | 6.9565e-6 | 7.0921e-10 |
| 실제 log-sigma 변화 절댓값 중앙값 | 1.7630e-5 | 2.3842e-7 |
| gradient=0인 기록 | 0/54 | 0/54 |
| 실제 parameter 변화=0인 기록 | 0/54 | 11/54 |

**LR을 50배 높였는데도 후기 right gradient는 약 1만 배 작아졌다.** 작은 폭에 들어간 이후에는 더 높은 LR이 지속적인 큰 update를 보장하지 않는다. clipping 전 값이 이미 작으므로 이 현상을 clipping만으로 설명할 수도 없다. 새 실행에서는 이 구간 22/54 step에 clipping이 있었으나, 나머지 step에도 매우 작은 gradient가 존재한다.

이것은 autograd가 끊겼다는 뜻이 아니다. 기록된 gradient는 여전히 유한한 비영 값이고, 함수 자체가 sigma에 거의 반응하지 않는 영역이다.

## 2. Adam의 누적 상태가 폭 축소와 후기 정체에 관여했다

새 fold 1의 step 1450 직전 right 폭은 약 0.177314였다.

- 그 step의 clipping 후 gradient: `-3.1913e-10`. 현재 batch만 따라 gradient descent를 하면 폭을 늘리는 방향이다.
- 하지만 저장된 이전 moment를 사용한 Adam 1차 moment는 여전히 양수였고, 그 update에서 실제 log-sigma는 `-0.00203443` 감소했다.
- 현재 gradient가 이미 작아진 뒤에도 직전 여러 step의 누적 상태가 축소 방향을 유지하는 사례다. 이 한 step이 전체 축소의 유일한 원인이라는 뜻은 아니다.

Step 1800에서는 sigma LR이 여전히 **0.033296**이었는데도 실제 right update가 0이었다. 기록된 gradient와 moment로 재구성한 update는 약 `-2.827e-8`, 해당 FP32 log-sigma 값의 인접 표현 간격은 약 `1.192e-7`이다. 반올림으로 표현되지 않을 정도로 작은 update와 일치한다. 이는 **sigma 파라미터의 BF16 저장 문제라는 설명과 다르다.** 커널 파라미터는 FP32이며, 작은 gradient와 Adam 상태가 update를 FP32 정밀도 부근까지 줄인 사례다.

이 재구성은 beta1=.9, beta2=.999, epsilon=1e-8을 가정했다. 해당 값들은 결과 ZIP에 직접 기록되지 않았다. 두 고학습률 fold의 모든 기록과 비교한 delta 최대 오차는 각각 `1.28e-7`, `9.10e-8`로 FP32 반올림 규모이며, 식의 일관성을 뒷받침한다.

따라서 fold 1의 평평한 sigma 곡선을 곧바로 '최적 span에 수렴했다'고 읽으면 안 된다. **near-identity 영역의 낮은 민감도, 이전 Adam 상태, 점차 줄어드는 LR**이 함께 정체를 만들 수 있다. 진짜 최적점인지 확인하려면 다른 가중치를 고정한 sigma별 전체 평가 loss 비교가 필요하다.

## 3. Fold 2는 같은 식으로 정체한 것이 아니다

Fold 2의 epoch 4 이후 right gradient 절댓값 중앙값은 `4.2745e-6`, 실제 log-sigma update 절댓값 중앙값은 `9.2137e-4`다. 기록된 54개 update 중 0인 것은 없다. 같은 구간 clipping도 없다.

Left는 0.5–1.1 부근에서 계속 움직였고, right도 0.28에서 0.38 부근으로 회복했다. 선택 모델은 `1.136265 / 0.356580`이며, fold 1처럼 전체 커널이 identity로 사라진 상태가 아니다. 이 사례에는 '큰 LR 때문에 gradient가 사라졌다'만 적용해서 설명할 수 없다.

두 fold의 training 문장과 minibatch 경로는 다르다. 동일 seed가 동일 loss surface나 동일 최종 sigma를 보장하지 않는다. 특정 fold의 label 통계나 길이 분포가 방향 차이의 원인이라는 주장은 현재 진단만으로 확인되지 않았다.

## 4. 읽기 span과 현재 sigma 사이에는 추가 가정이 있다

사용자가 제공한 논문 문장 원문:

> Importantly, in left-to-right reading, the span typically extends further to the right of fixation than to the left

이 문장은 fixation을 기준으로 한 정보 획득 범위의 비대칭을 말한다. 여기서 **'ET2가 예측한 TRT를 재분배하는 전역 Gaussian을 VA loss로 학습하면 오른쪽 sigma가 더 커져야 한다'**로 넘어가려면 추가적인 전이 가정이 필요하다. 위 원문 자체가 현재 모델의 최적 sigma를 보장하는 것은 아니다. 원문 전체 논문의 추가 주장으로 확대해서 해석하지 않았다.

현재 pipeline은 코드상 다음과 같다.

1. 문장을 입력으로 고정 ET2가 word feature를 예측한다. 현재 선택 feature는 TRT 하나다.
2. Word feature를 Qwen의 첫 subword 위치에 놓는다. 인간의 시선 좌표나 fixation 순간별 위치를 입력하지 않는다.
3. Gaussian 두 폭을 모든 문장·V/A 출력에 공유하고, token index 거리로 재분배한다.
4. 전체 gaze sequence를 text 앞의 prefix로 넣는다.
5. 마지막 text 위치의 hidden state로 V/A를 예측하고 MSE를 줄인다. 사람의 span과 맞추는 loss 항은 없다.

근거: [ET2 mapping](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/gaze.py:421), [재분배 후 projector](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/model.py:302), [prefix 순서](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/packing.py:122), [pooling과 VA 출력](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/model.py:373), [학습 loss](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/trainer.py:166).

따라서 이 sigma는 현재 실험에서 **예측된 gaze feature의 변환 방식**으로 해석하는 것이 안전하다. 모든 text token 앞에 전체 gaze prefix가 있으므로 실제 fixation이 옮겨가며 주변 제한된 범위만 읽는 과정을 구현한 것도 아니다. Sigma가 오른쪽 비대칭을 보이면 흥미로운 일치가 되지만, 그 일치가 필연적인 구조는 아니다.

## 5. Left 방향이 실제로 무엇을 바꾸는지 검증했다

### 좌우 이름이 뒤집힌 구현 오류는 발견되지 않았다

현재 코드 [redistribution.py:233](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/redistribution.py:233)에서 차이는 target index minus source index다. 음수이면 left를 선택한다. 현재 커널에 중앙 위치의 unit impulse를 입력해 확인했고, 폭을 좌우로 바꾸면 출력도 좌우 대칭으로 바뀌었다.

\[
y_i=\sum_j W_{ij}x_j.
\]

- **Source j 기준:** left 폭을 키우면 j의 TRT가 이전 위치 i<j로 더 퍼진다.
- **Target i 기준:** i의 결과에는 뒤쪽 source j>i의 TRT가 더 섞인다.

따라서 왼쪽 재분배는 뒤쪽 단어의 gaze 신호를 앞쪽 gaze 위치에 섞는 연산으로도 볼 수 있다. 이는 구현된 방향을 바꾸어 부르는 근거가 아니다. 사람의 오른쪽 perceptual span을 재현했다고 주장할 근거도 아니다. 단지 해당 연산이 모델 입력에서 어떤 정보 혼합을 일으키는지를 구분한다.

### Source별 정규화는 경계에서 위치별 값도 바꾼다

현재는 한 source가 내보내는 총 질량을 1로 정규화한다. 전체 TRT 합은 보존하지만, 모든 token의 TRT가 같을 때 모든 출력도 같게 만드는 정규화는 아니다.

실제 커널에 길이 17의 `[1, 1, ..., 1]`을 넣고 fold 2의 선택 sigma `1.136265 / 0.356580`을 적용하면:

- 첫 위치: **1.508525**
- 가운데 위치: 약 **1.0**
- 마지막 위치: **0.529806**
- 전체 합: **17**, 보존됨.

이는 마스크가 전부 올바른 상태에서도 나타나는 수학적 경계 효과이며, 그 자체를 구현 버그라고 판단하지 않는다. 커널은 퍼짐 방향뿐 아니라 문장 앞뒤의 feature 크기도 바꾼다. Decoder가 이런 순서·경계 효과를 이용했을 가능성이 있지만, 실제 VA 성능 개선의 원인이라고 확인된 것은 아니다.

Causal prefix 내부의 처리 순서도 또 다른 방향성이다. 다만 모든 text 위치 앞에 전체 gaze prefix가 있으므로, '왼쪽으로 옮겨야만 text가 뒤쪽 gaze를 볼 수 있다'는 설명은 틀리다. 해당 정보는 원래도 prefix에 있다. 효과가 있다면 prefix 내부 표현과 위치별 이용 방식의 차이이며 이를 분리하는 대조는 아직 없다.

## 6. 최종 sigma와 성능 개선을 바로 인과적으로 연결할 수 없는 이유

실험은 \(L_{VA}(\phi,\sigma_L,\sigma_R)\)를 공동 최적화한다. \(\phi\)에는 projector, backbone, VA head가 포함된다. Sigma LR을 바꾸면 gaze 입력이 달라지고, 그 입력으로 학습하는 전체 가중치의 경로도 달라진다.

따라서 비교 중인 것은 서로 다른 \(\phi\)를 가진 모델들이다. 최종 checkpoint의 성능이 좋다는 사실은 그 checkpoint의 sigma만이 최적이라는 뜻이 아니다. Epoch별 성능 향상과 sigma 이동의 상관도, 동시에 학습된 다른 가중치를 고정하지 않으면 sigma의 인과효과가 아니다.

Fold 1이 좋은 예다. 선택 커널은 자기 위치에 질량 99.9968%를 남겨 거의 raw TRT를 사용하지만, 처음부터 raw TRT로 학습한 모델과 학습 경로가 같지는 않다. 초기에 변하는 재분배가 입력 변형이나 정규화와 비슷한 역할을 했을 가능성은 있으나, 현재 결과만으로 그런 메커니즘을 확정하지 않는다.

### 현재 probe가 말하는 범위

새 fold 2의 선택 모델에서 다른 가중치를 고정하고 sigma를 바꾸면, 두 훈련 문장의 평균 MSE는 다음과 같다.

| 개입 | 두 문장 MSE |
|---|---:|
| 학습된 1.136265 / 0.356580 | 0.000983201 |
| 오른쪽형 0.5 / 2 | 0.000865517 |
| Raw TRT | 0.000865517 |

이 표본에서는 오른쪽형의 MSE가 학습된 폭보다 낮다. 하지만 raw 개입과 오른쪽형의 예측도 정확히 같다. 따라서 오른쪽형의 고유한 이득이라는 증거가 아니다. **전체 데이터에서 오른쪽형이 나쁘다고 확정할 수도 없다.** Probe는 fold당 같은 훈련 문장 2개이며 held-out 전체 실험이 아니다.

TRT 변화를 본체가 약하게 반영하는 경우, 서로 다른 sigma가 같은 예측을 만들 수 있다. BF16에서 작은 변화가 표현되지 않는 현상도 이 probe 해석에 포함된다. 기존 분석에서 언급한 LayerNorm은 그 가능한 경로 중 하나이지만, 참조 코드에도 있으므로 두 실험 차이의 단독 원인으로 확정하지 않는다.

## 7. 확인된 설명과 남은 설명

| 판단 | 근거 수준 |
|---|---|
| Fold 1의 작은 폭은 재분배와 폭의 미분을 거의 없앤다 | 수식과 실제 커널 검증 |
| Fold 1의 후기 right gradient는 매우 작고 일부 update는 FP32 표현 한계 부근이다 | training update 기록과 Adam 재구성 |
| Fold 2는 여전히 sigma gradient/update가 살아 있다 | training update 기록 |
| 좌우가 뒤집혀 기록된 것이다 | 현재 커널 impulse 검사에서 지지되지 않음 |
| 현재 loss가 사람의 오른쪽 span을 직접 복원한다 | 코드에 그런 supervision이나 constraint가 없음 |
| Left 재분배가 prefix 처리나 경계 위치 효과 때문에 유리하다 | 가능한 메커니즘; downstream 대조 미실시 |
| 큰 LR이 fold 1의 진짜 최적점을 지나쳐 정체시켰다 | 가능하지만 최적점에 대한 전체 sigma 개입 평가 필요 |
| 오른쪽 비대칭은 이 과제에서 원천적으로 쓸모없다 | 현재 자료로 결론 불가 |

성능 개선은 raw 대비 평균 MSE 1.63%, 기존 learned 대비 1.08%다. 기존 learned와 비교하면 fold 1은 1.19% 악화, fold 2는 3.40% 개선이다. 단일 seed와 같은 held-out fold에서의 best-epoch 선택까지 고려하면, 작은 차이를 하나의 인지적 메커니즘의 증거로 해석하기에는 부족하다.

## 8. 다음으로 가장 직접적인 검증

방향의 유용성을 검증하려면 **같은 두 폭을 좌우만 바꾼 고정 커널 대조**가 우선이다. 예를 들어 `(left, right)=(0.5, 2.0)`과 `(2.0, 0.5)`를 동일 초기화·데이터·학습 조건에서 각각 처음부터 학습한다.

이 비교는 두 조건의 kernel 모양을 서로 거울상으로 맞추고, 큰 sigma LR이나 좁은 폭으로의 축소를 제거해 방향 차이를 검증한다. 기존의 대칭 fixed 1/1 실험만으로는 이 질문에 답할 수 없다. 단, 이 한 폭 쌍의 결과가 모든 가능한 sigma에서 방향의 우열을 증명하는 것은 아니다.

현재 CLI의 `fixed-gaussian`은 좌우 동일 폭만 허용하므로, 이 제안은 바로 실행 가능한 기존 명령이라고 제시한 것이 아니다. 이번 분석에서는 training 코드를 변경하거나 새 학습을 시작하지 않았다. 먼저 원인에 대한 판단과 필요한 대조를 명확히 했다.
