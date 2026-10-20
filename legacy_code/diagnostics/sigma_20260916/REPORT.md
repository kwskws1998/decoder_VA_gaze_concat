# Sentence-only sigma 실험 결과 및 원인 진단

분석일: 2026-09-16. 대상은 사용자가 제공한 세 조건의 실제 실행 결과다.

후속 자료: [첨부 supplementary 코드와의 비교](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_reference_20260916/REPORT.md)에서 참조 기본 sigma LR 0.05, custom schedule, projector 동결 경로를 추가 확인했다. 아래 실제 결과 수치는 유지되지만, 참조와의 차이를 설명하는 LayerNorm 가설과 다음 실험의 우선순위는 후속 보고서의 판단을 따른다.

## 결론

**Sigma 학습 경로는 작동한다. 두 fold 모두 작은 비대칭을 학습했다.** 이번 seed 42에서는 학습형의 전체 OOF 평균 MSE가 raw보다 0.55%, 고정형보다 0.75% 낮지만, 개선 방향이 fold마다 일치하지 않아 안정적인 성능 향상으로 판단할 수 없다.

관측된 작은 변화는 다음 증거와 함께 해석해야 한다.

1. 학습률이 양수인 진단 step 178개에서 left/right log-sigma가 모두 실제로 업데이트됐다. 고정, optimizer 누락, 모든 업데이트가 반올림으로 사라지는 현상은 이 기록과 맞지 않는다.
2. 첫 epoch에는 sigma gradient도 강한 전역 clipping을 받았다. 3 epoch 이후에는 대부분 clipping되지 않으므로, 후기의 작은 변화까지 clipping만으로 설명할 수 없다.
3. 다른 가중치를 유지한 sigma 개입에서 TRT와 projector 출력 변화에 비해 최종 VA 예측 변화가 작았다. 다만 fold당 첫 훈련 문장 2개에 한정된 관측이다.
4. 일부 문장·출력 차원에서 sigma gradient가 서로 반대 방향이고, 학습 후반 sigma도 단조롭게 변하지 않는다. 공유 sigma에 대한 상충하는 학습 신호와 일치하지만, 전체 데이터의 상쇄량은 이번 소규모 probe로 확정할 수 없다.

## 1. 자료 및 검증

- 입력: `/Users/wansookim/Downloads/sigma_sentence_only_seed42_results_only.zip`
- SHA-256: `eb5ddd199076fefb4141e6da6b1455051c0c04737180d66e361262dc9d56b9e5`
- 외부 ZIP과 실험별 내부 ZIP 3개 CRC 검사 통과.
- 내부 ZIP마다 결과 파일 22개와 manifest 1개. 총 66개 결과 파일의 크기와 SHA-256이 manifest와 일치.
- 세 실험 모두 콘솔 로그에 해당 실행의 완료 메시지 1개, traceback 0개.
- 저장된 결과의 엄격한 검증 통과: fold별/OOF 예측과 지표, 데이터 출처 수, fold 조합, manifest, architecture, best checkpoint 소속.
- 예측 TSV로 재계산한 OOF 지표의 최대 절대 오차는 `3.33e-16`.
- 세 조건의 `(index, held_out_fold, text, dataset_of_origin, valence, arousal)`가 정확히 일치.
- 6개 fold 실행 모두 10 epoch, 4,490 optimizer step. 각 fold에 probe 12개와 optimizer 진단 90개가 존재.
- 세 조건에서 공통 sigma 개입 4종의 학습 전 예측이 각 fold 내에서 정확히 일치. 이는 기록된 초기 probe의 일치이며, 가중치 전체의 바이트 단위 일치를 검증했다는 뜻은 아니다.

### 사용 데이터 및 조건

| 데이터셋 | 문장 수 |
|---|---:|
| EmoTales sentences | 1,395 |
| Emobank | 10,062 |
| fb | 2,895 |
| 합계 | 14,352 |

`sentence_only=true`, TRT-only, full fine-tuning, BF16, batch 16, accumulation 1, max length 200, backbone LR `6e-6`, seed 42, RTX 4090 조건이 동일하다. 저장된 원본 fold SHA-256도 세 조건에서 동일하다. `no_iemocap=false`여도 sentence-only 필터의 제외 목록에 IEMOCAP이 포함돼 실제로 사용되지 않았다.

설정의 차이는 실행 이름/경로, redistribution 계약, sigma 전용 LR/weight decay뿐이다. 고정형은 `trainable=false`, 학습형은 `trainable=true`, sigma LR `1e-3`, weight decay `0`이다. 최적화 대상은 `log_sigma`이며, 보고되는 폭은 `exp(log_sigma) + 1e-6`이다.

## 2. 최종 OOF 성능

아래 수치는 최종 선택 모델로 만든 전체 14,352개 예측의 지표다. epoch 10의 evaluation 로그나 fold 지표의 단순 평균을 사용하지 않았다. 평균 MSE/MAE/Pearson/CCC는 valence와 arousal 두 값의 산술평균이다.

| 조건 | 평균 MSE ↓ | 평균 MAE ↓ | Pearson V ↑ | Pearson A ↑ | 평균 CCC ↑ |
|---|---:|---:|---:|---:|---:|
| Raw TRT | 0.006575700 | 0.056296581 | 0.762269 | 0.824560 | 0.782912 |
| 고정 sigma | 0.006588945 | 0.055901954 | 0.764650 | 0.823109 | 0.783044 |
| 학습 sigma | 0.006539251 | 0.055776673 | 0.764870 | 0.825056 | 0.784833 |

학습형은 raw 대비 평균 MSE가 `0.0000364493` 낮다(상대 감소 0.5543%). 평균 Pearson은 `0.00154878`, 평균 CCC는 `0.00192149` 높다. 고정형 대비 평균 MSE 상대 감소는 0.7542%다. 학습형의 valence MSE `0.005141710`은 고정형 `0.005139614`보다 미세하게 높으므로 모든 개별 지표에서 학습형이 우세한 것은 아니다.

### Fold별 평균 MSE

| 조건 | held-out fold 1 | held-out fold 2 |
|---|---:|---:|
| Raw TRT | 0.006688176 | 0.006463256 |
| 고정 sigma | 0.006824082 | 0.006353872 |
| 학습 sigma | 0.006600034 | 0.006478485 |

학습형은 fold 1에서 두 대조군보다 좋고, fold 2에서는 두 대조군보다 나쁘다. 이 차이가 작다는 점과 단일 seed라는 점을 함께 고려해야 한다.

### 데이터셋별 평균 MSE

| 데이터셋 | Raw TRT | 고정 sigma | 학습 sigma |
|---|---:|---:|---:|
| EmoTales sentences | 0.00724340 | 0.00723766 | 0.00705195 |
| Emobank | 0.00644766 | 0.00652265 | 0.00644633 |
| fb | 0.00669898 | 0.00650678 | 0.00661516 |

가장 큰 데이터셋인 Emobank에서는 학습형과 raw가 거의 같다. fb에서는 고정형이 가장 낮다. 데이터셋 전반에 일관된 학습형 우위를 보여주는 결과는 아니다.

## 3. 실제 sigma와 체크포인트 선택

| 조건 | fold 1 선택 epoch | fold 2 선택 epoch |
|---|---:|---:|
| Raw TRT | 6 | 5 |
| 고정 sigma | 4 | 6 |
| 학습 sigma | 4 | 6 |

학습형의 최종 선택 모델은 다음과 같다. `train_end_selected_model` probe의 sigma와 예측이 해당 best epoch의 probe와 정확히 일치하는 것도 확인했다.

| held-out fold | 선택 epoch / step | 선택 모델 σ left | 선택 모델 σ right | epoch 10 σ left | epoch 10 σ right |
|---|---|---:|---:|---:|---:|
| 1 | 4 / 1,796 | 1.054003 | 0.936260 | 1.053444 | 0.938549 |
| 2 | 6 / 2,694 | 1.061833 | 0.967911 | 1.064912 | 0.973909 |

초기 폭은 양쪽 모두 약 `1.000001`이다. 선택된 left는 약 5.4–6.2% 커졌고 right는 약 3.2–6.4% 작아졌다. fold 2에서는 epoch 3에 `1.088320 / 0.921527`까지 벌어졌다가 다시 가까워졌다. 단순히 학습 내내 1에 붙어 있었던 상황이 아니다.

![Epoch별 sigma; 별표는 최종 선택 모델](/Users/wansookim/Documents/decoder_based_va_prediction/diagnostics/sigma_20260916/sigma_trajectory.png)

`fixed-gaussian`의 `1.000001 / 1.000001` 유지와 진단된 update 0은 의도된 동작이다. 고정형 probe의 local gradient는 임시로 교체한 학습 가능한 커널에 대한 민감도이며, 고정형 파라미터가 실제로 학습됐다는 뜻이 아니다.

### Kernel 관점의 크기

경계나 마스크가 없는 연속적인 정수 token 위치를 가정해 Gaussian을 정규화하면 다음과 같다. 실제 문장의 비연속적인 유효 위치에서는 달라질 수 있는 설명용 계산이다.

| 폭 | 이전 token 방향 질량 | 자기 위치 질량 | 이후 token 방향 질량 |
|---|---:|---:|---:|
| 초기 1/1 | 30.05% | 39.89% | 30.05% |
| fold 1 선택 폭 | 32.91% | 40.09% | 27.00% |
| fold 2 선택 폭 | 32.66% | 39.31% | 28.03% |

초기 1/1부터 이미 상당한 평활화가 적용된다. 이후 학습으로 바뀐 것은 그 위에 더해진 몇 percentage point 정도의 방향별 비중이다. left/right는 화면 좌우가 아니라 정렬된 Qwen token index의 앞/뒤를 뜻한다.

## 4. 원인별 증거

### 4.1 Sigma의 gradient 및 optimizer 연결은 살아 있다

- 학습형의 각 fold에서 진단된 90개 step 모두 left/right gradient가 유한한 비영 값이다.
- 첫 step은 warmup으로 sigma LR이 0이며 delta도 0이다. 이후 기록된 89개 step에서는 두 파라미터 모두 delta가 비영이다. 두 fold 합계로 LR이 양수인 178개 optimizer step이다.
- `1e-3`은 sigma LR의 설정값이며 warmup/decay 스케줄이 적용된다. epoch 10 종료 로그의 LR 0은 스케줄 종료에 따른 것으로, 학습 내내 LR이 0이었다는 뜻이 아니다.
- 기록된 실제 `delta_log_sigma`의 중앙값 크기는 대략 `3.4e-5`에서 `5.5e-5` 수준이다. BF16 출력 때문에 log-sigma 업데이트 자체가 전부 반올림돼 사라진 상황은 아니다.
- 기록된 clipped gradient, 이전 Adam moment, LR로 AdamW 식을 대조하면 delta의 최대 절대 차이는 `3.90e-9` 미만이다. 이 대조는 beta1=0.9, beta2=0.999, epsilon=1e-8을 가정한다. 해당 hyperparameter가 ZIP에 직접 기록된 것은 아니므로 식의 일관성 검증으로 해석한다.

ET2 캐시는 재분배 전 feature를 저장하고, redistribution은 그 뒤 다시 계산한다. 위의 비영 gradient와 실제 parameter delta도 캐시 때문에 sigma 학습 경로 전체가 끊겼다는 설명과 맞지 않는다.

### 4.2 초기에는 강한 전역 gradient clipping이 존재한다

진단 간격은 첫 step 및 이후 50 step마다이며, 전체 step을 기록한 통계가 아니다.

| 구간 / 지표 | fold 1 | fold 2 |
|---|---:|---:|
| 첫 epoch에 기록된 step 수 | 9 | 9 |
| 첫 epoch의 clipping 적용 비율 | 9/9 | 9/9 |
| 첫 epoch의 gradient 유지 배율 중앙값 | 0.002335 | 0.000843 |
| 3 epoch 이후 clipping 적용 | 0/63 | 2/63 |

첫 epoch의 sigma gradient는 중앙값 기준 원래 값의 약 0.23%, 0.084%만 남는다. 전역 norm clipping이 backbone의 큰 gradient와 함께 sigma에도 적용되는 것이 관찰된다.

그러나 Adam은 gradient의 1차·2차 moment로 정규화하므로, gradient가 1,000배 작아졌다는 사실을 parameter update가 1,000배 작아졌다는 뜻으로 바꾸면 안 된다. 실제 초기 parameter update도 존재한다. 특히 3 epoch 이후에는 대부분 clipping이 없는데도 sigma의 순변화가 작아지므로, clipping은 초기 영향 요인이며 후기 정체의 단독 설명은 아니다.

### 4.3 Sigma 개입에 대한 출력 민감도가 작다

각 fold의 최종 선택 모델에서 다른 가중치를 고정하고, 동일한 첫 훈련 문장 2개에 대해 sigma만 바꿨다. dropout은 껐고 BF16 autocast를 사용했다. 아래 상대 L2는 `norm(변경-기준)/norm(기준)`이며 서로 다른 표현 공간 사이의 수치를 직접 인과적인 감쇠 계수로 간주하지 않는다.

| 개입 | fold | TRT 상대 L2 변화 | projector 상대 L2 변화 | VA 예측 RMS 변화 |
|---|---:|---:|---:|---:|
| 선택 폭 → 1/1 | 1 | 2.73% | 0.764% | 0.002184 |
| 선택 폭 → 1/1 | 2 | 2.99% | 0.955% | 0.001381 |
| 선택 폭 → 0.5/2 | 1 | 34.24% | 13.83% | 0.002762 |
| 선택 폭 → 0.5/2 | 2 | 45.24% | 16.45% | 0.007873 |
| 선택 폭 → 2/0.5 | 1 | 28.20% | 5.88% | 0.002184 |
| 선택 폭 → 2/0.5 | 2 | 36.35% | 9.62% | 0.000977 |

VA 예측은 정규화된 0–1 척도다. 재분배가 입력을 거의 바꾸지 않았던 것은 아니다. 재분배로 바뀐 gaze에 비해 최종 예측이 작게 움직였다. 그 설명으로 TRT 1차원 입력 뒤의 Linear/LayerNorm, gaze prefix에 대한 decoder의 낮은 민감도 등을 검토할 수 있다. 다만 중간 층별 개입이나 LayerNorm 제거 대조를 하지 않았으므로 어느 층이 주원인인지 확정할 수 없다.

BF16 출력에서 양자화된 예측 간격도 보인다. FP32 대조가 없으므로 낮은 민감도 중 BF16의 기여와 모델 자체의 민감도를 분리할 수 없다. 이 사실과 sigma parameter의 실제 업데이트 여부는 구분해야 한다.

### 4.4 공유 sigma의 gradient가 일부 표본에서 상쇄된다

최종 선택 모델의 configured probe에서 `abs(mean(g))/mean(abs(g))`는 다음과 같다. 각 값은 문장 2개 × V/A 2개, 총 4개 loss 성분에 대한 log-sigma gradient로 계산했다.

| fold | left 방향 일치도 | right 방향 일치도 |
|---|---:|---:|
| 1 | 0.6587 | 0.8882 |
| 2 | 0.0850 | 0.0514 |

fold 2에서는 개별 gradient의 절대크기에 비해 합쳐진 gradient가 작다. 예를 들어 두 번째 문장의 left gradient는 valence에서 `+2.0433e-4`, arousal에서 `-2.6354e-4`이다. 두 출력이 같은 전역 sigma를 서로 반대 방향으로 밀 수 있음을 보여준다.

fold 1에서는 이런 상쇄가 훨씬 약하다. 따라서 모든 데이터에서 대부분의 gradient가 상쇄된다고 일반화하지 않는다. 3 epoch 이후 기록된 minibatch gradient도 양수·음수가 모두 나타나며, sigma의 epoch별 궤적이 되돌아오는 현상과 일치한다. 현재 자료는 단순한 학습 불능보다 작고 상충하는 학습 신호를 뒷받침한다.

## 5. 판단의 범위와 다음 확인

- **확인됨:** 세 조건 정상 완료, 공통 입력/설정 일치, 실제 sigma gradient 및 update, 선택된 모델의 작은 비대칭, 이번 seed의 작은 성능 차이.
- **표본에서 확인됨:** 다른 가중치를 고정했을 때 작은 VA 출력 변화, 일부 문장·출력 간 gradient 상쇄.
- **확정되지 않음:** 전체 데이터에서의 민감도와 상쇄 비율, LayerNorm/BF16 각각의 기여, 여러 seed에서의 재현성.
- probe는 각 fold에서 반복 측정한 훈련 문장 2개다. 12회 측정했어도 독립 문장 24개가 아니다. 전체 데이터 통계나 held-out 성능으로 해석하지 않는다.
- 현재 구현은 held-out fold의 epoch별 MSE로 best checkpoint를 선택하고 그 fold의 예측을 OOF에 넣는다. 별도 검증 split을 둔 완전히 독립적인 평가 추정치는 아니다. 세 조건에 같은 절차가 적용됐지만 작은 차이의 일반화 주장을 제한한다.

**다음 확인의 우선순위는, 선택된 체크포인트의 다른 가중치를 고정한 채 전체 held-out 문장을 작은 minibatch로 나눠 raw / 1:1 / 학습된 폭 / 강한 비대칭 폭의 예측을 비교하는 것이다.** 이렇게 해야 두 문장에서 관찰한 낮은 민감도가 전체에도 나타나는지 확인할 수 있다. 이번 로그만으로 sigma LR을 크게 올리거나 특정 층을 수정해야 한다고 결론 내릴 근거는 부족하다.

## 6. 재현 및 코드 근거

실행 예:

```text
python diagnostics/sigma_20260916/analyze_results.py /Users/wansookim/Downloads/sigma_sentence_only_seed42_results_only.zip
```

`analysis.json`에는 비교 설정, 원본 수치, 선택 모델 probe, epoch 궤적, sampled optimizer 통계와 검증 결과를 저장했다. 분석 스크립트는 가중치를 로드하거나 새 학습을 수행하지 않는다.

- `va_model_code/decoder_va/redistribution.py`: log-sigma, FP32 kernel 계산, source별 질량 보존.
- `va_model_code/decoder_va/trainer.py`: sigma 별도 LR/zero-decay optimizer group, clipping 전 gradient 기록 호출.
- `va_model_code/decoder_va/model.py`: raw gaze → redistribution → projector → gaze prefix → decoder → VA head 순서.
- `va_model_code/decoder_va/sigma_diagnostics.py`: step 진단 및 첫 훈련 문장 probe의 정확한 정의.
- `va_model_code/train_model.py`: epoch별 평가, best checkpoint 선택, 최종 OOF 생성.

원본 텍스트/예측 파일을 별도로 추출하지 않았으며, 입력 ZIP과 기록된 SHA-256으로 분석 대상을 고정했다.
