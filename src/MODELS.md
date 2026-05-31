# 모델 버전 계보 (v1 → v5)

피크 전력수요(`peak_mw`, 2010–2025) SARIMAX 예측 모델의 변천사 한눈 맵.
**주모델은 v3c.** v4(전기차)는 검정 후 미채택, v5는 검증된 간결 변형(주모델 아님).

학습 2010-01-01~2024-12-31 / 평가 2025 홀드아웃(365일) 봉인은 전 버전 공통.
차수도 v1에서 확정된 `SARIMAX(1,1,2)(1,1,1,7)`로 v2 이후 고정.

```
        [주모델]                        ┌──────── v5 묶음 ────────┐
v1 ─► v2 ─► v3c ──────────────────────► v5_greedy ─► v5_simplify
 그리드 다중공선 log+OR+비선형              변수선택 감사   14→10 단순화
                 └─► v4_ev : 전기차 추가 검정 → 무신호(미채택)
```

## 계보 표

| ver | 스크립트 | 노트북 | 핵심 변경 | one-shot MAE / MAPE | rolling MAE / MAPE | 채택 |
|---|---|---|---|---|---|---|
| **v1** | `model_arimax.py` | `notebooks/modeling.ipynb` | 차수 그리드(3×3) + phase1 16 exog, AIC 최저 선택 | 3,062 / 4.16% | 1,523 / 2.09% | — |
| **v2** | `model_arimax_v2.py` | `notebooks/modeling.ipynb` | 다중공선 정리(`is_weekend`·`ip_total` 제거) → 14 exog | 2,969 / 4.03% | 1,516 / 2.08% | — |
| **v3c** ★ | `model_arimax_v3.py` | `notebooks/modeling.ipynb` | a:log → b:OR더미 통합 → c:비선형 기온(cdd²·hdd²) | **2,668 / 3.63%** | **1,357 / 1.87%** | **주모델** |
| v4_ev | `model_arimax_v4_ev.py` | `notebooks/v4_ev.ipynb` | v3c + 전기차 stock/flow ×3 window 민감도 | 무신호(p≫0.05) | — | 미채택 |
| v5_greedy | `model_arimax_v5_greedy.py` | `notebooks/v5_greedy.ipynb` | v4_ev 게이트를 31변수로 일반화(forward-backward) | (감사 분석) | — | 분석 |
| v5 | `model_arimax_v5_simplify.py` | `notebooks/v5_simplify.ipynb` | parsimony 14→10 (BIC 후방제거) | 2,670 / 3.64% | 1,354 / 1.87% | 검증된 간결변형 |

> **AIC/BIC 비교 주의.** v1·v2는 원 스케일(non-log) 적합이라 AIC가 log 모델과
> 직접 비교 불가. log 스케일 AIC: **v3c −23,830.96 / BIC −23,698.85**,
> **v5 AIC −23,836.49 / BIC −23,730.80** (v5가 ΔAIC −5.5·ΔBIC −32.0로 더 낮음).

## 버전별 요약

- **v1 — 베이스라인 그리드.** 차수 3종 × 외생 3서브셋 = 9조합을 AIC로 비교,
  `(1,1,2)(1,1,1,7)` + phase1 16 exog 선택. Naive 대비 MAE −29%.
  로그: `docs/research_log/2026-05-26_1830_ARIMAX_모델링_2025홀드아웃.md`.
- **v2 — 다중공선 정리.** `is_weekend`(s=7이 흡수), `ip_total`(pop와 공선·부호반전) 제거,
  냉/난방 도일 중복 정리 → **14 exog 확정**. 점예측 v1과 유사하나 계수 해석 안정화.
  로그: `docs/research_log/2026-05-26_1734_ARIMAX_v2_다중공선성_정리.md`.
- **v3c — 주모델.** 3단계: (a) `log(peak_mw)` 변환[편향보정 exp(μ+σ²/2)] →
  MAE −7.9%; (b) 희소 더미 OR 통합(`heat/cold_th_2day_any`) → 단순화; (c) 비선형
  기온항 `cdd_18²`·`hdd_18²` → MAE −2.6%. **v1 대비 one-shot MAE −12.9%.**
  로그: `docs/research_log/2026-05-26_1815_ARIMAX_v3_log_OR_비선형.md`.
- **v4_ev — 전기차 변형(미채택).** v3c에 EV 등록대수(stock=누적, flow=월증분)를 추가해
  3 window(2010/2018/2021~)×3 variant로 증분신호 검정. **6/6 무신호**(EV 계수 p≫0.05,
  추세는 SARIMA 적분항이 이미 흡수) → 본모델 미반영.
  로그: `2026-05-30_1100_*`, `2026-05-30_1600_*`.
- **v5_greedy — 변수선택 감사.** v4_ev의 4조건 게이트(ΔAIC·p·VIF + ΔMAE)를 수집 31변수
  전체로 일반화(forward-backward, window 바깥루프). 전 window 공통 견고핵심 = 휴일·태양복사·
  난방(선형+비선형)·냉방도일·습도. `ws_avg`·`pop_total`·`heat/cold_th_2day_any`는 **어느
  window에서도 미선택** → v5의 근거. 로그: `docs/research_log/2026-05-30_1645_탐욕적_변수선택_AddDrop.md`.
- **v5_simplify — 간결 변형.** greedy 미선택 4개를 v3c에서 BIC 후방제거 → **14→10 exog**.
  AIC −5.5·BIC −32.0로 오히려 개선, rolling MAE도 −2.8(1,357→1,354). one-shot MAE +0.11%·
  Winkler +0.6%만 미세 비용 → **주모델은 v3c 유지, v5는 "14→10 동등" robustness 근거.**
  로그: `docs/research_log/2026-05-30_1730_v5_단순화_검증.md`.

## 산출물 매핑

| 종류 | 경로 |
|---|---|
| v1 결과 | `data/processed/arimax/grid_results.csv`, `metrics.csv`, `forecast.csv` |
| v2 결과 | `data/processed/arimax/v2_grid.csv`, `v2_forecast.csv` |
| v3c 결과 | `data/processed/arimax/v3_results.csv`, `v3c_forecast.csv` |
| v4_ev 결과 | `data/processed/arimax/ev_sensitivity_results.csv` (+ `build_ev_stock.py`) |
| v5_greedy 결과 | `data/processed/arimax/greedy_{selection_path,final_summary,window_matrix}.csv` |
| v5 결과 | `data/processed/arimax/v5_{ablation,results}.csv`, `v5c_forecast.csv` |
| 버전 합본 | `data/processed/arimax/model_lineage.csv` (← `notebooks/model_lineage.ipynb` 생성) |

> 데이터 산출 CSV 파일명은 버전 rename(2026-05-31) 후에도 **유지**(스크립트 내부
> 경로 호환). 버전 ↔ 파일 매핑은 본 문서가 단일 진실원천(SSOT).

## 전체 비교·계보 시각화

- `notebooks/model_lineage.ipynb` — v1~v5 MAE/MAPE 추이·예측 오버레이·변수수·잔차 개선.
- 정리 로그: `docs/research_log/2026-05-31_1000_모델_변천사_v1_v5_정리.md`.
