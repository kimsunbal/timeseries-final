"""
전기차 등록대수 — 서브기간 민감도 분석 (Phase 2, 본모델과 완전 분리).

목적: 본모델(ARIMAX v3c)을 **건드리지 않고**, 전기차 등록대수(stock/flow)를
외생변수로 추가했을 때 증분 신호가 있는지 robustness 차원에서만 검정한다.
본모델 외생변수 셋·코드·노트북은 변경하지 않는다.
경위: docs/research_log/2026-05-30_1100_신산업_전기차_외생변수_검토.md

설계 (v3c와 동일하게 self-contained 재현):
  - 차수: (1,1,2)(1,1,1,7), log 타깃, 편향보정 exp(μ+σ²/2)
  - 외생변수 표준화: 각 window 학습구간에서만
  - 평가: 2025 홀드아웃 one-shot (365일)
  - 본모델 외생 14개(v3c) = baseline. 여기에 EV 변수 추가.

비교 격자 (3 window × 3 variant):
  window  ∈ {2010~, 2018~, 2021~}  (학습 시작연도, 종료 2024 / 평가 2025 고정)
  variant ∈ {baseline(EV 없음), +ev_stock(level), +ev_flow(증분)}

보고: AIC/BIC, MAE/MAPE_2025, EV 계수·p값, EV의 VIF(공선성),
      ΔAIC·ΔMAE (baseline 대비). 결론은 본모델 미반영 robustness로만.

출력: data/processed/arimax/ev_sensitivity_results.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT_CSV = PROCESSED / "arimax" / "ev_sensitivity_results.csv"

ORDER = (1, 1, 2)
SORDER = (1, 1, 1, 7)
SPLIT = "2024-12-31"          # 학습 종료(평가 2025 고정)
WINDOWS = [2010, 2018, 2021]  # 학습 시작연도
EV_VARIANTS = {
    "baseline": None,
    "+ev_stock": "ev_stock",
    "+ev_flow": "ev_flow",
}

# v3c 확정 외생변수 14개 (model_arimax_v3.py와 동일, 여기서 재현)
V3C_EXOG = [
    "cdd_18", "cdd_18_ma7", "hdd_18", "hdd_18_ma7",
    "hm_avg", "ws_avg", "rn_day", "ss_day",
    "is_holiday", "pop_total",
    "heat_th_2day_any", "cold_th_2day_any",
    "cdd_18_sq", "hdd_18_sq",
]


def load_frame() -> pd.DataFrame:
    """peak_mw + exog + ev 를 일자 인덱스로 병합, v3c 파생변수 부착."""
    y = pd.read_csv(PROCESSED / "peak_mw_2010_2025.csv",
                    encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm")
    X = pd.read_csv(PROCESSED / "exog_daily_2010_2025.csv",
                    encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm")
    ev = pd.read_csv(PROCESSED / "ev_stock_daily_2010_2025.csv",
                     encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm")

    df = X.join(y["peak_mw"]).join(ev).sort_index()
    df.index.freq = "D"

    # v3c 파생: OR 통합 더미 + 비선형 기온항
    df["heat_th_2day_any"] = (df["heat_feels_th_2day_adv"] | df["heat_feels_th_2day_wrn"]).astype(int)
    df["cold_th_2day_any"] = (df["cold_th_2day_adv"] | df["cold_th_2day_wrn"]).astype(int)
    df["cdd_18_sq"] = df["cdd_18"] ** 2
    df["hdd_18_sq"] = df["hdd_18"] ** 2
    return df


def fit_eval(df: pd.DataFrame, start_year: int, ev_col: str | None) -> dict:
    """단일 (window, variant) 학습·평가. log 타깃 + 편향보정."""
    exog_names = V3C_EXOG + ([ev_col] if ev_col else [])

    train = df.loc[f"{start_year}-01-01":SPLIT]
    test = df.loc["2025":]

    mu = train[exog_names].mean()
    sd = train[exog_names].std().replace(0, 1.0)
    Xtr = (train[exog_names] - mu) / sd
    Xte = (test[exog_names] - mu) / sd

    y_train = np.log(train["peak_mw"])

    res = SARIMAX(
        y_train, exog=Xtr, order=ORDER, seasonal_order=SORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=200)

    var = res.params.get("sigma2", res.scale)
    fc = res.get_forecast(steps=len(test), exog=Xte)
    pred = np.exp(fc.predicted_mean.values + var / 2)
    actual = test["peak_mw"].values
    err = actual - pred

    row = {
        "window": f"{start_year}~2024",
        "variant": "baseline" if ev_col is None else f"+{ev_col}",
        "n_train": len(train),
        "n_exog": len(exog_names),
        "AIC": res.aic,
        "BIC": res.bic,
        "MAE_2025": np.abs(err).mean(),
        "MAPE_2025_%": (np.abs(err) / actual).mean() * 100,
        "ev_coef": np.nan,
        "ev_p": np.nan,
        "ev_VIF": np.nan,
    }
    if ev_col is not None:
        row["ev_coef"] = float(res.params.get(ev_col, np.nan))
        row["ev_p"] = float(res.pvalues.get(ev_col, np.nan))
        # EV 변수의 VIF (표준화 학습 외생 행렬 기준)
        vif_X = Xtr.to_numpy()
        j = exog_names.index(ev_col)
        row["ev_VIF"] = float(variance_inflation_factor(vif_X, j))
    return row


def main() -> None:
    df = load_frame()
    print(f"데이터: {df.index.min().date()} ~ {df.index.max().date()} ({len(df)}행)")

    rows = []
    for start in WINDOWS:
        for _, ev_col in EV_VARIANTS.items():
            tag = "baseline" if ev_col is None else f"+{ev_col}"
            print(f"  fit window={start}~ variant={tag} ...")
            rows.append(fit_eval(df, start, ev_col))

    res = pd.DataFrame(rows)

    # baseline 대비 ΔAIC·ΔMAE
    base = res[res["variant"] == "baseline"].set_index("window")
    res["dAIC_vs_base"] = res.apply(
        lambda r: r["AIC"] - base.loc[r["window"], "AIC"], axis=1)
    res["dMAE_vs_base"] = res.apply(
        lambda r: r["MAE_2025"] - base.loc[r["window"], "MAE_2025"], axis=1)

    PROCESSED.joinpath("arimax").mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    show = ["window", "variant", "n_train", "AIC", "MAE_2025", "MAPE_2025_%",
            "ev_coef", "ev_p", "ev_VIF", "dAIC_vs_base", "dMAE_vs_base"]
    print("\n=== EV 민감도 비교 (본모델 미반영, robustness 전용) ===")
    print(res[show].round(3).to_string(index=False))
    print(f"\n저장: {OUT_CSV}")
    print("\n해석 기준: ΔAIC<0 & ΔMAE<0 & EV p<0.05 & VIF<10 이어야 증분 신호 인정.")


if __name__ == "__main__":
    main()
