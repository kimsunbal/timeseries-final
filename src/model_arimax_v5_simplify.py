"""
계보(전체 비교: src/MODELS.md): v1 → v2 → v3c(주모델) → v4_ev(EV, 미채택) → v5_greedy·v5_simplify(변수선택→단순화) → v6_holiday(공휴일 구조화)
  이 파일 = v5_simplify (parsimony 14→10, BIC 후방제거 — 검증된 간결변형, 주모델 아님).
  prev: model_arimax_v5_greedy.py   next: —

ARIMAX v5 — v3c 단순화(parsimony) 검증. **본모델 v3c 대체 아님, 별도 변형.**

greedy 변수선택(`model_arimax_v5_greedy.py`, 로그 2026-05-30_1645)이 v3c 변수 중
`ws_avg`·`pop_total`·`cold_th_2day_any`·`heat_th_2day_any` 등을 **어느 window에서도
미선택**으로 식별. 이 4개는 **AIC·BIC 모두 제거 시 개선**되는 잉여 변수다
(BIC 페널티 ln(5479)≈8.61 > AIC 2 라 BIC가 더 강하게 떨군다). 후방제거로 식별·검증한다.

설계 (v3c와 동일, self-contained 재현; 기존 모듈 import 없음):
  - 차수 (1,1,2)(1,1,1,7), log 타깃·편향보정, train-only 표준화, maxiter=200 일관.
  - 단일 window: 학습 2010-2024 / 평가 2025 홀드아웃 **봉인**.
  - 선택 구동 = **학습내 BIC**. 2025 MAE(one-shot·rolling)·PI는 확인용(비게이트).
  - EV 변수 미사용(v5 ⊂ v3c 피처).

v5 정의 (둘 다):
  - v5_bic_backward : v3c 14개에서 BIC 후방제거로 수렴한 세트 (헤드라인)
  - v5_drop4        : v3c − {ws_avg, pop_total, cold_th_2day_any, heat_th_2day_any}

출력 (data/processed/arimax/):
  - v5_ablation.csv   LOO + BIC-backward 경로 + AIC-backward demo
  - v5_results.csv    v3c vs v5_bic_backward vs v5_drop4 (전체 지표 + rolling)
  - v5c_forecast.csv  최종 v5 예측(actual/pred/ci/rolling)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "arimax"

ORDER = (1, 1, 2)
SORDER = (1, 1, 1, 7)
SPLIT = "2024-12-31"
TEST_START = "2025"
MAXITER = 200

V3C_EXOG = [
    "cdd_18", "cdd_18_ma7", "hdd_18", "hdd_18_ma7",
    "hm_avg", "ws_avg", "rn_day", "ss_day",
    "is_holiday", "pop_total",
    "heat_th_2day_any", "cold_th_2day_any",
    "cdd_18_sq", "hdd_18_sq",
]
DROP4 = ["ws_avg", "pop_total", "cold_th_2day_any", "heat_th_2day_any"]

# smoke 기준값 (data/processed/arimax/v3_results.csv, "v3c log+OR+nonlinear")
V3C_REF = {
    "AIC": (-23830.96, 0.5), "BIC": (-23698.85, 0.5),
    "MAE_2025": (2667.51, 3.0), "MAPE_2025_%": (3.6297, 0.02),
    "PI95_cov_%": (95.342, 0.4), "PINAW": (0.29766, 0.006),
    "Winkler": (18304.86, 60.0), "resid_skew": (-1.7547, 0.03),
    "resid_kurt": (13.8997, 0.08),
}


def load_frame() -> pd.DataFrame:
    """peak_mw + exog + v3c 파생변수. (EV 미사용)."""
    y = pd.read_csv(PROCESSED / "peak_mw_2010_2025.csv",
                    encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm")
    X = pd.read_csv(PROCESSED / "exog_daily_2010_2025.csv",
                    encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm")
    df = X.join(y["peak_mw"]).sort_index()
    df.index.freq = "D"
    df["heat_th_2day_any"] = (df["heat_feels_th_2day_adv"] | df["heat_feels_th_2day_wrn"]).astype(int)
    df["cold_th_2day_any"] = (df["cold_th_2day_adv"] | df["cold_th_2day_wrn"]).astype(int)
    df["cdd_18_sq"] = df["cdd_18"] ** 2
    df["hdd_18_sq"] = df["hdd_18"] ** 2
    return df


def _std(train, test, exog):
    mu = train[exog].mean()
    sd = train[exog].std().replace(0, 1.0)
    return (train[exog] - mu) / sd, (test[exog] - mu) / sd


def pi_metrics(y_true, ci_low, ci_high, alpha=0.05, eta=50.0) -> dict:
    """PICP/PINAW/CWC/Winkler (model_arimax.pi_metrics 재현)."""
    y = np.asarray(y_true, float); lo = np.asarray(ci_low, float); hi = np.asarray(ci_high, float)
    inside = (y >= lo) & (y <= hi)
    picp = inside.mean()
    width = hi - lo
    yr = y.max() - y.min()
    pinaw = width.mean() / yr if yr > 0 else np.nan
    mu = 1.0 - alpha
    gamma = 0.0 if picp >= mu else 1.0
    cwc = pinaw * (1.0 + gamma * np.exp(-eta * (picp - mu)))
    below = np.maximum(0.0, lo - y); above = np.maximum(0.0, y - hi)
    winkler = (width + (2.0 / alpha) * (below + above)).mean()
    return {"PI95_cov_%": picp * 100, "PINAW": pinaw, "CWC": cwc, "Winkler": winkler}


def fit_eval(df, exog, label, maxiter=MAXITER) -> dict:
    """단일 exog 세트 학습·평가 (v3c fit_eval 재현 + 전체 지표)."""
    train = df.loc["2010-01-01":SPLIT]
    test = df.loc[TEST_START:]
    Xtr, Xte = _std(train, test, exog) if exog else (None, None)
    y_train = np.log(train["peak_mw"])

    res = SARIMAX(y_train, exog=Xtr, order=ORDER, seasonal_order=SORDER,
                  enforce_stationarity=False, enforce_invertibility=False
                  ).fit(disp=False, maxiter=maxiter)

    var = res.params.get("sigma2", res.scale)
    fc = res.get_forecast(steps=len(test), exog=Xte)
    pred = np.exp(fc.predicted_mean.values + var / 2)
    ci = fc.conf_int(alpha=0.05)
    ci_low = np.exp(ci.iloc[:, 0].values + var / 2)
    ci_high = np.exp(ci.iloc[:, 1].values + var / 2)

    actual = test["peak_mw"].values
    err = actual - pred
    pim = pi_metrics(actual, ci_low, ci_high, alpha=0.05)

    resid = res.resid.iloc[60:]
    lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)

    coef_diag = {n: {"coef": float(res.params.get(n, np.nan)),
                     "p": float(res.pvalues.get(n, np.nan))} for n in exog}

    return {
        "model": label, "log_target": True, "n_exog": len(exog),
        "AIC": float(res.aic), "BIC": float(res.bic),
        "MAE_2025": float(np.abs(err).mean()),
        "RMSE_2025": float(np.sqrt((err ** 2).mean())),
        "MAPE_2025_%": float((np.abs(err) / actual).mean() * 100),
        "bias_2025": float(err.mean()),
        "PI95_cov_%": pim["PI95_cov_%"], "PINAW": pim["PINAW"],
        "CWC": pim["CWC"], "Winkler": pim["Winkler"],
        "resid_skew": float(stats.skew(resid)), "resid_kurt": float(stats.kurtosis(resid)),
        "LB_p_lag10": float(lb.loc[10, "lb_pvalue"]), "LB_p_lag20": float(lb.loc[20, "lb_pvalue"]),
        "exog_list": ";".join(exog),
        "_res": res, "_pred": pred, "_ci_low": ci_low, "_ci_high": ci_high,
        "_test_index": test.index, "_actual": actual, "coef_diag": coef_diag,
    }


def rolling_one_step_log(df, exog, res) -> np.ndarray:
    """rolling 1-step (v3c rolling_one_step_log 재현)."""
    train = df.loc["2010-01-01":SPLIT]
    test = df.loc[TEST_START:]
    _, Xte = _std(train, test, exog) if exog else (None, None)
    var = res.params.get("sigma2", res.scale)
    preds = []
    for i in range(len(test)):
        ex = Xte.iloc[[i]] if Xte is not None else None
        fc = res.get_forecast(steps=1, exog=ex)
        preds.append(np.exp(float(fc.predicted_mean.iloc[0]) + var / 2))
        res = res.append(np.log(test["peak_mw"].iloc[[i]]), exog=ex, refit=False)
    return np.asarray(preds)


# ---------------------------------------------------------------------------
def smoke_check(df) -> None:
    print("=== smoke: v3c 재현 대조 (v3_results.csv) ===")
    r = fit_eval(df, V3C_EXOG, "v3c", MAXITER)
    ok = True
    for k, (exp, tol) in V3C_REF.items():
        got = r[k]
        hit = abs(got - exp) <= tol
        ok &= hit
        print(f"  {k:14s} got={got:>12.4f} exp={exp:>12.4f} tol=±{tol:<6} {'✓' if hit else '✗ MISMATCH'}")
    roll = rolling_one_step_log(df, V3C_EXOG, r["_res"])
    rmae = float(np.abs(r["_actual"] - roll).mean())
    # 기준값: v3c_forecast.csv rolling 열의 실제 MAE (v3 모델이 산출·저장한 값)
    hit = abs(rmae - 1357.3) <= 5.0
    ok &= hit
    print(f"  {'rolling_MAE':14s} got={rmae:>12.1f} exp={1357.3:>12.1f} tol=±5     {'✓' if hit else '✗'}")
    assert ok, "v3c 재현 불일치 — 레시피 드리프트(표준화/편향보정/60일 trim/exog 순서 확인)"
    print("  → PASS: self-contained 레시피가 v3c와 일치\n")


def leave_one_out(df) -> tuple[list[dict], dict]:
    full = fit_eval(df, V3C_EXOG, "v3c_full", MAXITER)
    rows = []
    for v in V3C_EXOG:
        sub = [x for x in V3C_EXOG if x != v]
        r = fit_eval(df, sub, f"drop_{v}", MAXITER)
        rows.append({
            "phase": "LOO", "step": np.nan, "var": v, "action": "leave_one_out",
            "n_exog": r["n_exog"], "AIC": r["AIC"], "BIC": r["BIC"],
            "dAIC": r["AIC"] - full["AIC"], "dBIC": r["BIC"] - full["BIC"],
            "MAE_2025": r["MAE_2025"], "dMAE": r["MAE_2025"] - full["MAE_2025"],
            "MAPE_2025_%": r["MAPE_2025_%"],
            "p_in_full": full["coef_diag"][v]["p"], "accepted": np.nan,
            "selected_after": "",
        })
    return rows, full


def _backward(df, criterion: str) -> tuple[list[dict], list[str]]:
    """criterion ∈ {'BIC','AIC'}. 후방제거: 해당 기준 가장 낮추는 제거 채택, 개선 없으면 종료."""
    current = list(V3C_EXOG)
    cur = fit_eval(df, current, "start", MAXITER)
    cur_val = cur[criterion]
    rows, step = [], 0
    while len(current) >= 1:
        trials = []
        full_step = fit_eval(df, current, "full_step", MAXITER)  # 현 세트 p값(제거 전)
        for v in current:
            sub = [x for x in current if x != v]
            r = fit_eval(df, sub, f"drop_{v}", MAXITER)
            trials.append((v, r))
            rows.append({
                "phase": f"{criterion}_backward", "step": step, "var": v, "action": "drop",
                "n_exog": r["n_exog"], "AIC": r["AIC"], "BIC": r["BIC"],
                "dAIC": r["AIC"] - cur["AIC"], "dBIC": r["BIC"] - cur["BIC"],
                "MAE_2025": r["MAE_2025"], "dMAE": r["MAE_2025"] - cur["MAE_2025"],
                "MAPE_2025_%": r["MAPE_2025_%"],
                "p_in_full": full_step["coef_diag"][v]["p"], "accepted": False,
                "selected_after": "",
            })
        v_best, r_best = min(trials, key=lambda t: t[1][criterion])
        if r_best[criterion] < cur_val:
            current.remove(v_best)
            cur, cur_val = r_best, r_best[criterion]
            for pr in reversed(rows):
                if pr["step"] == step and pr["var"] == v_best and pr["action"] == "drop":
                    pr["accepted"] = True
                    pr["selected_after"] = ";".join(current)
                    break
            print(f"  [{criterion}] step {step}: -{v_best} ({criterion}={cur_val:.2f})  |S|={len(current)}")
            step += 1
        else:
            print(f"  [{criterion}] 수렴: 더 낮추는 제거 없음  |S|={len(current)}")
            break
    return rows, current


def head_to_head(df, v5_exog) -> tuple[pd.DataFrame, dict]:
    sets = {"v3c": V3C_EXOG, "v5_bic_backward": v5_exog,
            "v5_drop4": [x for x in V3C_EXOG if x not in DROP4]}
    fits = {name: fit_eval(df, ex, name, MAXITER) for name, ex in sets.items()}

    # rolling: v3c + v5_bic_backward
    roll_mae, roll_mape = {}, {}
    for name in ["v3c", "v5_bic_backward"]:
        roll = rolling_one_step_log(df, sets[name], fits[name]["_res"])
        act = fits[name]["_actual"]
        roll_mae[name] = float(np.abs(act - roll).mean())
        roll_mape[name] = float((np.abs(act - roll) / act).mean() * 100)
        if name == "v5_bic_backward":
            fits[name]["_rolling"] = roll

    base = fits["v3c"]
    rows = []
    keep = ["model", "log_target", "n_exog", "AIC", "BIC", "MAE_2025", "RMSE_2025",
            "MAPE_2025_%", "bias_2025", "PI95_cov_%", "PINAW", "CWC", "Winkler",
            "resid_skew", "resid_kurt", "LB_p_lag10", "LB_p_lag20", "exog_list"]
    for name, f in fits.items():
        row = {k: f[k] for k in keep}
        row["MAE_2025_rolling"] = roll_mae.get(name, np.nan)
        row["MAPE_2025_rolling"] = roll_mape.get(name, np.nan)
        row["dAIC_vs_v3c"] = f["AIC"] - base["AIC"]
        row["dBIC_vs_v3c"] = f["BIC"] - base["BIC"]
        row["dMAE_vs_v3c"] = f["MAE_2025"] - base["MAE_2025"]
        rows.append(row)
    return pd.DataFrame(rows), fits


def adoption_decision(res_df) -> dict:
    v3 = res_df[res_df.model == "v3c"].iloc[0]
    v5 = res_df[res_df.model == "v5_bic_backward"].iloc[0]
    checks = {
        "n_exog<14": v5.n_exog < 14,
        "BIC<=v3c": v5.BIC <= v3.BIC + 1e-6,
        "one-shot MAE<=1.02x": v5.MAE_2025 <= 1.02 * v3.MAE_2025,
        "rolling MAE<=1.02x": v5.MAE_2025_rolling <= 1.02 * v3.MAE_2025_rolling,
        "PICP>=94": v5["PI95_cov_%"] >= 94.0,
        "Winkler<=v3c": v5.Winkler <= v3.Winkler,
    }
    return {"accept": all(checks.values()), "checks": checks}


def main(smoke=False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    print(f"데이터: {df.index.min().date()} ~ {df.index.max().date()} ({len(df)}행)\n")

    smoke_check(df)
    if smoke:
        print("--smoke: 재현 확인만 하고 종료.")
        return

    print("=== Leave-one-out 진단 ===")
    loo_rows, _ = leave_one_out(df)

    print("\n=== BIC 후방제거 ===")
    bic_rows, v5_exog = _backward(df, "BIC")
    print(f"  → v5_bic_backward({len(v5_exog)}): {v5_exog}")
    dropped = [v for v in V3C_EXOG if v not in v5_exog]
    print(f"  → dropped: {dropped}")

    print("\n=== AIC 후방제거 (교차확인) ===")
    aic_rows, aic_set = _backward(df, "AIC")
    print(f"  → AIC-backward 잔존({len(aic_set)})")

    pd.DataFrame(loo_rows + bic_rows + aic_rows).to_csv(
        OUT_DIR / "v5_ablation.csv", index=False, encoding="utf-8-sig")

    print("\n=== head-to-head (v3c vs v5) ===")
    res_df, fits = head_to_head(df, v5_exog)
    res_df.to_csv(OUT_DIR / "v5_results.csv", index=False, encoding="utf-8-sig")
    show = ["model", "n_exog", "AIC", "BIC", "MAE_2025", "MAE_2025_rolling",
            "MAPE_2025_%", "PI95_cov_%", "Winkler", "dBIC_vs_v3c", "dMAE_vs_v3c"]
    print(res_df[show].round(2).to_string(index=False))

    # 최종 v5 forecast 저장
    f5 = fits["v5_bic_backward"]
    pd.DataFrame({
        "actual": f5["_actual"], "pred": f5["_pred"],
        "ci_low": f5["_ci_low"], "ci_high": f5["_ci_high"],
        "rolling": f5.get("_rolling", np.full(len(f5["_actual"]), np.nan)),
    }, index=f5["_test_index"]).rename_axis("tm").to_csv(
        OUT_DIR / "v5c_forecast.csv", encoding="utf-8-sig")

    verdict = adoption_decision(res_df)
    print("\n=== 채택 판정 (v5_bic_backward) ===")
    for k, v in verdict["checks"].items():
        print(f"  {'✓' if v else '✗'} {k}")
    print(f"  → {'채택: v5는 검증된 간결 변형' if verdict['accept'] else '미채택: 더 간결하나 OOS 비용 — v3c 유지'}")
    print(f"  (greedy 미선택 4개 = {DROP4})")
    print(f"\n저장: v5_ablation.csv, v5_results.csv, v5c_forecast.csv")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
