"""
계보(전체 비교: src/MODELS.md): v1 → v2 → v3c(주모델) → v4_ev(EV, 미채택) → v5_greedy·v5_simplify(변수선택→단순화) → v6_holiday(공휴일 구조화)
  이 파일 = v5_greedy (v4_ev의 게이트를 31변수로 일반화한 변수선택 감사 → v5의 근거).
  prev: model_arimax_v4_ev.py   next: model_arimax_v5_simplify.py (감사 결과로 단순화)

탐욕적 변수선택 (Forward-Backward Add/Drop) — 증분신호 검정 일반화.

EV 민감도(`model_arimax_v4_ev.py`)의 4조건 게이트(ΔAIC·p·VIF + ΔMAE 확인)를
**모든 수집 변수**로 일반화한다. 빈 세트에서 시작해 변수를 greedy하게 넣었다
뺐다(forward-backward)하며, 어떤 변수가 증분 신호를 갖는지 재발견한다.
본모델(ARIMAX v3c)·기존 노트북·exog 산출물은 **일절 건드리지 않는** 독립 분석.
경위: docs/research_log/2026-05-30_*_탐욕적_변수선택_AddDrop.md

설계 (본모델과 분리, self-contained 재현):
  - 차수 (1,1,2)(1,1,1,7), log 타깃·편향보정, train-only 표준화
  - 선택 구동 = **AIC + 계수 p + VIF** (학습 내 지표). 2025 MAE는 비게이트
    OOS 확인용으로만 기록(선택 루프가 홀드아웃을 엿보면 누수).
  - 게이트(ADD): ΔAIC < -2  AND  p < 0.05  AND  VIF < 10
  - 게이트(DROP): p > 0.05  OR  VIF > 10  OR  제거 시 AIC 하락
  - window(학습 시작연도) ∈ {2010, 2018, 2021} 바깥 루프. 평가 2025 고정 봉인.

출력 (data/processed/arimax/):
  - greedy_selection_path.csv    평가된 모든 move 1행씩 (감사 가능)
  - greedy_final_summary.csv     window별 discovered vs v3c 비교
  - greedy_window_matrix.csv     변수×window 선택 매트릭스(long form)
"""
from __future__ import annotations

# BLAS 스레드 1 고정 — 멀티프로세싱 오버서브스크립션 방지 (numpy import 전에).
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "arimax"
PATH_CSV = OUT_DIR / "greedy_selection_path.csv"
SUMMARY_CSV = OUT_DIR / "greedy_final_summary.csv"
MATRIX_CSV = OUT_DIR / "greedy_window_matrix.csv"

ORDER = (1, 1, 2)
SORDER = (1, 1, 1, 7)
SPLIT = "2024-12-31"        # 학습 종료(평가 2025 고정 봉인)
TEST_START = "2025"
WINDOWS = [2010, 2018, 2021]

SEARCH_MAXITER = 50        # 검색 중 (AIC 랭킹은 견고)
FINAL_MAXITER = 200       # 최종 discovered/v3c 재적합

ADD_DAIC = -2.0           # ADD 게이트: 이만큼 AIC 개선해야
P_THRESH = 0.05
VIF_THRESH = 10.0

# v3c 확정 외생 14개 (비교 baseline; model_arimax_v3.py와 동일, 여기서 재현)
V3C_EXOG = [
    "cdd_18", "cdd_18_ma7", "hdd_18", "hdd_18_ma7",
    "hm_avg", "ws_avg", "rn_day", "ss_day",
    "is_holiday", "pop_total",
    "heat_th_2day_any", "cold_th_2day_any",
    "cdd_18_sq", "hdd_18_sq",
]

# 후보 변수의 공선 클러스터 (주석/해석용 — 알고리즘엔 영향 없음, VIF가 기계적으로 처리)
CLUSTERS = {
    "cdd": ["cdd_18", "cdd_18_ma3", "cdd_18_ma7", "cdd_18_sq"],
    "hdd": ["hdd_18", "hdd_18_ma3", "hdd_18_ma7", "hdd_18_sq"],
    "temp_level": ["ta_avg", "ta_max", "ta_min", "ta_hourly_max", "ta_hourly_min",
                   "feels_like_max", "feels_like_min"],
    "trend": ["pop_total", "ip_total", "ev_stock", "ev_flow"],
    "heat_dummy": ["heat_feels_th_2day_adv", "heat_feels_th_2day_wrn", "heat_th_2day_any"],
    "cold_dummy": ["cold_th_2day_adv", "cold_th_2day_wrn", "cold_th_2day_any"],
    "weather_misc": ["hm_avg", "ws_avg", "rn_day", "ss_day"],
    "calendar": ["is_weekend", "is_holiday"],
}
VAR_CLUSTER = {v: c for c, vs in CLUSTERS.items() for v in vs}


def load_frame() -> pd.DataFrame:
    """peak_mw + exog + ev 를 일자 인덱스로 병합, v3c 파생변수 부착.
    (model_arimax_v4_ev.load_frame 과 동일 레시피)."""
    y = pd.read_csv(PROCESSED / "peak_mw_2010_2025.csv",
                    encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm")
    X = pd.read_csv(PROCESSED / "exog_daily_2010_2025.csv",
                    encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm")
    ev = pd.read_csv(PROCESSED / "ev_stock_daily_2010_2025.csv",
                     encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm")

    df = X.join(y["peak_mw"]).join(ev).sort_index()
    df.index.freq = "D"

    df["heat_th_2day_any"] = (df["heat_feels_th_2day_adv"] | df["heat_feels_th_2day_wrn"]).astype(int)
    df["cold_th_2day_any"] = (df["cold_th_2day_adv"] | df["cold_th_2day_wrn"]).astype(int)
    df["cdd_18_sq"] = df["cdd_18"] ** 2
    df["hdd_18_sq"] = df["hdd_18"] ** 2
    return df


def make_candidate_pool(df: pd.DataFrame) -> list[str]:
    """31개 후보. 제외: ss_day_n_stations(메타), dow·month(원 범주 정수)."""
    pool = [
        # 기상 raw
        "ta_avg", "ta_max", "ta_min", "hm_avg", "ws_avg", "rn_day", "ss_day",
        # 도일
        "cdd_18", "cdd_18_ma3", "cdd_18_ma7", "hdd_18", "hdd_18_ma3", "hdd_18_ma7",
        # 비선형
        "cdd_18_sq", "hdd_18_sq",
        # 체감/시간
        "feels_like_max", "feels_like_min", "ta_hourly_max", "ta_hourly_min",
        # 극단 더미
        "heat_feels_th_2day_adv", "heat_feels_th_2day_wrn",
        "cold_th_2day_adv", "cold_th_2day_wrn",
        "heat_th_2day_any", "cold_th_2day_any",
        # 달력
        "is_weekend", "is_holiday",
        # 추세
        "pop_total", "ip_total",
        # 전기차
        "ev_stock", "ev_flow",
    ]
    missing = [c for c in pool if c not in df.columns]
    assert not missing, f"후보 누락: {missing}"
    return pool


# ---------------------------------------------------------------------------
# 단일 적합 primitive
# ---------------------------------------------------------------------------
def fit_once(df: pd.DataFrame, exog_names: list[str], start_year: int,
             maxiter: int, want_diag: str | None = None,
             want_diag_all: bool = False) -> dict:
    """단일 (window, exog set) 학습·평가. 순수 dict 반환(피클 가능)."""
    train = df.loc[f"{start_year}-01-01":SPLIT]
    test = df.loc[TEST_START:]
    y_train = np.log(train["peak_mw"])

    out: dict = {
        "exog": tuple(exog_names), "n_exog": len(exog_names),
        "AIC": np.inf, "BIC": np.inf,
        "MAE_2025": np.nan, "RMSE_2025": np.nan, "MAPE_2025_%": np.nan,
        "converged": None, "coef": np.nan, "p": np.nan, "VIF": np.nan,
        "diag": {},
    }

    if exog_names:
        mu = train[exog_names].mean()
        sd = train[exog_names].std().replace(0, 1.0)
        Xtr = (train[exog_names] - mu) / sd
        Xte = (test[exog_names] - mu) / sd
    else:
        Xtr = Xte = None

    try:
        res = SARIMAX(y_train, exog=Xtr, order=ORDER, seasonal_order=SORDER,
                      enforce_stationarity=False, enforce_invertibility=False
                      ).fit(disp=False, maxiter=maxiter)
    except Exception as e:  # noqa: BLE001 — 한 조합 실패가 전체를 죽이지 않게
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    out["AIC"] = float(res.aic)
    out["BIC"] = float(res.bic)
    out["converged"] = bool(res.mle_retvals.get("converged", False)) if hasattr(res, "mle_retvals") else None

    var = res.params.get("sigma2", res.scale)
    fc = res.get_forecast(steps=len(test), exog=Xte)
    pred = np.exp(fc.predicted_mean.values + var / 2)
    actual = test["peak_mw"].values
    err = actual - pred
    out["MAE_2025"] = float(np.abs(err).mean())
    out["RMSE_2025"] = float(np.sqrt((err ** 2).mean()))
    out["MAPE_2025_%"] = float((np.abs(err) / actual).mean() * 100)

    def _vif(names: list[str], col: str) -> float:
        if len(names) < 2:
            return 1.0
        try:
            mat = ((train[names] - train[names].mean()) /
                   train[names].std().replace(0, 1.0)).to_numpy()
            return float(variance_inflation_factor(mat, names.index(col)))
        except Exception:  # noqa: BLE001
            return float("inf")

    if want_diag is not None and want_diag in exog_names:
        out["coef"] = float(res.params.get(want_diag, np.nan))
        out["p"] = float(res.pvalues.get(want_diag, np.nan))
        out["VIF"] = _vif(exog_names, want_diag)

    if want_diag_all:
        out["diag"] = {
            n: {"coef": float(res.params.get(n, np.nan)),
                "p": float(res.pvalues.get(n, np.nan)),
                "VIF": _vif(exog_names, n)}
            for n in exog_names
        }
    return out


# ---------------------------------------------------------------------------
# 멀티프로세싱 (df 는 워커 글로벌에 1회만 적재)
# ---------------------------------------------------------------------------
_WORKER_DF: pd.DataFrame | None = None


def _init_worker(df: pd.DataFrame) -> None:
    global _WORKER_DF
    _WORKER_DF = df


def _fit_worker(args: tuple) -> dict:
    exog_names, start_year, maxiter, want_diag = args
    r = fit_once(_WORKER_DF, list(exog_names), start_year, maxiter, want_diag=want_diag)
    r["_var"] = want_diag
    return r


def parallel_map(df, tasks, pool):
    """tasks: list[(exog_names, start_year, maxiter, want_diag)] → list[dict]."""
    if pool is None:
        results = []
        for exog_names, sy, mi, wd in tasks:
            r = fit_once(df, list(exog_names), sy, mi, want_diag=wd)
            r["_var"] = wd
            results.append(r)
        return results
    return pool.map(_fit_worker, tasks)


# ---------------------------------------------------------------------------
# Forward-Backward 선택 (단일 window)
# ---------------------------------------------------------------------------
def forward_backward_select(df, pool_vars, start_year, mp_pool):
    """빈 세트 → forward(추가) + backward(제거) 반복. (path_rows, selected)."""
    base = fit_once(df, [], start_year, SEARCH_MAXITER)
    current_aic = base["AIC"]
    current_mae = base["MAE_2025"]

    selected: list[str] = []
    path: list[dict] = []
    step = 0
    max_steps = 2 * len(pool_vars)

    def row(phase, action, r, accepted):
        return {
            "window": f"{start_year}~2024", "step": step, "phase": phase,
            "action": action, "var": r.get("_var") or r.get("var"),
            "n_exog": r["n_exog"], "AIC": r["AIC"], "dAIC": r["AIC"] - current_aic,
            "BIC": r["BIC"], "MAE_2025": r["MAE_2025"],
            "dMAE": r["MAE_2025"] - current_mae, "MAPE_2025_%": r["MAPE_2025_%"],
            "coef": r["coef"], "p": r["p"], "VIF": r["VIF"],
            "gate_pass": bool((r["AIC"] - current_aic < ADD_DAIC)
                              and (r["p"] < P_THRESH) and (r["VIF"] < VIF_THRESH)),
            "accepted": accepted,
            "cluster": VAR_CLUSTER.get(r.get("_var") or r.get("var"), ""),
            "selected_after": "",
        }

    while step < max_steps:
        improved = False

        # ---------- FORWARD ----------
        cands = [c for c in pool_vars if c not in selected]
        tasks = [(selected + [c], start_year, SEARCH_MAXITER, c) for c in cands]
        results = parallel_map(df, tasks, mp_pool)
        results.sort(key=lambda r: r["AIC"])  # 안정 정렬 → pool 순서로 동률 처리

        best = None
        for r in results:
            passes = ((r["AIC"] - current_aic < ADD_DAIC)
                      and (r["p"] < P_THRESH) and (r["VIF"] < VIF_THRESH))
            path.append(row("forward", "add", r, accepted=(passes and best is None)))
            if passes and best is None:
                best = r

        if best is not None:
            selected.append(best["_var"])
            current_aic, current_mae = best["AIC"], best["MAE_2025"]
            # 표시: accepted 행에 현재 선택셋 기록
            for prow in reversed(path):
                if prow["step"] == step and prow["action"] == "add" and prow["accepted"]:
                    prow["selected_after"] = ";".join(selected)
                    break
            print(f"  [{start_year}~] step {step}: +{best['_var']} "
                  f"(AIC={best['AIC']:.1f}, dAIC={best['AIC']-base['AIC']:.1f}, "
                  f"p={best['p']:.3f}, VIF={best['VIF']:.1f})  |S|={len(selected)}")
            improved = True
            step += 1

        # ---------- BACKWARD ----------
        if len(selected) >= 2:
            full = fit_once(df, selected, start_year, SEARCH_MAXITER, want_diag_all=True)
            drop_tasks = [( [s for s in selected if s != d], start_year, SEARCH_MAXITER, None)
                          for d in selected]
            drop_results = parallel_map(df, drop_tasks, mp_pool)

            dropped = None
            for d, dres in zip(list(selected), drop_results):
                diag = full["diag"].get(d, {"p": np.nan, "VIF": np.nan})
                aic_without = dres["AIC"]
                stat_bad = (diag["p"] > P_THRESH) or (diag["VIF"] > VIF_THRESH)
                removal_helps = aic_without < current_aic
                do_drop = bool(stat_bad or removal_helps)
                rr = {
                    "window": f"{start_year}~2024", "step": step, "phase": "backward",
                    "action": "drop", "var": d, "n_exog": dres["n_exog"],
                    "AIC": aic_without, "dAIC": aic_without - current_aic,
                    "BIC": dres["BIC"], "MAE_2025": dres["MAE_2025"],
                    "dMAE": dres["MAE_2025"] - current_mae,
                    "MAPE_2025_%": dres["MAPE_2025_%"],
                    "coef": np.nan, "p": diag["p"], "VIF": diag["VIF"],
                    "gate_pass": np.nan, "accepted": do_drop and dropped is None,
                    "cluster": VAR_CLUSTER.get(d, ""), "selected_after": "",
                }
                path.append(rr)
                if do_drop and dropped is None:
                    dropped = d

            if dropped is not None:
                selected.remove(dropped)
                # 제거 후 상태 갱신
                refit = fit_once(df, selected, start_year, SEARCH_MAXITER)
                current_aic, current_mae = refit["AIC"], refit["MAE_2025"]
                for prow in reversed(path):
                    if prow["step"] == step and prow["action"] == "drop" and prow["accepted"]:
                        prow["selected_after"] = ";".join(selected)
                        break
                print(f"  [{start_year}~] step {step}: -{dropped} (제거)  |S|={len(selected)}")
                improved = True
                step += 1

        if not improved:
            break

    return path, selected


# ---------------------------------------------------------------------------
# 최종 비교 (discovered vs v3c, window별, maxiter=200)
# ---------------------------------------------------------------------------
def final_summary_rows(df, start_year, discovered) -> list[dict]:
    sy = start_year
    n_train = len(df.loc[f"{sy}-01-01":SPLIT])
    rows = []
    for name, exog in [("greedy_discovered", discovered), ("v3c_baseline", V3C_EXOG)]:
        r = fit_once(df, exog, sy, FINAL_MAXITER)
        rows.append({
            "window": f"{sy}~2024", "model": name, "n_exog": len(exog),
            "exog_list": ";".join(exog), "AIC": r["AIC"], "BIC": r["BIC"],
            "MAE_2025": r["MAE_2025"], "RMSE_2025": r["RMSE_2025"],
            "MAPE_2025_%": r["MAPE_2025_%"], "n_train": n_train,
        })
    # ΔvsV3c
    v3c = rows[-1]
    for rr in rows:
        rr["dAIC_vs_v3c"] = rr["AIC"] - v3c["AIC"]
        rr["dMAE_vs_v3c"] = rr["MAE_2025"] - v3c["MAE_2025"]
    return rows


def main(smoke: bool = False, serial: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    print(f"데이터: {df.index.min().date()} ~ {df.index.max().date()} ({len(df)}행)")

    if smoke:
        pool_vars = ["cdd_18", "hdd_18", "ip_total", "ev_stock"]
        windows = [2021]
        search_iter = 10
        global SEARCH_MAXITER
        SEARCH_MAXITER = search_iter
    else:
        pool_vars = make_candidate_pool(df)
        windows = WINDOWS
    print(f"후보 {len(pool_vars)}개, windows={windows}, smoke={smoke}, serial={serial}")

    n_proc = 1 if serial else min(cpu_count() - 1, 8)
    mp_pool = None
    if not serial:
        mp_pool = Pool(processes=n_proc, initializer=_init_worker,
                       initargs=(df,), maxtasksperchild=20)

    all_path, all_summary, matrix_rows = [], [], []
    try:
        for sy in windows:
            print(f"\n=== window {sy}~2024 선택 시작 ===")
            path, selected = forward_backward_select(df, pool_vars, sy, mp_pool)
            all_path.extend(path)
            print(f"  → discovered({len(selected)}): {selected}")

            all_summary.extend(final_summary_rows(df, sy, selected))

            # window matrix (선택된 변수의 최종 진단)
            full = fit_once(df, selected, sy, FINAL_MAXITER, want_diag_all=True) if selected else {"diag": {}}
            for v in pool_vars:
                d = full["diag"].get(v, {})
                matrix_rows.append({
                    "var": v, "cluster": VAR_CLUSTER.get(v, ""),
                    "window": f"{sy}~2024", "selected": int(v in selected),
                    "coef": d.get("coef", np.nan), "p": d.get("p", np.nan),
                    "VIF": d.get("VIF", np.nan),
                })
    finally:
        if mp_pool is not None:
            mp_pool.close()
            mp_pool.join()

    pd.DataFrame(all_path).to_csv(PATH_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(all_summary).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(matrix_rows).to_csv(MATRIX_CSV, index=False, encoding="utf-8-sig")

    print("\n=== 최종 요약 (discovered vs v3c) ===")
    s = pd.DataFrame(all_summary)
    cols = ["window", "model", "n_exog", "AIC", "MAE_2025", "MAPE_2025_%",
            "dAIC_vs_v3c", "dMAE_vs_v3c"]
    print(s[cols].round(2).to_string(index=False))
    print(f"\n저장: {PATH_CSV.name}, {SUMMARY_CSV.name}, {MATRIX_CSV.name}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, serial="--serial" in sys.argv)
