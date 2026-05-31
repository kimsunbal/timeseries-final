"""
계보(전체 비교: src/MODELS.md): v1 → v2 → v3c(주모델) → v4_ev(EV, 미채택) → v5_greedy·v5_simplify(변수선택→단순화) → v6_holiday(공휴일 구조화)
  이 파일 = v6 (v3c의 단일 is_holiday를 공휴일 유형별 더미로 분해 → 잔차 개선).
  prev: model_arimax_v3.py (주모델 v3c)   next: —

ARIMAX v6 — 공휴일 구조화로 잔차(skew/kurtosis/자기상관) 개선.

v3c (주모델, 2026-05-26_1815 로그) 잔차 진단 (2026-05-31):
  - skew -1.75, kurtosis 13.9 (정규성 깨짐), Ljung-Box p<0.001 (자기상관 잔존)
  - 3σ 초과 outlier 123일 중 67%가 공휴일(전체의 4.6%)
  - 비공휴일 잔차는 kurt 5.1로 거의 정규 → 꼬리는 사실상 공휴일이 만든다
  - 공휴일은 is_holiday 더미가 있는데도 평균 -0.016(과대예측) — 단일 더미가
    "평균 공휴일 효과"만 잡아 설날·추석 같은 다일 음력 연휴의 더 큰 붕괴를 못 흡수

v6 개선: 단일 `is_holiday`(1개)를 공휴일 유형별 더미로 분해(연휴 구조화).
  - hol_lunar   : 설날·추석(+인접 대체/임시공휴일) — 다일 음력 연휴(큰 붕괴)
  - hol_single  : 그 외 단일 공휴일(현충일·광복절·신정 등) = is_holiday − lunar
                  (hol_lunar + hol_single = 기존 is_holiday, 정확 분할)
  - lunar_eve   : 음력 연휴 블록 직전일(명절 전날, 수요 선반영 급감)  ← 신규 비공휴일
  - lunar_after : 음력 연휴 블록 직후일(복귀 중)                      ← 신규 비공휴일
  - is_bridge   : 샌드위치 낀 평일(양옆이 공휴일·주말인 평일, 휴가 몰림)  ← 신규 비공휴일

단계별 변형 (v3 방식):
  - v6a: v3c에서 is_holiday → hol_lunar + hol_single (분할만)
  - v6b: v6a + lunar_eve + lunar_after (연휴 내 위치)
  - v6c: v6b + is_bridge (낀 평일) — 중간 범위 최종

설계 고정 (v3c와 동일, 공휴일 항만 교체):
  - 차수 (1,1,2)(1,1,1,7), log 타깃 + 편향보정 exp(μ+σ²/2)
  - 학습 2010-2024 / 2025 홀드아웃, train-only 표준화
  - 나머지 13개 외생변수(기상·달력·추세)는 v3c 그대로

산출: data/processed/arimax/v6_results.csv (v3c + 3단계 비교)
      data/processed/arimax/v6c_forecast.csv (최종 v6c 예측)

검증: `python src/model_arimax_v6_holiday.py --smoke` → v3c 재현(공휴일 항 교체 전) 확인.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import model_arimax_v3 as v3
from model_arimax import SplitData

ORDER = (1, 1, 2)
SORDER = (1, 1, 1, 7)
OUT_DIR = ROOT / "data" / "processed" / "arimax"
HOLIDAYS_CSV = ROOT / "data" / "raw" / "holidays_2010_2025.csv"

# v6a~c: 공휴일 유형 더미 (5개)
HOL_COLS = ["hol_lunar", "hol_single", "lunar_eve", "lunar_after", "is_bridge"]
# v6d: 간결 인코딩 (2개) — 연휴 지속성(연속 비근무일수, 주말포함) + 명절여부
RUN_COLS = ["holiday_run", "is_myeongjeol"]
# v6f/g: 명절효과 약화 — 평탄(길이無) / 포화(log run)
SOFT_COLS = ["holiday_run_log"]


def build_holiday_features(full_index: pd.DatetimeIndex) -> pd.DataFrame:
    """공휴일 원천(이름 포함)에서 유형별 더미 파생.

    - hol_lunar  : name∈{설날,추석} ∪ {설날/추석 ±2일 내 대체·임시·휴무}
    - hol_single : 그 외 is_holiday=Y (hol_lunar + hol_single = is_holiday)
    - lunar_eve / lunar_after : 음력 연휴 연속블록의 직전/직후일
    - is_bridge  : 양옆이 비근무(공휴일 or 주말)인 평일(자신은 비공휴일)
    """
    h = pd.read_csv(HOLIDAYS_CSV, encoding="utf-8-sig", parse_dates=["date"])
    h = h[h["is_holiday"] == "Y"].copy()

    lunar_core = pd.DatetimeIndex(sorted(h.loc[h["name"].isin(["설날", "추석"]), "date"]))
    hol_dates = set(h["date"])

    def near_lunar(d: pd.Timestamp) -> bool:
        return lunar_core.size > 0 and int(np.abs((lunar_core - d).days).min()) <= 2

    lunar_dates: set[pd.Timestamp] = set()
    for _, r in h.iterrows():
        nm, d = r["name"], r["date"]
        if nm in ("설날", "추석"):
            lunar_dates.add(d)
        elif ("대체" in nm or "임시" in nm or "휴무" in nm) and near_lunar(d):
            lunar_dates.add(d)
    single_dates = hol_dates - lunar_dates

    # 음력 연휴 연속블록 → eve/after
    ld = sorted(lunar_dates)
    eve: set[pd.Timestamp] = set()
    after: set[pd.Timestamp] = set()
    i = 0
    while i < len(ld):
        j = i
        while j + 1 < len(ld) and (ld[j + 1] - ld[j]).days == 1:
            j += 1
        eve.add(ld[i] - pd.Timedelta(days=1))
        after.add(ld[j] + pd.Timedelta(days=1))
        i = j + 1

    def nonwork(d: pd.Timestamp) -> bool:
        return (d in hol_dates) or (d.weekday() >= 5)

    bridge = {
        d for d in full_index
        if d.weekday() < 5 and d not in hol_dates
        and nonwork(d - pd.Timedelta(days=1)) and nonwork(d + pd.Timedelta(days=1))
    }

    # v6d: 연휴 지속성 — 연속 비근무(공휴일 or 주말) 블록 길이(주말포함).
    # 공휴일 day에 그 블록 길이를 부여(주말 day 자체는 0, s=7이 흡수). 명절여부는 lunar.
    nonwork_days = sorted(d for d in full_index if (d in hol_dates) or (d.weekday() >= 5))
    run_len: dict[pd.Timestamp, int] = {}
    k = 0
    while k < len(nonwork_days):
        j = k
        while j + 1 < len(nonwork_days) and (nonwork_days[j + 1] - nonwork_days[j]).days == 1:
            j += 1
        length = j - k + 1
        for d in nonwork_days[k:j + 1]:
            run_len[d] = length
        k = j + 1

    df = pd.DataFrame(index=full_index)
    df["hol_lunar"] = df.index.isin(lunar_dates).astype(float)
    df["hol_single"] = df.index.isin(single_dates).astype(float)
    df["lunar_eve"] = df.index.isin(eve).astype(float)
    df["lunar_after"] = df.index.isin(after).astype(float)
    df["is_bridge"] = df.index.isin(bridge).astype(float)
    # 공휴일 day에만 연휴 길이 부여(주말 day는 0)
    df["holiday_run"] = [
        float(run_len.get(d, 0)) if d in hol_dates else 0.0 for d in full_index
    ]
    # 포화(concave) 인코딩 — 긴 연휴의 한계효과 체감(명절 약화)
    df["holiday_run_log"] = np.log1p(df["holiday_run"])
    df["is_myeongjeol"] = df.index.isin(lunar_dates).astype(float)
    return df


def derive_v6_features(data: SplitData) -> SplitData:
    """v3c 파생(OR더미·비선형 기온) + 공휴일 유형 더미 부착, train 통계 재계산."""
    data = v3.derive_extra_features(data)  # heat/cold_th_any, cdd/hdd_sq
    full_index = data.X_train.index.append(data.X_test.index)
    hol = build_holiday_features(full_index)

    for col in HOL_COLS + RUN_COLS + SOFT_COLS:
        data.X_train[col] = hol[col].reindex(data.X_train.index).fillna(0.0).values
        data.X_test[col] = hol[col].reindex(data.X_test.index).fillna(0.0).values

    mu = data.X_train.mean(numeric_only=True)
    sd = data.X_train.std(numeric_only=True)
    return SplitData(data.y_train, data.y_test, data.X_train, data.X_test, mu, sd)


def main(smoke: bool = False) -> None:
    print("=== v6 공휴일 구조화 ===")
    data = v3.load_data()
    data = derive_v6_features(data)
    print(f"train: {data.y_train.shape}, test: {data.y_test.shape}")

    # v3c 외생변수 (14개) — is_holiday 포함
    v3b = [c for c in v3.V2_EXOG if c not in {
        "heat_feels_th_2day_adv", "heat_feels_th_2day_wrn",
        "cold_th_2day_adv", "cold_th_2day_wrn",
    }] + ["heat_th_2day_any", "cold_th_2day_any"]
    v3c_exog = v3b + ["cdd_18_sq", "hdd_18_sq"]

    base_no_hol = [c for c in v3c_exog if c != "is_holiday"]  # 13개 (is_holiday 제거)
    v6a_exog = base_no_hol + ["hol_lunar", "hol_single"]
    v6b_exog = v6a_exog + ["lunar_eve", "lunar_after"]
    v6c_exog = v6b_exog + ["is_bridge"]
    v6d_exog = base_no_hol + ["holiday_run", "is_myeongjeol"]  # 간결: 연휴길이+명절
    # 하이브리드: v3c 기본 더미 유지(단일공휴일 base) + 길이증분 + 명절증분
    v6e_exog = v3c_exog + ["holiday_run", "is_myeongjeol"]
    # 명절 약화 변형
    v6f_exog = v3c_exog + ["is_myeongjeol"]                      # 평탄: 길이無, 명절 flat bump
    v6g_exog = v3c_exog + ["holiday_run_log", "is_myeongjeol"]   # 포화: log(1+run)

    if smoke:
        print("\n[smoke] v3c 재현 (공휴일 항 교체 전, is_holiday 그대로)")
        r = v3.fit_eval(data, v3c_exog, "v3c (smoke)", log_target=True)
        print(f"  AIC={r['AIC']:.2f}  MAE={r['MAE_2025']:.2f}  "
              f"skew={r['resid_skew']:.3f}  kurt={r['resid_kurt']:.3f}  "
              f"LB10={r['LB_p_lag10']:.2e}")
        print("  기대: AIC≈-23830.96, MAE≈2667.51 (v3_results.csv와 일치)")
        return

    results = []
    print("\n[v3c — baseline 재학습 (단일 is_holiday, 14 exog)]")
    results.append(v3.fit_eval(data, v3c_exog, "v3c (baseline)", log_target=True))

    print("[v6a — is_holiday → hol_lunar+hol_single (14 exog)]")
    results.append(v3.fit_eval(data, v6a_exog, "v6a 음력분할", log_target=True))

    print("[v6b — +lunar_eve+lunar_after (16 exog)]")
    results.append(v3.fit_eval(data, v6b_exog, "v6b +연휴위치", log_target=True))

    print("[v6c — +is_bridge (17 exog)]")
    results.append(v3.fit_eval(data, v6c_exog, "v6c +낀평일", log_target=True))

    print("[v6d — 간결: holiday_run+is_myeongjeol (15 exog)]")
    results.append(v3.fit_eval(data, v6d_exog, "v6d 연휴길이+명절", log_target=True))

    print("[v6e — 하이브리드: is_holiday+holiday_run+is_myeongjeol (16 exog)]")
    results.append(v3.fit_eval(data, v6e_exog, "v6e 하이브리드", log_target=True))

    print("[v6f — 평탄 명절: is_holiday+is_myeongjeol (15 exog)]")
    results.append(v3.fit_eval(data, v6f_exog, "v6f 평탄명절", log_target=True))

    print("[v6g — 포화: is_holiday+log(run)+is_myeongjeol (16 exog)]")
    results.append(v3.fit_eval(data, v6g_exog, "v6g 포화명절", log_target=True))

    cmp_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")} for r in results
    ])
    cmp_df.to_csv(OUT_DIR / "v6_results.csv", index=False, encoding="utf-8-sig")

    print("\n=== 단계별 비교 (잔차 개선 중심) ===")
    show = ["model", "n_exog", "AIC", "BIC", "MAE_2025", "MAPE_2025_%",
            "PI95_cov_%", "Winkler", "resid_skew", "resid_kurt",
            "LB_p_lag10", "LB_p_lag20"]
    print(cmp_df[show].round(3).to_string(index=False))

    # 예측 저장 (rolling 포함): v6c(전체 더미) + v6d(간결)
    by_label = {r["model"]: (ex, r) for r, ex in zip(
        results, [v3c_exog, v6a_exog, v6b_exog, v6c_exog, v6d_exog, v6e_exog,
                  v6f_exog, v6g_exog])}
    for label, fname in [("v6c +낀평일", "v6c_forecast.csv"),
                         ("v6d 연휴길이+명절", "v6d_forecast.csv"),
                         ("v6e 하이브리드", "v6e_forecast.csv"),
                         ("v6f 평탄명절", "v6f_forecast.csv"),
                         ("v6g 포화명절", "v6g_forecast.csv")]:
        ex, r = by_label[label]
        print(f"\n[{label} rolling 1-step → {fname}]")
        rolling = v3.rolling_one_step_log(data, ex, r["_fit_obj"])
        out = pd.DataFrame({
            "actual": data.y_test,
            "pred": r["_pred"].values,
            "ci_low": r["_ci"].iloc[:, 0].values,
            "ci_high": r["_ci"].iloc[:, 1].values,
            "rolling": rolling.values,
        }, index=data.y_test.index)
        out.to_csv(OUT_DIR / fname, encoding="utf-8-sig")
        re = data.y_test.values - rolling.values
        print(f"  rolling MAE={np.abs(re).mean():,.0f}, "
              f"MAPE={(np.abs(re)/data.y_test.values).mean()*100:.2f}%")

    # 계수 비교: v6c(유형 더미) vs v6d(연휴길이+명절) vs v6e(하이브리드)
    for label, cols in [("v6c 공휴일 유형", HOL_COLS), ("v6d 간결", RUN_COLS),
                        ("v6e 하이브리드", ["is_holiday"] + RUN_COLS)]:
        res = by_label[{"v6c 공휴일 유형": "v6c +낀평일",
                        "v6d 간결": "v6d 연휴길이+명절",
                        "v6e 하이브리드": "v6e 하이브리드"}[label]][1]["_fit_obj"]
        idx = [n for n in res.params.index if n in cols]
        coef = pd.DataFrame({
            "var": idx,
            "coef": res.params.loc[idx].values,
            "p": res.pvalues.loc[idx].values,
        }).sort_values("coef")
        coef["sig"] = (coef.p < 0.05).map({True: "✓", False: ""})
        print(f"\n=== {label} 계수 ===")
        print(coef.round(4).to_string(index=False))

    print(f"\n저장: {OUT_DIR}/v6_results.csv, v6c_forecast.csv, v6d_forecast.csv, v6e_forecast.csv")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
