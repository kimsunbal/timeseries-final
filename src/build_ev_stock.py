"""
전기차 등록대수(월별 누적 stock) → 일별 외생변수 산출 (Phase 2 민감도 분석 전용).

본모델(ARIMAX v3 계열) 외생변수 셋과 분리된 **독립 산출물**이다.
전기차 변수는 추세항(ip_total/pop_total)과 공선이고 2010~2017 구간이
바닥(~1%)이라 본모델에 넣지 않고, 서브기간 민감도 분석에서만 사용한다.
경위: docs/research_log/2026-05-30_1100_신산업_전기차_외생변수_검토.md

입력:  data/201001_202605_전기차등록현황.xls  (시·도별 월 누적, 합계 컬럼)
출력:  data/processed/ev_stock_daily_2010_2025.csv  (5,844행)
  - ev_stock : 전국 누적 등록대수 (월값을 해당 월 모든 날에 ffill)
  - ev_flow  : 월별 신규 등록(증분 = stock 월차분, 첫 월은 stock 자체)

가공 원칙:
  - 월 → 일: 해당 월 stock/flow 값을 그 달 모든 일자에 동일 적용(ffill).
    (ip_total/pop_total 일별 변환과 동일한 step-function 방식)
  - EV 원자료는 2025-10까지 → 2025-11·12는 마지막 관측월(2025-10) 값으로
    ffill. (stock은 거의 불변, flow는 0 처리하여 인위적 증분 방지)
  - 일자 인덱스는 종속변수(peak_mw_2010_2025.csv)와 1:1 정렬.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_XLS = ROOT / "data" / "201001_202605_전기차등록현황.xls"
PEAK_CSV = ROOT / "data" / "processed" / "peak_mw_2010_2025.csv"
OUT_CSV = ROOT / "data" / "processed" / "ev_stock_daily_2010_2025.csv"


def load_monthly() -> pd.DataFrame:
    """xls에서 월별 전국 합계 stock 시계열 추출 (오름차순)."""
    raw = pd.read_excel(RAW_XLS, engine="xlrd", header=3)
    raw = raw.rename(columns={raw.columns[0]: "ym"})
    raw = raw.dropna(subset=["ym"]).copy()
    raw["ym"] = raw["ym"].astype(str).str.strip()
    raw = raw[raw["ym"].str.match(r"\d{4}-\d{2}")].copy()

    df = pd.DataFrame({
        "ym": pd.PeriodIndex(raw["ym"], freq="M"),
        "ev_stock": pd.to_numeric(raw["합계"], errors="coerce"),
    }).dropna()
    df = df.sort_values("ym").reset_index(drop=True)

    # flow = 월별 신규 등록 (누적 stock의 월차분), 첫 월은 stock 자체(0에서 출발)
    df["ev_flow"] = df["ev_stock"].diff()
    df.loc[df.index[0], "ev_flow"] = df.loc[df.index[0], "ev_stock"]
    # 등록말소 우위 등으로 음수 증분이 나오면 0으로 (신규 등록 신호이므로)
    df["ev_flow"] = df["ev_flow"].clip(lower=0)
    return df


def expand_daily(monthly: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """월값을 일자 인덱스에 매핑(같은 달 → 같은 값). 데이터 종료 후는 ffill."""
    m = monthly.set_index("ym")
    ym_of_day = pd.PeriodIndex(daily_index, freq="M")

    stock = m["ev_stock"].reindex(ym_of_day)
    flow = m["ev_flow"].reindex(ym_of_day)
    stock.index = daily_index
    flow.index = daily_index

    # 원자료 종료월(2025-10) 이후: stock은 마지막 값 ffill, flow는 0
    last_ym = m.index.max()
    after = np.asarray(ym_of_day > last_ym)
    stock = stock.ffill()
    flow[after] = 0.0
    flow = flow.fillna(0.0)

    out = pd.DataFrame({"ev_stock": stock.astype(float), "ev_flow": flow.astype(float)})
    out.index.name = "tm"
    return out


def main() -> None:
    monthly = load_monthly()
    print(f"월별 stock: {len(monthly)}개월  {monthly['ym'].min()} ~ {monthly['ym'].max()}")

    peak = pd.read_csv(PEAK_CSV, encoding="utf-8-sig", parse_dates=["tm"])
    daily_index = pd.DatetimeIndex(peak["tm"]).sort_values()

    out = expand_daily(monthly, daily_index)
    assert out.isna().sum().sum() == 0, "결측 발생"
    assert len(out) == len(daily_index), "행수 불일치"

    out.to_csv(OUT_CSV, encoding="utf-8-sig")

    # sanity
    ye = out["ev_stock"].groupby(out.index.year).last()
    print(f"\n저장: {OUT_CSV}  ({len(out)}행)")
    print("연말 누적 stock:")
    for y, v in ye.items():
        print(f"  {y}: {int(v):>9,}")
    print(f"\nev_flow 합(연도별 신규, 일중복 제거 위해 월기준):")
    mf = monthly.set_index("ym")["ev_flow"]
    print(mf.groupby(mf.index.year).sum().astype(int).to_string())


if __name__ == "__main__":
    main()
