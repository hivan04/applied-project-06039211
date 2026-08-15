import sys
from pathlib import Path

# Make `from src...` imports work regardless of how/where this file is run
# (repo root, `src/`, IDE "Run" button, etc.) by putting the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import requests
import pandas as pd
from src.local_config import *

def pull_hk_rf(
        start_date = "2015-01-01",
        end_date = "2026-04-20"):
    """ 
    Fetch HONIA (overnight) rate from HKMA API.
    Returns a date-indexed DataFrame with a single 'rf' column
    in daily decimal format, matching the rf_oos CSV structure.
    
    Collect the two main segments that are required for the risk-free rate:
    Segments:
        'hibor.fixing'  -> HIBOR daily fixings
        'honia'         -> HONIA overnight rate
    """
    base_url = (
        "https://api.hkma.gov.hk/public/market-data-and-statistics/"
        "monthly-statistical-bulletin/er-ir/hk-interbank-ir-daily"
    )

    all_records = []
    offset = 0
    limit = 100

    while True:
        params = {
            "segment": "honia",
            "offset": offset,
            "from": start_date,
            "to": end_date,
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Correct key is 'records' not 'dataSet'
        records = data.get("result", {}).get("records", [])
        if not records:
            break

        all_records.extend(records)

        # Stop when we get fewer than a full page
        if len(records) < limit:
            break

        offset += limit

    if not all_records:
        raise ValueError("No data returned — check date range or segment name.")

    df = pd.DataFrame(all_records)

    rf = (
        df[["end_of_day", "ir_overnight"]]
        .rename(columns={"end_of_day": "date", "ir_overnight": "rf"})
        .assign(
            date=lambda x: pd.to_datetime(x["date"]),
            # API returns annualised % (e.g. 2.89 = 2.89%) -> daily decimal
            rf=lambda x: x["rf"].astype(float) / 100 / 252
        )
        .set_index("date")
        .sort_index()
    )

    # The HKMA API does not reliably honour the from/to params (it can return
    # its full available history regardless), so enforce the requested window
    # ourselves to keep this function's output deterministic and match its docstring.
    rf = rf.loc[start_date:end_date]

    return rf


def main():
    """Fetch HONIA, save the full series and the IS/OOS split to data/.

    Date range and split point match notebook 1's price-data setup
    (START_DATE="2015-01-01"/"2026-01-01", split_date="2022-10-01") so the
    risk-free series lines up with the return series it gets reindexed against.
    """
    rf_honia = pull_hk_rf(start_date="2015-01-01", end_date="2026-01-01")

    print(rf_honia.head())
    print(f"\nAnnualised range (%):\n{(rf_honia['rf'] * 252 * 100).describe().round(4)}")

    rf_honia.to_csv(PROJECT_ROOT / "data/raw_data/rf_honia.csv", index=True)
    print("\nSaved to rf_honia.csv")

    # Split the risk-free rate into the training and testing set split respectively
    split_date = pd.Timestamp("2022-10-01")

    rf_is  = rf_honia.loc[rf_honia.index < split_date].copy()
    rf_oos = rf_honia.loc[rf_honia.index >= split_date].copy()

    rf_is.to_csv(PROJECT_ROOT / "data/rf_is")
    rf_oos.to_csv(PROJECT_ROOT / "data/rf_oos")


if __name__ == "__main__":
    main()


