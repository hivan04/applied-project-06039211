import requests 
import pandas as pd
from local_config import *

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

    return rf

# Fetch using function from above and save to a dataframe
rf_honia = pull_hk_rf(start_date="2021-01-01", end_date="2023-12-31")

print(rf_honia.head())
print(f"\nAnnualised range (%):\n{(rf_honia['rf'] * 252 * 100).describe().round(4)}")

# Save to CSV 
rf_honia.to_csv(PROJECT_ROOT / "data/raw_data/rf_honia.csv", index=True)
print("\nSaved to rf_honia.csv")

# Ensure that the date is time-set before splitting data
# rf_honia["date"] = pd.to_datetime(rf_honia["date"])
# rf_honia = rf_honia.set_index("date").sort_index()

# Split the risk-free rate into the training and testing set split respectively 

split_date = pd.Timestamp("2021-01-01")

rf_is  = rf_honia.loc[rf_honia.index < split_date].copy()
rf_oos = rf_honia.loc[rf_honia.index >= split_date].copy()

rf_is.to_csv(PROJECT_ROOT / "data/rf_is")
rf_oos.to_csv(PROJECT_ROOT / "data/rf_oos")


