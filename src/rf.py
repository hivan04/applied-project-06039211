import pandas as pd 
from local_config import *

# Load in data
df = pd.read_csv(PROJECT_ROOT / "data/raw_data/DTB3_RF.csv")

# Rename columns
df = df.rename(columns={
    "observation_date": "date",
    "DTB3": "rf"
})

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Sort by date before forward-filling
df = df.sort_values("date")

# Forward-fill missing risk-free rates
df["rf"] = df["rf"].ffill()

# Split the data
# Make sure the date column is datetime
df["date"] = pd.to_datetime(df["date"])

# Set date as the index
df = df.set_index("date")

# Sort the index just to be safe
df = df.sort_index()

# Filter between your desired dates
start_date = "2015-01-01"
end_date = "2023-12-31"

df = df.loc[start_date:end_date]

# Convert rf from percentage to decimal 
df["rf"] = (1+ df["rf"] / 100) ** (1 / 252) -1

# Load in df1 dataframes for is/oos date splits
df1_is = pd.read_csv(PROJECT_ROOT / "data/df1_is",
                     index_col=0, parse_dates=True)
df1_oos = pd.read_csv(PROJECT_ROOT / "data/df1_oos",
                      index_col=0, parse_dates=True)

rf_is = df.loc[df1_is.index.min():df1_is.index.max()]
rf_oos = df.loc[df1_oos.index.min():df1_oos.index.max()]

rf_is.to_csv(PROJECT_ROOT / "data/rf_is")
rf_oos.to_csv(PROJECT_ROOT / "data/rf_oos")