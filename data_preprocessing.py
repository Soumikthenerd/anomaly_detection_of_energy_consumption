import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. LOAD RAW DATASET (1,749,494 rows x 57 columns)
# ---------------------------------------------------------
RAW_PATH = "/home/soumik-mondal/Downloads/LEAD/train_features.csv"
CLEAN_OUTPUT_PATH = "/home/soumik-mondal/Documents/aecd_LEAD/clean_reduced_lead_dataset.csv"

print("Loading raw LEAD dataset...")
df = pd.read_csv(RAW_PATH)
print(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ---------------------------------------------------------
# 2. COHORT AUDIT & PRUNING (DROP 16 CORRUPTED BUILDINGS)
# ---------------------------------------------------------
# Buildings with >35% missing meter_reading (e.g., building 144 missing Apr-Sep)
missing_pct_per_bldg = df.groupby("building_id")["meter_reading"].apply(
    lambda s: s.isna().mean() * 100
)
corrupted_buildings = sorted(missing_pct_per_bldg[missing_pct_per_bldg > 35.0].index.tolist())
print(f"Dropping {len(corrupted_buildings)} corrupted buildings (>35% missing): {corrupted_buildings}")

# Expected 16 buildings: [1, 32, 41, 55, 69, 79, 82, 91, 107, 112, 144, 693, 886, 1241, 1264, 1303]
df = df[~df["building_id"].isin(corrupted_buildings)].copy()
print(f"Shape after cohort pruning: {df.shape[0]:,} rows")  # 1,608,955 rows

# ---------------------------------------------------------
# 3. CHRONOLOGICAL SORT & INTRA-BUILDING TARGET IMPUTATION
# ---------------------------------------------------------
# Two-level sort: group by building_id first, then strictly hour-by-hour by timestamp
df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

# Isolated bidirectional linear interpolation + boundary fill per building
df["meter_reading"] = df.groupby("building_id")["meter_reading"].transform(
    lambda s: s.interpolate(method="linear").ffill().bfill()
)
assert df["meter_reading"].isna().sum() == 0, "Target still contains NaNs!"

# ---------------------------------------------------------
# 4. SENTINEL SANITIZATION (57 -> 53 COLUMNS)
# ---------------------------------------------------------
# Prune 4 high-missingness sentinel columns
sparse_sentinel_cols = [
    "floor_count",        # sentinel = 0     (76.0% missing)
    "cloud_coverage",     # sentinel = 255   (45.6% missing)
    "precip_depth_1_hr",  # sentinel = -2    (21.8% missing)
    "wind_direction",     # sentinel = 65535 (10.1% missing)
]
df = df.drop(columns=[c for c in sparse_sentinel_cols if c in df.columns])

# Mask low-sparsity integer sentinels as NaN
df.loc[df["wind_speed"] == -1, "wind_speed"] = np.nan
if "year_built" in df.columns:
    df.loc[df["year_built"] == 0, "year_built"] = np.nan

# Impute wind_speed via site-level (site_id) temporal filling
df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
df["wind_speed"] = df.groupby("site_id")["wind_speed"].transform(
    lambda s: s.interpolate(method="linear").ffill().bfill()
)

# Restore canonical (building_id, timestamp) ordering
df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

# ---------------------------------------------------------
# 5. SAVE CLEANED BASE DATASET (1,608,955 rows x 53 columns)
# ---------------------------------------------------------
df.to_csv(CLEAN_OUTPUT_PATH, index=False)
print(f"Saved cleaned dataset to: {CLEAN_OUTPUT_PATH}")
print(f"Final cleaned shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ---------------------------------------------------------
# 6. PHYSICAL SCALE NORMALIZATION (EUI & 24H ROLLING Z-SCORE)
# ---------------------------------------------------------
# Computed per building prior to V3 feature synthesis and model training
eps = 1e-5
df["meter_per_sqft"] = df["meter_reading"] / (df["square_feet"] + 1.0)

roll_mean_24h = df.groupby("building_id")["meter_reading"].transform(
    lambda s: s.rolling(window=24, min_periods=1).mean()
)
roll_std_24h = df.groupby("building_id")["meter_reading"].transform(
    lambda s: s.rolling(window=24, min_periods=1).std().fillna(0.0)
)
df["z_score_24h"] = np.clip(
    (df["meter_reading"] - roll_mean_24h) / (roll_std_24h + eps), -10.0, 10.0
)
print("Computed EUI (meter_per_sqft) and clipped 24-hour rolling Z-score (z_score_24h).")
