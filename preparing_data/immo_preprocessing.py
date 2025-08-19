import pandas as pd
import geopandas as gpd
import fiona
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict
def select_and_rename_immo_columns(df):
    """
    Selects relevant columns for the immo dataset and renames them to snake_case.
    """
    df = df[['CD_STAT_SECTOR', 'CD_YEAR', 'CD_TYPE', 'CD_TYPE_FR',
             'MS_TRANSACTIONS', 'MS_P25', 'MS_P50 (MEDIAN_PRICE)', 'MS_P75',
             'MS_P10', 'MS_P90']].copy()
    
    df.rename(columns={
        'CD_STAT_SECTOR': 'cd_stat_sector', 
        'CD_YEAR': 'cd_year', 
        'CD_TYPE': 'cd_type', 
        'MS_TRANSACTIONS': 'n', 
        'MS_P25': 'p25', 
        'MS_P50 (MEDIAN_PRICE)': 'p50', 
        'MS_P75': 'p75',
        'MS_P10': 'p10', 
        'MS_P90': 'p90'
    }, inplace=True)
    
    return df


def add_geo_info(
    immo_df: pd.DataFrame,
    geo_parquet_path: str = "data/geo_data_2024.parquet",
    key_col: str = "cd_stat_sector",
    geo_key_raw: str = "cd_sector",
    keep_geo_cols: list | None = None,
    drop_geo_dupes: bool = True):
    """
    Merge geographic attributes into the immo dataframe on STAT sector key.

    Parameters
    ----------
    immo_df : DataFrame
        Your immo data (already renamed to snake_case).
    geo_parquet_path : str
        Path to the geo parquet file.
    key_col : str
        Key column name in immo_df (after renaming); default 'cd_stat_sector'.
    geo_key_raw : str
        Key column name in the raw geo parquet to be renamed to key_col.
    keep_geo_cols : list | None
        If provided, only keep these extra geo columns (key_col is kept automatically).
    drop_geo_dupes : bool
        If True, drop duplicate geo rows by key_col (keep first).

    Returns
    -------
    merged : DataFrame
        immo_df with geo columns merged (left join).
    diagnostics : dict
        Helpful stats about the merge (row counts, unmatched keys, duplicates).
    """

    # --- Load & rename key
    geo = pd.read_parquet(geo_parquet_path)
    if geo_key_raw in geo.columns and geo_key_raw != key_col:
        geo = geo.rename(columns={geo_key_raw: key_col})

    # --- Normalize join keys (strip, as string)
    immo = immo_df.copy()
    immo[key_col] = immo[key_col].astype(str).str.strip()
    geo[key_col]  = geo[key_col].astype(str).str.strip()

    # --- (Optional) select geo columns
    if keep_geo_cols is not None:
        cols = [key_col] + [c for c in keep_geo_cols if c != key_col and c in geo.columns]
        missing = sorted(set(keep_geo_cols) - set(geo.columns))
        if missing:
            print(f"[add_geo_info] Warning: requested geo columns not found: {missing}")
        geo = geo[cols]

    # --- Handle duplicates on geo side
    dup_count = int(geo.duplicated(subset=[key_col]).sum())
    if dup_count and drop_geo_dupes:
        geo = geo.drop_duplicates(subset=[key_col], keep="first")
    elif dup_count:
        print(f"[add_geo_info] Warning: {dup_count} duplicate geo keys present (not dropped).")

    # --- Merge
    before_rows = len(immo)
    merged = immo.merge(geo, how="left", on=key_col)
    after_rows = len(merged)

    # --- Diagnostics
    unmatched_mask = merged.filter(items=[key_col]).notna().iloc[:, 0] & merged.isna().any(axis=1)
    # above heuristic: rows with a key but at least one NaN likely from missing geo match (or legit NaNs in geo)
    # For strict unmatched, compare against geo keys:
    unmatched_strict = ~merged[key_col].isin(geo[key_col])
    unmatched_count = int(unmatched_strict.sum())

    diagnostics = {
        "immo_rows_before": before_rows,
        "rows_after_merge": after_rows,
        "geo_unique_keys": int(geo[key_col].nunique()),
        "geo_duplicates_dropped": int(dup_count if drop_geo_dupes else 0),
        "unmatched_keys_count": unmatched_count,
        "unmatched_keys_pct": round(100 * unmatched_count / max(before_rows, 1), 3),
    }

    if diagnostics["unmatched_keys_count"]:
        sample_unmatched = (
            merged.loc[unmatched_strict, key_col].dropna().astype(str).unique().tolist()[:10]
        )
        diagnostics["unmatched_sample"] = sample_unmatched

    return merged, diagnostics


def backfill_geo_from_cd_stat_sector(
    immo_df: pd.DataFrame,
    geo_df: pd.DataFrame | None = None,
    *,
    geo_parquet_path: str = "data/geo_data_2024.parquet",
    key_col: str = "cd_stat_sector",
    sub_munty_col: str = "cd_sub_munty",
    munty_col: str = "cd_munty_refnis",
    dstr_col: str = "cd_dstr_refnis",
    prov_col: str = "cd_prov_refnis",
    region_col: str = "cd_rgn_refnis",
    unknown_suffix: str = "_UNKNOWN"
    ) -> Tuple[pd.DataFrame, Dict]:
    """
    Backfill missing geographic fields using patterns in cd_stat_sector.

    Strategy
    --------
    - Split rows where `sub_munty_col` is missing into:
        A) 'suspicious' rows: key DOES NOT end with `_UNKNOWN`
        B) 'unknown' rows:   key DOES end with `_UNKNOWN`
    - For both A and B, derive munty/dstr/prov from the first digits of key.
      Then map province -> region using geo_df.
    - Recombine with the rows that weren't missing.

    Returns
    -------
    full_df : DataFrame
        Same number of rows as input, with backfilled columns when possible.
    diagnostics : dict
        Counts and samples of what was fixed/left unresolved.
    """

    # --- Preconditions
    df = immo_df.copy()
    if key_col not in df.columns:
        raise KeyError(f"'{key_col}' not found in immo_df")
    if sub_munty_col not in df.columns:
        raise KeyError(f"'{sub_munty_col}' not found in immo_df")

    # Load geo if needed
    if geo_df is None:
        geo = pd.read_parquet(geo_parquet_path)
    else:
        geo = geo_df.copy()

    # We need a province->region mapping
    for needed in (prov_col, region_col):
        if needed not in geo.columns:
            raise KeyError(f"'{needed}' not found in geo data")

    # Normalize keys as strings
    df[key_col] = df[key_col].astype(str).str.strip()
    if prov_col in df.columns:
        df[prov_col] = df[prov_col].astype(str).str.strip()
    if munty_col in df.columns:
        df[munty_col] = df[munty_col].astype(str).str.strip()
    if dstr_col in df.columns:
        df[dstr_col] = df[dstr_col].astype(str).str.strip()

    # Province -> Region single-valued map (first non-null per province)
    region_by_prov = (
        geo[[prov_col, region_col]]
        .dropna(subset=[prov_col])
        .sort_values(prov_col)
        .drop_duplicates(subset=[prov_col], keep="first")
        .set_index(prov_col)[region_col]
    )

    # Masks
    mask_missing = df[sub_munty_col].isna()
    mask_unknown = mask_missing & df[key_col].str.endswith(unknown_suffix, na=False)
    mask_suspicious = mask_missing & ~mask_unknown

    # Helper to extract codes from key string (digits only, from start)
    # Example keys: '11002J81-' -> '11002' for munty, '11002J' may contain letters; we rely on digits at start
    def extract_munty_from_key(s: pd.Series) -> pd.Series:
        return s.str.extract(r'^(\d{5})')[0]

    def extract_sub_munty_from_key(s: pd.Series) -> pd.Series:
        return s.str.extract(r'^(\d{6})')[0]

    def derive_dstr_from_munty(munty: pd.Series) -> pd.Series:
        # Your original approach keeps length 5 and zeroes last two digits
        # e.g., '11002' -> '11000'
        return munty.str.replace(r'\d{2}$', '00', regex=True)

    def derive_prov_from_munty(munty: pd.Series) -> pd.Series:
        # Your original approach replaced last four digits with zeros on a 5-digit base,
        # which effectively yields something like '10000'.
        # We’ll mirror that behavior for consistency.
        return munty.str.replace(r'\d{4}$', '0000', regex=True)

    # Prepare containers
    not_missing_df = df.loc[~mask_missing].copy()

    # ---- SUSPICIOUS: key does NOT end with _UNKNOWN
    suspicious = df.loc[mask_suspicious].copy()
    if not suspicious.empty:
        munty_ = extract_munty_from_key(suspicious[key_col])
        sub_munty_ = extract_sub_munty_from_key(suspicious[key_col])

        # Only fill where currently missing
        suspicious.loc[suspicious[munty_col].isna(), munty_col] = munty_
        suspicious.loc[suspicious[sub_munty_col].isna(), sub_munty_col] = sub_munty_
        suspicious.loc[suspicious[dstr_col].isna(), dstr_col] = derive_dstr_from_munty(munty_)
        suspicious.loc[suspicious[prov_col].isna(), prov_col] = derive_prov_from_munty(munty_)

        # Region fill: don't drop, just fill nulls
        need_region = suspicious[region_col].isna()
        if need_region.any():
            # map using province
            suspicious.loc[need_region, region_col] = (
                suspicious.loc[need_region, prov_col].map(region_by_prov)
            )

    # ---- UNKNOWN: key ends with _UNKNOWN
    unknown = df.loc[mask_unknown].copy()
    if not unknown.empty:
        munty_unk = extract_munty_from_key(unknown[key_col])

        unknown.loc[unknown[munty_col].isna(), munty_col] = munty_unk
        unknown.loc[unknown[dstr_col].isna(), dstr_col] = derive_dstr_from_munty(munty_unk)
        unknown.loc[unknown[prov_col].isna(), prov_col] = derive_prov_from_munty(munty_unk)

        need_region = unknown[region_col].isna()
        if need_region.any():
            unknown.loc[need_region, region_col] = (
                unknown.loc[need_region, prov_col].map(region_by_prov)
            )

    # Recombine
    parts = [not_missing_df]
    if not suspicious.empty:
        parts.append(suspicious)
    if not unknown.empty:
        parts.append(unknown)

    full_df = pd.concat(parts, ignore_index=True)

    # Diagnostics
    # What remains unresolved (still missing sub_munty after our best effort)?
    still_missing_sub = full_df[sub_munty_col].isna()
    unresolved = full_df.loc[still_missing_sub, key_col].dropna().astype(str).unique().tolist()[:15]

    diagnostics = {
        "rows_total": int(len(df)),
        "initial_missing_sub_munty": int(mask_missing.sum()),
        "suspicious_rows": int(mask_suspicious.sum()),
        "unknown_rows": int(mask_unknown.sum()),
        "remaining_missing_sub_munty": int(still_missing_sub.sum()),
        "unresolved_key_samples": unresolved,
        "shape_unchanged": (full_df.shape[0] == df.shape[0]),
    }

    return full_df, diagnostics

    
def fill_hierarchy_by_type(df, target_col, type_col='cd_type',
                           levels=('cd_sub_munty','cd_munty_refnis','cd_dstr_refnis','cd_prov_refnis'),
                           agg='mean'):
    """
    Fill NaNs in `target_col` by cascading from fine -> coarse geography,
    but always within the same `cd_type`.

    agg: 'mean' (default) or 'median' etc.
    """
    df_copy = df.copy()

    for level in levels:
        # compute stat on the ORIGINAL df to avoid leakage from previous fills
        group_stat = df.groupby([level, type_col])[target_col].transform(agg)

        # fill only where still missing in the working copy
        mask = df_copy[target_col].isna()
        df_copy.loc[mask, target_col] = group_stat[mask]

    return df_copy

# Now expanding the data from aglomerated to units

def expand_to_synthetic_transactions(df):
    """
    Given a DataFrame with quantiles and number of transactions per group (e.g. region-year-type),
    generates a synthetic transaction-level dataset.

    Required columns in df:
        - 'cd_stat_sector', 'cd_year', 'cd_type'
        - 'p10', 'p25', 'p50', 'p75', 'p90', 'n'

    Returns:
        A DataFrame of synthetic prices with metadata and a 'source' column.
    """

    def generate_synthetic_prices(p10, p25, p50, p75, p90, n):
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        values = [p10, p25, p50, p75, p90]
        probs = np.random.uniform(0.1, 0.9, size=n)
        return np.interp(probs, quantiles, values)

    # Drop rows with missing data
    df_clean = df.dropna(subset=['p10', 'p25', 'p50', 'p75', 'p90', 'n'])

    synthetic_rows = []

    for _, row in df_clean.iterrows():
        prices = generate_synthetic_prices(
            p10=row['p10'], p25=row['p25'], p50=row['p50'],
            p75=row['p75'], p90=row['p90'], n=int(row['n'])
        )
        for price in prices:
            synthetic_rows.append({
                'cd_stat_sector': row['cd_stat_sector'],
                'cd_year': row['cd_year'],
                'cd_type': row['cd_type'],
                'price': price,
                'source': 'synthetic'
            })

    return pd.DataFrame(synthetic_rows)