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
    drop_geo_dupes: bool = True,
):
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
