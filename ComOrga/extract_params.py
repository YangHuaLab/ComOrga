import os
import pandas as pd
import re


def split_and_save(df, output_dir, file_prefix, group_by, target_cols=None, drop_na_rows=False, log=print):
    """
    Split DataFrame based on Metadata_Well and save/append to corresponding CSVs.
    """
    # 1. Filter target columns if specified
    if target_cols:
        if 'Metadata_Well' not in target_cols:
            target_cols.insert(0, 'Metadata_Well')
        existing_cols = [c for c in target_cols if c in df.columns]
        df = df[existing_cols].copy()
    else:
        existing_cols = df.columns.tolist()

    # 2. Check missing parameters
    if target_cols:
        missing_cols = set(target_cols) - set(existing_cols)
        if missing_cols:
            log(f"    -> Warning: {len(missing_cols)} target parameter(s) missing in CSV (Ignored).")

    # 3. Drop rows with missing values if requested
    if drop_na_rows:
        original_len = len(df)
        df = df.dropna()
        dropped_len = original_len - len(df)
        if dropped_len > 0:
            log(f"    -> Cleaned up {dropped_len} incomplete row(s) containing missing values.")

    if df.empty:
        log(f"    -> Error: DataFrame is empty after filtering. Skipping {file_prefix}.")
        return False

    # 4. Ensure Metadata_Well exists for grouping
    if 'Metadata_Well' not in df.columns:
        log(f"    -> Warning: 'Metadata_Well' not found. Saving as a single file without grouping.")
        out_path = os.path.join(output_dir, f"{file_prefix}_all.csv")
        if os.path.exists(out_path):
            df.to_csv(out_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(out_path, mode='w', header=True, index=False, encoding='utf-8-sig')
        return True

    # 5. Group by Metadata_Well and parse Letter/Number
    for well, group_df in df.groupby('Metadata_Well'):
        well_str = str(well).strip()
        match = re.match(r'([A-Za-z]+)(\d+)', well_str)

        if match:
            letter, number = match.groups()
            key = letter if 'Letter' in group_by else number
        else:
            key = 'Unknown'

        out_path = os.path.join(output_dir, f"{key}_{file_prefix}.csv")

        # 6. Save or Append
        if os.path.exists(out_path):
            group_df.to_csv(out_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            group_df.to_csv(out_path, mode='w', header=True, index=False, encoding='utf-8-sig')

    return True


def extract_target_parameters(input_csv_dir, out_cell_dir, out_image_dir, group_by='Letter',
                              extract_mode='preset', custom_mode='include', custom_params=None,
                              status_callback=None):
    """
    Main extraction and grouping function called by GUI.
    """

    def log(msg):
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    if not os.path.exists(input_csv_dir):
        log(f"Error: Input directory does not exist -> {input_csv_dir}")
        return False, "Input directory not found."

    os.makedirs(out_cell_dir, exist_ok=True)
    os.makedirs(out_image_dir, exist_ok=True)

    cell_csv = os.path.join(input_csv_dir, "Control_Cell_Watershed.csv")
    image_csv = os.path.join(input_csv_dir, "Control_Image.csv")

    # 预设核心参数列表 (Preset List)
    target_params_preset = [
        'Metadata_Well',
        'Correlation_Correlation_ER_Mito',
        'Correlation_Correlation_ER_Threshold_Mito_Threshold',
        'Correlation_K_ER_Mito',
        'Correlation_K_Mito_ER',
        'Correlation_Manders_ER_Mito',
        'Correlation_Manders_ER_Threshold_Mito_Threshold',
        'Correlation_Manders_Mito_ER',
        'Correlation_Manders_Mito_Threshold_ER_Threshold',
        'Correlation_Overlap_ER_Mito',
        'Correlation_RWC_ER_Mito',
        'Correlation_RWC_ER_Threshold_Mito_Threshold',
        'Correlation_RWC_Mito_ER',
        'Correlation_RWC_Mito_Threshold_ER_Threshold'
    ]

    success_any = False

    def get_target_cols(df_columns):
        if extract_mode == 'all':
            return None
        elif extract_mode == 'preset':
            return target_params_preset.copy()
        elif extract_mode == 'custom':
            if custom_mode == 'include':
                return custom_params.copy() if custom_params else []
            else:  # exclude
                return [c for c in df_columns if c not in custom_params]
        return None

    # --- Process Cell Watershed CSV ---
    if os.path.exists(cell_csv):
        log(f"Processing {os.path.basename(cell_csv)}... (Mode: {extract_mode})")
        df_cell = pd.read_csv(cell_csv, encoding='utf-8-sig')
        df_cell.columns = df_cell.columns.str.strip()

        t_cols = get_target_cols(df_cell.columns)
        split_and_save(df_cell, out_cell_dir, "Cell_Watershed", group_by,
                       target_cols=t_cols, drop_na_rows=True, log=log)
        success_any = True
    else:
        log(f"Warning: {os.path.basename(cell_csv)} not found in the input folder.")

    # --- Process Image CSV ---
    if os.path.exists(image_csv):
        log(f"Processing {os.path.basename(image_csv)}... (Mode: {extract_mode})")
        df_image = pd.read_csv(image_csv, encoding='utf-8-sig')
        df_image.columns = df_image.columns.str.strip()

        t_cols = get_target_cols(df_image.columns)
        split_and_save(df_image, out_image_dir, "Image", group_by,
                       target_cols=t_cols, drop_na_rows=False, log=log)
        success_any = True
    else:
        log(f"Warning: {os.path.basename(image_csv)} not found in the input folder.")

    # --- Generate Summary Table for Image Data ---
    if success_any and os.path.exists(out_image_dir):
        log("Generating parameter mean summary for individual wells in Image groups...")
        out_image_summary_dir = out_image_dir + "_Summary"
        os.makedirs(out_image_summary_dir, exist_ok=True)

        for filename in os.listdir(out_image_dir):
            if filename.endswith("_Image.csv"):
                group_prefix = filename.replace("_Image.csv", "")
                filepath = os.path.join(out_image_dir, filename)
                try:
                    df_group = pd.read_csv(filepath, encoding='utf-8-sig')

                    if 'Metadata_Well' in df_group.columns:
                        well_means = df_group.groupby('Metadata_Well').mean(numeric_only=True).reset_index()
                        well_means.insert(0, 'Group_Prefix', group_prefix)

                        base_cols = ['Group_Prefix', 'Metadata_Well']
                        other_cols = [c for c in well_means.columns if c not in base_cols]
                        summary_df = well_means[base_cols + other_cols]
                    else:
                        numeric_df = df_group.select_dtypes(include='number')
                        if not numeric_df.empty:
                            mean_series = numeric_df.mean()
                            mean_dict = mean_series.to_dict()
                            mean_dict['Group_Prefix'] = group_prefix
                            mean_dict['Metadata_Well'] = 'Unknown'
                            summary_df = pd.DataFrame([mean_dict])

                            base_cols = ['Group_Prefix', 'Metadata_Well']
                            other_cols = [c for c in summary_df.columns if c not in base_cols]
                            summary_df = summary_df[base_cols + other_cols]
                        else:
                            continue

                    summary_df = summary_df.sort_values(by=['Metadata_Well'])
                    summary_out_path = os.path.join(out_image_summary_dir, f"{group_prefix}_Image_Well_Means.csv")
                    summary_df.to_csv(summary_out_path, index=False, encoding='utf-8-sig')
                    log(f"    -> Saved summary for '{group_prefix}' to: {os.path.basename(summary_out_path)}")
                except Exception as e:
                    log(f"    -> Warning: Could not calculate mean for {filename}: {e}")

    if success_any:
        log(f"🎉 Parameter extraction and grouping by '{group_by}' completed successfully!")
        return True, "Extraction completed."
    else:
        log("❌ Error: Neither Cell nor Image CSVs were found.")
        return False, "Target CSV files missing."