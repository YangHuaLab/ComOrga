import os
import subprocess
import pandas as pd
import shutil

def assign_groups(n_samples, config):
    labels = []
    current_idx = 0
    remaining_group = None
    for group in config:
        name = group['name']
        count = group['count']
        if count == 0:
            remaining_group = name
            continue
        if current_idx < n_samples:
            end_idx = min(current_idx + count, n_samples)
            labels.extend([name] * (end_idx - current_idx))
            current_idx = end_idx
    if remaining_group and current_idx < n_samples:
        labels.extend([remaining_group] * (n_samples - current_idx))
    if len(labels) < n_samples:
        labels.extend(['Unknown'] * (n_samples - len(labels)))
    return labels

def _process_single_csv(csv_file, r_script_path, output_dir, group_config,
                        is_rows_are_samples, perplexity, point_size, point_alpha,
                        rscript_exec, selected_features, log):
    temp_data_csv = os.path.join(output_dir, "temp_data_matrix.csv")
    temp_group_csv = os.path.join(output_dir, "temp_group_labels.csv")

    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')

        if is_rows_are_samples:
            if selected_features:
                valid_cols = [c for c in selected_features if c in df.columns]
            else:
                valid_cols = df.select_dtypes(include=['number']).columns
            df_numeric = df[valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            aligned_df = df_numeric.T
            sample_names = [f"Sample_{i}" for i in range(df.shape[0])]
            n_samples = df.shape[0]
        else:
            if selected_features:
                df = df[df.iloc[:, 0].astype(str).isin(selected_features)]
            df_numeric = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0)
            aligned_df = df_numeric
            sample_names = df.columns[1:].tolist()
            n_samples = len(sample_names)

        aligned_df.to_csv(temp_data_csv, index=True, index_label="Feature", encoding='utf-8-sig')
        labels = assign_groups(n_samples, group_config)

        # 【核心修复1】：提取 UI 配置的颜色字典，把每一个样本的颜色也记录下来
        color_dict = {g['name']: g['color'] for g in group_config}
        colors = [color_dict.get(lbl, '#808080') for lbl in labels]

        # 连同 Color 列一起传给 R 脚本
        df_groups = pd.DataFrame({"sample": sample_names, "Type": labels, "Color": colors})
        df_groups.to_csv(temp_group_csv, index=False, encoding='utf-8-sig')

    except Exception as e:
        log(f"❌ Error preparing data for {os.path.basename(csv_file)}: {str(e)}")
        return False

    cmd = [
        rscript_exec, r_script_path, temp_data_csv, temp_group_csv,
        output_dir, str(perplexity), str(point_size), str(point_alpha)
    ]

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding='utf-8', errors='replace'
        )
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None: break
            if output and output.strip(): log(f"[R-Console] {output.strip()}")

        if process.poll() == 0:
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            default_out = os.path.join(output_dir, "tsne_merged_result.tiff")
            final_out = os.path.join(output_dir, f"{base_name}_tsne.tiff")
            if os.path.exists(default_out): shutil.move(default_out, final_out)
            return True
        return False
    except Exception as e:
        log(f"Failed to run R process: {str(e)}")
        return False
    finally:
        for tmp_file in [temp_data_csv, temp_group_csv]:
            if os.path.exists(tmp_file):
                try: os.remove(tmp_file)
                except Exception: pass

def run_merged_tsne_by_r(r_script_path, input_path, output_dir, group_config,
                         is_rows_are_samples=True, perplexity=30,
                         point_size=1.5, point_alpha=0.6, selected_features=None,
                         rscript_exec="Rscript", status_callback=None):
    def log(msg):
        if status_callback: status_callback(msg)
        else: print(msg)

    if not os.path.exists(input_path): return False
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    csv_files = []
    if os.path.isfile(input_path) and input_path.lower().endswith('.csv'):
        csv_files.append(input_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith('.csv'): csv_files.append(os.path.join(root, f))

    if not csv_files: return False

    success_count = 0
    for idx, csv_file in enumerate(csv_files):
        log(f"\n--- [{idx+1}/{len(csv_files)}] Processing: {os.path.basename(csv_file)} ---")
        ok = _process_single_csv(
            csv_file, r_script_path, output_dir, group_config,
            is_rows_are_samples, perplexity, point_size, point_alpha,
            rscript_exec, selected_features, log
        )
        if ok: success_count += 1

    return success_count > 0