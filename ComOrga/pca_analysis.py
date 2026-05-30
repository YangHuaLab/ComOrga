import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import gaussian_kde

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def assign_groups(n_samples, config):
    labels = []
    palette = {}
    current_idx = 0
    remaining_group = None

    for group in config:
        name = group['name']
        count = group['count']
        palette[name] = group['color']
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

    return labels, palette

def get_significance_asterisks(p_value):
    if pd.isna(p_value): return 'ns'
    if p_value <= 0.001: return '***'
    elif p_value <= 0.01: return '**'
    elif p_value <= 0.05: return '*'
    else: return 'ns'

def process_file(input_file_path, output_dir, relative_path, file_name, group_config,
                 is_rows_are_samples, scatter_size, alpha_val, selected_features=None, log_func=print):
    log_func(f"Processing: {file_name}")

    try:
        df = pd.read_csv(input_file_path, encoding='utf-8-sig')
    except Exception as e:
        log_func(f"Error reading file {file_name}: {e}")
        return

    # 【核心修复】：统一使用明确的方向指令读取和提纯矩阵
    if is_rows_are_samples:
        if selected_features:
            valid_cols = [c for c in selected_features if c in df.columns]
        else:
            valid_cols = df.select_dtypes(include=['number']).columns
        features = df[valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    else:
        if selected_features:
            df = df[df.iloc[:, 0].astype(str).isin(selected_features)]
        # 转置：让 Scikit-learn 的 PCA 接受 标准的 (Samples x Features) 矩阵
        features = df.iloc[:, 1:].T
        features.columns = df.iloc[:, 0].astype(str)
        features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

    if features.empty or features.shape[1] < 2:
        log_func("⚠️ PCA requires at least 2 numerical features. Skipping.")
        return

    n_samples = len(features)
    group_labels, palette_dict = assign_groups(n_samples, group_config)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    pca = PCA()
    pca_scores = pca.fit_transform(X_scaled)
    comp_scores = np.dot(pca_scores, pca.explained_variance_ratio_)

    base_name = os.path.splitext(file_name)[0]
    out_file_base = os.path.join(output_dir, relative_path, base_name)
    os.makedirs(os.path.dirname(out_file_base), exist_ok=True)

    df_result = pd.DataFrame({
        'Sample_Index': range(1, n_samples + 1),
        'Group': group_labels,
        'Comprehensive_Score': comp_scores
    })

    df_result.to_csv(f"{out_file_base}_comprehensive_score.csv", index=False, encoding='utf_8_sig')

    groups_present = [g for g in group_config if g['name'] in df_result['Group'].values]
    order = [g['name'] for g in groups_present]
    if 'Unknown' in df_result['Group'].values:
        order.append('Unknown')
        palette_dict['Unknown'] = '#808080'

    plt.figure(figsize=(9, 7))
    sns.boxplot(x='Group', y='Comprehensive_Score', data=df_result, order=order,
                palette=palette_dict, showfliers=False, width=0.4, boxprops=dict(alpha=0.6, zorder=2))

    for i, group in enumerate(order):
        group_data = df_result[df_result['Group'] == group]['Comprehensive_Score'].values
        if len(group_data) > 1:
            try:
                kde = gaussian_kde(group_data)
                density = kde(group_data)
                jitter_width = 0.35
                density_norm = density / density.max() * jitter_width
                jitter = (np.random.random(len(group_data)) * 2 - 1) * density_norm
                plt.scatter(i + jitter, group_data, color='black', alpha=alpha_val, s=scatter_size, zorder=3, linewidth=0)
            except Exception:
                plt.scatter([i] * len(group_data), group_data, color='black', alpha=alpha_val, s=scatter_size)

    y_max, y_min = df_result['Comprehensive_Score'].max(), df_result['Comprehensive_Score'].min()
    y_range = y_max - y_min
    h_base = y_max + y_range * 0.05
    step = y_range * 0.1

    if len(order) >= 2:
        group_arrays = [df_result[df_result['Group'] == g]['Comprehensive_Score'].values for g in order]
        try:
            f_stat, anova_p = stats.f_oneway(*group_arrays)
            plt.title(f"Comprehensive Score by Group\n(ANOVA p={anova_p:.2e})", fontsize=12, pad=15)
        except Exception:
            plt.title("Comprehensive Score by Group", fontsize=12, pad=15)

        target_controls = ['C', 'M1']
        pairs = [(g1, g2) for g1, g2 in itertools.combinations(order, 2) if g1 in target_controls or g2 in target_controls]
        valid_pair_idx = 0
        for g1_name, g2_name in pairs:
            g1_data = df_result[df_result['Group'] == g1_name]['Comprehensive_Score'].values
            g2_data = df_result[df_result['Group'] == g2_name]['Comprehensive_Score'].values
            if len(g1_data) > 1 and len(g2_data) > 1:
                _, p_val = stats.ttest_ind(g1_data, g2_data, equal_var=False)
                x1, x2 = order.index(g1_name), order.index(g2_name)
                h = h_base + (valid_pair_idx * step)
                plt.plot([x1, x1, x2, x2], [h, h + y_range * 0.02, h + y_range * 0.02, h], lw=1.2, c='k')
                plt.text((x1 + x2) / 2, h + y_range * 0.02, get_significance_asterisks(p_val), ha='center', va='bottom', color='k')
                valid_pair_idx += 1
        plt.ylim(y_min - y_range * 0.1, h_base + (valid_pair_idx * step) + y_range * 0.1)
    else:
        plt.title("Comprehensive Score by Group", fontsize=12, pad=15)

    plt.ylabel("PCA Comprehensive Score"); plt.xlabel("Group")
    plt.tight_layout()
    plt.savefig(f"{out_file_base}_comprehensive_score_boxplot.svg", format='svg')
    plt.savefig(f"{out_file_base}_comprehensive_score_boxplot.png", format='png', dpi=300)
    plt.close()

def run_pca_comprehensive(input_path, output_dir, group_config, is_rows_are_samples=True,
                          scatter_size=3, alpha=0.05, selected_features=None, status_callback=None):
    def log(msg):
        if status_callback: status_callback(msg)
        else: print(msg)

    if not os.path.exists(input_path): return False
    file_count = 0

    if os.path.isfile(input_path) and input_path.lower().endswith('.csv'):
        process_file(input_path, output_dir, '', os.path.basename(input_path), group_config,
                     is_rows_are_samples, scatter_size, alpha, selected_features, log)
        file_count += 1
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith('.csv'):
                    process_file(os.path.join(root, file), output_dir, os.path.relpath(root, input_path).replace('.',''),
                                 file, group_config, is_rows_are_samples, scatter_size, alpha, selected_features, log)
                    file_count += 1

    return file_count > 0