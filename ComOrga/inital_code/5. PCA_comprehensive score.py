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

# ==========================================
# 1. 实验配置区
# ==========================================
# 自定义输入与输出路径
# 提示：这里现在可以填入一个文件夹路径，也可以直接填入一个具体的 .csv 文件路径
INPUT_PATH = r'F:\work\pca\2\siv.csv'
OUTPUT_DIR = r'F:\work\pca\分析数据\2'

# 备用特征参数方向（当无法自动检测出 14 个维度时，作为后备方案使用）
# ('horizontal' 表示特征是列，样本是行；'vertical' 表示特征是行，样本是列)
FEATURE_DIRECTION = 'vertical'

# 自定义组别配置
# count: 样本数。填 0 表示分配剩余所有样本（仅允许一组为0）
# color: 该组在箱图和散点中的颜色 (支持十六进制或常见颜色名)
GROUP_CONFIG = [
    {'name': 'C', 'count': 1000, 'color': '#0000FF'},
    {'name': 'NC', 'count': 1000, 'color': '#8B00FF'},
    {'name': 'SIRNA', 'count': 1000, 'color': '#00A6FF'},
    #{'name': '抑制剂', 'count': 3, 'color': '#00A6FF'},
    {'name': 'M1', 'count': 1000, 'color': '#FF0000'},
    {'name': 'M1+NC', 'count': 1000, 'color': '#FF7F00'},
    {'name': 'M1+SIRNA', 'count': 1000, 'color': '#ffc0cb'},
    #{'name': 'M1+抑制剂', 'count': 3, 'color': '#FF7F00'},
    {'name': 'M2', 'count': 1000, 'color': '#999999'},
    #{'name': 'M2+抑制剂', 'count': 3, 'color': '#000000'},
    {'name': 'M2+NC', 'count': 1000, 'color': '#000000'},
    {'name': 'M2+SIRNA', 'count': 1000, 'color': '#D9D9D9'},
    #{'name': 'T', 'count': 0, 'color': '#FFA500'},
]

# 散点大小
SCATTER_POINT_SIZE = 3
alp = 0.05

# ==========================================
# 2. 基础环境配置
# ==========================================
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 3. 核心功能函数
# ==========================================
def assign_groups(n_samples, config):
    """根据配置将样本划分到对应组别，并提取颜色映射"""
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

    # 处理 count 为 0 的组（填充剩余样本）
    if remaining_group and current_idx < n_samples:
        labels.extend([remaining_group] * (n_samples - current_idx))

    # 如果样本仍未分完（配置的总数不足且没有设为0的组），补充为 Unknown
    if len(labels) < n_samples:
        labels.extend(['Unknown'] * (n_samples - len(labels)))

    return labels, palette


def load_csv_data(file_path, fallback_direction='horizontal'):
    """读取 CSV 数据并自动检测参数方向（特征维度应为14）"""
    try:
        data = pd.read_csv(file_path, header=None)

        # 自动检测逻辑：判断哪一个维度的长度为 14 (表头 + 13 个参数)
        if data.shape[0] == 14 and data.shape[1] != 14:
            # 行数是 14，说明特征在行上，样本在列上。需要转置。
            data = data.T
            data.reset_index(drop=True, inplace=True)
        elif data.shape[1] == 14 and data.shape[0] != 14:
            # 列数是 14，说明特征在列上，样本在行上。无需转置。
            pass
        else:
            # 如果都不为14，或都为14，退回使用配置区的备用方向判断
            if fallback_direction == 'vertical':
                data = data.T
                data.reset_index(drop=True, inplace=True)

        # 提取表头
        header = data.iloc[0].values
        data = data.iloc[1:].copy()
        data.columns = header.astype(str)

        # 去除全空列并转为数值
        features = data.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').fillna(0)
        return features
    except Exception as e:
        print(f"读取文件出错 {file_path}: {e}")
        return None


def get_significance_asterisks(p_value):
    """根据 p 值返回显著性标记"""
    if pd.isna(p_value):
        return 'ns'
    if p_value <= 0.001:
        return '***'
    elif p_value <= 0.01:
        return '**'
    elif p_value <= 0.05:
        return '*'
    else:
        return 'ns'


def process_file(input_file_path, output_dir, relative_path, file_name):
    """处理单个文件的 PCA 及综合得分计算与绘图"""
    print(f"正在处理: {file_name}")

    # 1. 读取数据
    features = load_csv_data(input_file_path, FEATURE_DIRECTION)
    if features is None or features.empty:
        print("数据无效，跳过。")
        return

    n_samples = len(features)
    group_labels, palette_dict = assign_groups(n_samples, GROUP_CONFIG)

    # 2. PCA 处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    pca = PCA()
    pca_scores = pca.fit_transform(X_scaled)

    # 3. 计算综合得分 (各主成分得分乘其方差贡献率求和)
    comp_scores = np.dot(pca_scores, pca.explained_variance_ratio_)

    # 4. 构建得分数据表并保存
    base_name = os.path.splitext(file_name)[0]
    out_file_base = os.path.join(output_dir, relative_path, base_name)
    os.makedirs(os.path.dirname(out_file_base), exist_ok=True)

    df_result = pd.DataFrame({
        'Sample_Index': range(1, n_samples + 1),
        'Group': group_labels,
        'Comprehensive_Score': comp_scores
    })

    csv_out_path = f"{out_file_base}_comprehensive_score.csv"
    df_result.to_csv(csv_out_path, index=False, encoding='utf_8_sig')

    # 5. 绘制综合得分箱图
    groups_present = [g for g in GROUP_CONFIG if g['name'] in df_result['Group'].values]
    order = [g['name'] for g in groups_present]

    # 如果数据中存在 Unknown 组，补充到最后
    if 'Unknown' in df_result['Group'].values:
        order.append('Unknown')
        palette_dict['Unknown'] = '#808080'

    plt.figure(figsize=(9, 7))
    sns.boxplot(x='Group', y='Comprehensive_Score', data=df_result, order=order,
                palette=palette_dict, showfliers=False, width=0.4,
                boxprops=dict(alpha=0.6, zorder=2))

    # 添加 KDE 密度散点
    for i, group in enumerate(order):
        group_data = df_result[df_result['Group'] == group]['Comprehensive_Score'].values
        if len(group_data) > 1:
            try:
                kde = gaussian_kde(group_data)
                density = kde(group_data)
                jitter_width = 0.35
                density_norm = density / density.max() * jitter_width
                jitter = (np.random.random(len(group_data)) * 2 - 1) * density_norm
                plt.scatter(i + jitter, group_data, color='black', alpha=alp,
                            s=SCATTER_POINT_SIZE, zorder=3, linewidth=0)
            except np.linalg.LinAlgError:
                # 处理数据完全一致导致无法计算密度的极少情况
                plt.scatter([i] * len(group_data), group_data, color='black', alpha=alp, s=SCATTER_POINT_SIZE)

    # 显著性计算 (ANOVA 及两两 T 检验)
    y_max = df_result['Comprehensive_Score'].max()
    y_min = df_result['Comprehensive_Score'].min()
    y_range = y_max - y_min

    # 动态设定检验线的高度基础值与步长
    h_base = y_max + y_range * 0.05
    step = y_range * 0.1

    if len(order) >= 2:
        # 整体 ANOVA
        group_arrays = [df_result[df_result['Group'] == g]['Comprehensive_Score'].values for g in order]
        try:
            f_stat, anova_p = stats.f_oneway(*group_arrays)
            plt.title(f"Comprehensive Score by Group\n(ANOVA p={anova_p:.2e})", fontsize=12, pad=15)
        except Exception:
            plt.title("Comprehensive Score by Group", fontsize=12, pad=15)

        # 两两比较 T 检验连线
        # 修改为：严格限定对比目标，仅保留参与方等于 'C' 或等于 'M1' 的组合
        target_controls = ['C', 'M1']
        pairs = []
        for g1, g2 in itertools.combinations(order, 2):
            if g1 in target_controls or g2 in target_controls:
                pairs.append((g1, g2))

        valid_pair_idx = 0
        for g1_name, g2_name in pairs:
            g1_data = df_result[df_result['Group'] == g1_name]['Comprehensive_Score'].values
            g2_data = df_result[df_result['Group'] == g2_name]['Comprehensive_Score'].values

            if len(g1_data) > 1 and len(g2_data) > 1:
                _, p_val = stats.ttest_ind(g1_data, g2_data, equal_var=False)
                x1 = order.index(g1_name)
                x2 = order.index(g2_name)

                h = h_base + (valid_pair_idx * step)
                plt.plot([x1, x1, x2, x2], [h, h + y_range * 0.02, h + y_range * 0.02, h], lw=1.2, c='k')
                plt.text((x1 + x2) / 2, h + y_range * 0.02, get_significance_asterisks(p_val),
                         ha='center', va='bottom', color='k')
                valid_pair_idx += 1

        plt.ylim(y_min - y_range * 0.1, h_base + (valid_pair_idx * step) + y_range * 0.1)
    else:
        plt.title("Comprehensive Score by Group", fontsize=12, pad=15)

    plt.ylabel("PCA Comprehensive Score")
    plt.xlabel("Group")
    plt.tight_layout()

    svg_out_path = f"{out_file_base}_comprehensive_score_boxplot.svg"
    plt.savefig(svg_out_path, format='svg')
    plt.close()


# ==========================================
# 4. 主执行流程
# ==========================================
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"输入的路径不存在: {INPUT_PATH}")
        return

    file_count = 0

    # 判断输入的是单个文件还是文件夹
    if os.path.isfile(INPUT_PATH):
        if INPUT_PATH.lower().endswith('.csv'):
            file_name = os.path.basename(INPUT_PATH)
            # 单个文件输出时不嵌套相对路径子文件夹
            process_file(INPUT_PATH, OUTPUT_DIR, '', file_name)
            file_count += 1
        else:
            print(f"指定的文件不是 CSV 格式: {INPUT_PATH}")

    elif os.path.isdir(INPUT_PATH):
        # 遍历文件夹及其子文件夹
        for root, dirs, files in os.walk(INPUT_PATH):
            for file in files:
                if file.lower().endswith('.csv'):
                    input_file_path = os.path.join(root, file)
                    # 计算相对路径，用于在输出目录中建立同名子文件夹
                    relative_path = os.path.relpath(root, INPUT_PATH)
                    if relative_path == '.':
                        relative_path = ''

                    process_file(input_file_path, OUTPUT_DIR, relative_path, file)
                    file_count += 1

    print(f"\n处理完成。共处理了 {file_count} 个 CSV 文件。")
    print(f"输出结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()