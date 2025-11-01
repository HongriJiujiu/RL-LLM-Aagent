import glob
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------
# 绘图风格设置
# -----------------------------
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'mathtext.fontset': 'stix',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10
})

line_colors = sns.color_palette("colorblind", 4)
line_styles = cycle(["-", "-.", "--", ":"])
sns.set_palette(line_colors)
line_colors = cycle(line_colors)

# -----------------------------
# 工具函数
# -----------------------------
def moving_average(data, window_size):
    """计算移动平均"""
    if window_size <= 1:
        return data
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, "same")


def plot_metric(df, x_col, y_col, color, ma_window=1, label=""):
    """绘制单条曲线及平滑曲线"""
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    mean_values = df.groupby(x_col).mean()[y_col]
    std_values = df.groupby(x_col).std()[y_col]

    if ma_window > 1:
        mean_values = moving_average(mean_values, ma_window)
        std_values = moving_average(std_values, ma_window)

    x_values = df.groupby(x_col)[x_col].mean().keys().values

    # 原始曲线
    plt.plot(x_values, mean_values, label=label, color=color, linestyle=next(line_styles))
    # 平滑曲线
    mean_smooth = moving_average(mean_values, ma_window)
    plt.plot(x_values, mean_smooth, label=label + ' (smoothed)',
             color=color, linestyle=next(line_styles))


# -----------------------------
# 主程序配置（直接定义）
# -----------------------------
# CSV 文件路径，可以使用 glob 模式
csv_file_patterns = [
    rf"固定信号配时\固定信号配时_conn0_ep1.csv",
    rf"LLM0\alpha0.1_gamma0.99_eps0.05_decay1.0_rewardwait_s3600_p1_conn0_ep1.csv"
]

# 对应图例标签
legend_labels = ["Baseline", "Strategy one"]

# 横纵轴设置
x_column = "step"
y_column = "system_mean_waiting_time"
x_label = "Time step (seconds)"
# Mean waiting time (s)系统平均等待时间
# 219_average_waiting_time 219号信号灯平均等待时间
y_label = "Mean waiting time (s)"
plot_title = "Comparison of Traffic Signal Strategies"
moving_avg_window = 20
max_steps = 3600

# 输出图片路径，如果不保存可以置为 None
output_file = rf"results\固定配时与强化学习对比\固定配时与强化学习对比.png"

# -----------------------------
# 绘图
# -----------------------------
plt.figure()
legend_cycle = cycle(legend_labels)

for file_pattern in csv_file_patterns:
    combined_df = pd.DataFrame()
    for csv_file in glob.glob(file_pattern + "*"):
        df = pd.read_csv(csv_file).iloc[:int(max_steps / 5), :]
        combined_df = pd.concat((combined_df, df)) if not combined_df.empty else df

    plot_metric(combined_df,
                x_col=x_column,
                y_col=y_column,
                color=next(line_colors),
                ma_window=moving_avg_window,
                label=next(legend_cycle))

plt.title(plot_title)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.ylim(bottom=0)
plt.legend(loc="upper right", frameon=False, fontsize=12)

if output_file:
    plt.savefig(output_file, bbox_inches="tight", dpi=600)

plt.show()
