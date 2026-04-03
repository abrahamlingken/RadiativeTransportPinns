#!/usr/bin/env python
"""
学术论文级训练曲线绘制 (Optimized for Paper Publication)
用法: python plot_training.py <结果目录>
示例: python plot_training.py Results_1D_Section32

输出格式:
- training_curves_paper.pdf/png/eps (四宫格综合分析)
- training_curve_single.pdf/png/tiff (单幅主图，推荐用于论文)
"""

import sys
import json
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

try:
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def smooth_curve(data, window=5):
    """平滑曲线"""
    if not HAS_SCIPY or len(data) < window:
        return data
    return uniform_filter1d(data, size=window, mode='nearest')

# 配置字体
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False

# 学术配色方案
COLORS = {
    'primary': '#0066CC',
    'secondary': '#CC3300',
    'accent': '#009933',
    'neutral': '#666666',
    'light_gray': '#CCCCCC',
    'grid': '#E5E5E5',
}

DPI = 600

def cm_to_inch(cm):
    return cm / 2.54

def create_paper_figure(epochs, loss, times, output_folder):
    """创建学术论文级四宫格图"""
    
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': DPI,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.2,
        'lines.markersize': 3,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(cm_to_inch(17), cm_to_inch(12)))
    fig.patch.set_facecolor('white')
    
    final_loss = loss[-1]
    initial_loss = loss[0]
    loss_smooth = smooth_curve(loss, window=3)
    
    # 图1: 线性尺度
    ax = axes[0, 0]
    ax.fill_between(epochs, loss, alpha=0.15, color=COLORS['primary'])
    ax.plot(epochs, loss, color=COLORS['primary'], linewidth=1.5, label='Training Loss', alpha=0.9)
    ax.plot(epochs, loss_smooth, color=COLORS['secondary'], linewidth=1.0, linestyle='--', 
            alpha=0.6, label='Smoothed')
    ax.annotate(f'$L_{{final}} = {final_loss:.2e}$',
                xy=(epochs[-1], final_loss),
                xytext=(epochs[-1] * 0.5, final_loss + initial_loss * 0.3),
                fontsize=7, color=COLORS['secondary'],
                arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], lw=0.5))
    ax.set_xlabel('Iteration $n$', fontweight='medium')
    ax.set_ylabel(r'Loss $\mathcal{L}(\theta)$', fontweight='medium')
    ax.set_title('(a) Linear Scale', loc='left', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4, color=COLORS['grid'])
    ax.legend(loc='upper right', framealpha=0.95, edgecolor=COLORS['light_gray'])
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # 图2: 对数尺度
    ax = axes[0, 1]
    ax.semilogy(epochs, loss, color=COLORS['primary'], linewidth=1.5, alpha=0.9)
    ax.fill_between(epochs, loss, alpha=0.1, color=COLORS['primary'])
    ax.axhline(y=final_loss, color=COLORS['secondary'], linestyle=':', linewidth=1.0, 
               alpha=0.7, label=f'$\\epsilon = {final_loss:.2e}$')
    ax.set_xlabel('Iteration $n$', fontweight='medium')
    ax.set_ylabel(r'Loss $\mathcal{L}(\theta)$ (log)', fontweight='medium')
    ax.set_title('(b) Logarithmic Scale', loc='left', fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.4, color=COLORS['grid'])
    ax.legend(loc='upper right', framealpha=0.95, edgecolor=COLORS['light_gray'])
    ax.set_xlim(left=0)
    
    # 图3: 训练效率
    ax = axes[1, 0]
    ax.semilogy(times, loss, color=COLORS['accent'], linewidth=1.5, alpha=0.9)
    time_50_idx = int(len(times) * 0.5)
    ax.scatter([times[time_50_idx], times[-1]], [loss[time_50_idx], final_loss], 
               color=[COLORS['secondary'], COLORS['primary']], s=30, zorder=5, 
               marker='o', edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Time $t$ (s)', fontweight='medium')
    ax.set_ylabel(r'Loss $\mathcal{L}(\theta)$', fontweight='medium')
    ax.set_title('(c) Time Efficiency', loc='left', fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.4, color=COLORS['grid'])
    ax.set_xlim(left=0)
    
    # 图4: 收敛分析
    ax = axes[1, 1]
    log_loss = np.log10(loss)
    if len(log_loss) > 1:
        d_log = np.diff(log_loss)
        ax.plot(epochs[1:], d_log, color=COLORS['secondary'], linewidth=0.8, 
                alpha=0.6, label='Instantaneous')
        d_log_smooth = smooth_curve(d_log, window=11)
        ax.plot(epochs[1:], d_log_smooth, color=COLORS['primary'], linewidth=1.5,
                label='Smoothed', zorder=3)
        ax.axhline(y=0, color=COLORS['neutral'], linestyle='--', alpha=0.5, linewidth=0.8)
        ax.fill_between(epochs[1:], d_log_smooth, 0, where=(d_log_smooth < 0), 
                       alpha=0.15, color=COLORS['accent'])
        ax.set_xlabel('Iteration $n$', fontweight='medium')
        ax.set_ylabel(r'$\Delta \log_{10} \mathcal{L}$', fontweight='medium')
        ax.set_title('(d) Convergence Rate', loc='left', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4, color=COLORS['grid'])
        ax.legend(loc='upper right', framealpha=0.95, edgecolor=COLORS['light_gray'])
        ax.set_xlim(left=0)
    
    plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.8)
    
    base_path = os.path.join(output_folder, 'training_curves_paper')
    plt.savefig(f'{base_path}.pdf', format='pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(f'{base_path}.png', format='png', bbox_inches='tight', facecolor='white')
    plt.savefig(f'{base_path}.eps', format='eps', bbox_inches='tight', facecolor='white')
    
    print(f"[1/2] Paper-quality 4-panel figure saved:")
    print(f"      - {base_path}.pdf (Vector)")
    print(f"      - {base_path}.png ({DPI} DPI)")
    
    return fig

def create_single_figure(epochs, loss, output_folder):
    """创建单幅高质量图（用于论文主图）"""
    
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': DPI,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.8,
    })
    
    fig, ax = plt.subplots(figsize=(cm_to_inch(15), cm_to_inch(10)))
    fig.patch.set_facecolor('white')
    
    initial_loss = loss[0]
    final_loss = loss[-1]
    
    # 主曲线带渐变填充
    ax.semilogy(epochs, loss, color=COLORS['primary'], linewidth=2.0, 
               label='Training Loss', zorder=3)
    ax.fill_between(epochs, loss, alpha=0.12, color=COLORS['primary'])
    ax.axhline(y=final_loss, color=COLORS['secondary'], linestyle='--', 
               linewidth=1.2, alpha=0.7, zorder=2)
    
    # 标注关键点
    ax.scatter([epochs[0], epochs[-1]], [initial_loss, final_loss],
               color=[COLORS['primary'], COLORS['secondary']],
               s=[60, 80], zorder=5, marker='o', edgecolors='white', linewidths=1.5)
    
    # 数值标注
    ax.annotate(f'$\\mathcal{{L}}_0 = {initial_loss:.2e}$',
                xy=(epochs[0], initial_loss),
                xytext=(epochs[-1] * 0.05, initial_loss * 1.5),
                fontsize=8, color=COLORS['primary'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=COLORS['primary'], alpha=0.9, linewidth=0.5))
    
    ax.annotate(f'$\\mathcal{{L}}_{{final}} = {final_loss:.2e}$',
                xy=(epochs[-1], final_loss),
                xytext=(epochs[-1] * 0.6, final_loss * 3),
                fontsize=8, color=COLORS['secondary'],
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=COLORS['secondary'], alpha=0.9, linewidth=0.5))
    
    # 统计信息框
    reduction = initial_loss / final_loss
    textstr = f'Reduction: ${reduction:.1f}\\times$\nIterations: {len(epochs)}'
    props = dict(boxstyle='round,pad=0.5', facecolor='#FFF8DC', 
                edgecolor=COLORS['neutral'], alpha=0.95, linewidth=0.8)
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=7,
           verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.set_xlabel('Iteration $n$', fontweight='medium')
    ax.set_ylabel(r'Loss $\mathcal{L}(\theta)$', fontweight='medium')
    ax.grid(True, which='both', linestyle='--', alpha=0.4, color=COLORS['grid'])
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    
    base_path = os.path.join(output_folder, 'training_curve_single')
    plt.savefig(f'{base_path}.pdf', format='pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(f'{base_path}.png', format='png', bbox_inches='tight', facecolor='white')
    plt.savefig(f'{base_path}.tiff', format='tiff', bbox_inches='tight', facecolor='white', dpi=DPI)
    
    print(f"[2/2] Single-panel figure saved:")
    print(f"      - {base_path}.pdf (Main figure)")
    print(f"      - {base_path}.tiff ({DPI} DPI)")
    
    return fig

def main():
    if len(sys.argv) < 2:
        folder = 'Results_1D_Section32'
    else:
        folder = sys.argv[1]
    
    history_file = os.path.join(folder, 'training_history.json')
    
    if not os.path.exists(history_file):
        print(f"Error: Training history not found at {history_file}")
        sys.exit(1)
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = np.array(history['epochs'])
    total_loss = np.array(history['total_loss'])
    times = np.array(history['time'])
    
    # 数据修复
    if len(epochs) > 1 and epochs[-1] == 0:
        if epochs[-2] > 0:
            epochs[-1] = epochs[-2] + (epochs[1] - epochs[0])
    
    # 对数形式转换
    if np.mean(total_loss) < 0:
        print("Detected log-scale loss values, converting to linear scale...")
        loss_display = 10 ** total_loss
    else:
        loss_display = total_loss
    
    print("="*60)
    print("Generating Paper-Quality Training Curves")
    print("="*60)
    
    create_paper_figure(epochs, loss_display, times, folder)
    create_single_figure(epochs, loss_display, folder)
    
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Data points: {len(loss_display)}")
    print(f"Time: {times[-1]:.1f}s ({times[-1]/60:.1f}min)")
    print(f"Loss: {loss_display[0]:.2e} → {loss_display[-1]:.2e}")
    print(f"Reduction: {loss_display[0]/loss_display[-1]:.1f}×")
    print("="*60)

if __name__ == "__main__":
    main()
