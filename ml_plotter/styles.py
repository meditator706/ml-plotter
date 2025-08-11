"""
ML Plotter - 样式配置模块
保持与原画图程序一致的专业学术风格
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, Tuple

# 颜色映射 - 与原程序保持一致
COLOR_MAP: Dict[str, str] = {
    "vanilla": "grey",
    "standard": "grey", 
    "regrama": "darkorange",
    "regrama pruning": "darkorange",
    "redo": "mediumvioletred",
    "redo pruning": "mediumvioletred",
    "kaiming normal": "darkorange",
    "redo prune": "mediumvioletred",
    "gate reset": "thistle",
    "default": "grey"
}

# 线型映射
LINESTYLE_MAP: Dict[str, str] = {
    "grama pruning": "--",
    "redo pruning": "--", 
    "regrama pruning": "--",
    "default": "-"
}

# 字体大小配置
FONT_SIZES = {
    "title": 16,
    "label": 15,
    "legend": 12,
    "tick": 12,
    "aggregate_title": 30,
    "aggregate_label": 25,
    "aggregate_legend": 30
}

# 线条和标记配置
LINE_CONFIG = {
    "width": 6,          # 线宽 - 与draw_as.py一致
    "marker_size": 20,   # 标记大小
    "border_width": 3.0  # 边框宽度
}

# 透明度配置
ALPHA_CONFIG = {
    "shade": 0.1,        # 阴影透明度
    "grid": 1.0,         # 网格透明度
    "error_bar": 0.6     # 误差棒透明度
}

# 图表尺寸配置
FIGURE_SIZES = {
    "default": (6, 4),   # 与draw_as.py一致
    "large": (10, 6),
    "wide": (12, 4)
}

class StyleManager:
    """样式管理器 - 统一管理所有样式设置"""
    
    def __init__(self):
        self.setup_matplotlib_defaults()
    
    def setup_matplotlib_defaults(self):
        """设置matplotlib默认样式"""
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['font.size'] = FONT_SIZES["tick"]
    
    def get_color(self, label: str) -> str:
        """获取标签对应的颜色"""
        label_lower = label.lower()
        return COLOR_MAP.get(label_lower, COLOR_MAP["default"])
    
    def get_linestyle(self, label: str) -> str:
        """获取标签对应的线型"""
        label_lower = label.lower()
        return LINESTYLE_MAP.get(label_lower, LINESTYLE_MAP["default"])
    
    def apply_academic_style(self, ax, title: str = "", 
                           xlabel: str = "Steps", ylabel: str = "Performance",
                           grid: bool = True):
        """应用学术风格样式"""
        # 设置标题和标签
        if title:
            ax.set_title(title, fontsize=FONT_SIZES["title"])
        ax.set_xlabel(xlabel, fontsize=FONT_SIZES["label"])
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES["label"], fontweight='bold')
        
        # 设置网格
        if grid:
            ax.grid(True, color='lightgray', linestyle='-', linewidth=1.8)
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(LINE_CONFIG["border_width"])
            spine.set_color('black')
        
        # 设置刻度
        ax.tick_params(axis='both', which='major', 
                      labelsize=FONT_SIZES["tick"], direction='in')
        
        # 设置坐标轴范围
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    
    def apply_aggregate_style(self, ax, title: str = "",
                            xlabel: str = "Steps (in millions)", 
                            ylabel: str = "Normalized Score"):
        """应用聚合图表的样式"""
        if title:
            ax.set_title(title, fontsize=FONT_SIZES["aggregate_title"], pad=20)
        
        ax.set_xlabel(xlabel, fontsize=FONT_SIZES["aggregate_label"], labelpad=15)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES["aggregate_label"], labelpad=15)
        
        # 网格设置
        ax.grid(True, linestyle='-', alpha=ALPHA_CONFIG["grid"], color='gray')
        
        # 移除顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        
        # 刻度设置
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
    
    def setup_scientific_notation(self, ax):
        """设置科学计数法格式"""
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 4))
        ax.xaxis.set_major_formatter(formatter)
    
    def get_default_colors(self, n_colors: int) -> list:
        """获取默认颜色列表"""
        if n_colors <= len(COLOR_MAP):
            return list(COLOR_MAP.values())[:n_colors]
        else:
            # 如果需要更多颜色，使用matplotlib默认颜色循环
            colors = list(COLOR_MAP.values())
            default_colors = plt.cm.tab10.colors
            for i in range(n_colors - len(colors)):
                colors.append(default_colors[i % len(default_colors)])
            return colors

# 全局样式管理器实例
style_manager = StyleManager()