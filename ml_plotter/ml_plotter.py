"""
ML Plotter - 主要绘图类
提供简洁的API接口用于生成专业的机器学习实验图表
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import warnings

from .styles import style_manager, FIGURE_SIZES, LINE_CONFIG, ALPHA_CONFIG, FONT_SIZES
from .data_utils import data_processor

class MLPlotter:
    """机器学习实验结果绘图器"""
    
    def __init__(self, style: str = "academic", smooth_window: int = 500):
        """
        初始化绘图器
        
        Args:
            style: 绘图风格，默认为"academic"
            smooth_window: 数据平滑窗口大小
        """
        self.style = style
        self.smooth_window = smooth_window
        data_processor.smooth_window = smooth_window
        
        # 应用默认样式
        style_manager.setup_matplotlib_defaults()
    
    def plot_training_curves(self, 
                           data_paths: Union[str, List[str], Dict[str, str]],
                           labels: Optional[List[str]] = None,
                           title: str = "",
                           xlabel: str = "Steps",
                           ylabel: str = "Episode Return",
                           smooth: bool = True,
                           show_std: bool = True,
                           max_steps: Optional[float] = None,
                           save_path: Optional[str] = None,
                           figsize: Tuple[float, float] = FIGURE_SIZES["default"],
                           legend_loc: str = 'upper left',
                           legend_bbox: Optional[Tuple[float, float]] = None) -> plt.Figure:
        """
        绘制训练曲线对比图
        
        Args:
            data_paths: 数据路径，可以是单个路径、路径列表或路径字典
            labels: 图例标签列表
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            smooth: 是否平滑数据
            show_std: 是否显示标准差阴影
            max_steps: 最大步数限制
            save_path: 保存路径
            figsize: 图表尺寸
            legend_loc: 图例位置，如'upper left', 'upper right', 'best'等
            legend_bbox: 图例精确位置坐标(x, y)，如(0.02, 0.98)
        
        Returns:
            matplotlib Figure对象
        """
        # 处理输入参数
        if isinstance(data_paths, str):
            # 单个路径，自动发现条件
            conditions = data_processor.auto_discover_conditions(data_paths)
            data_dict = conditions
        elif isinstance(data_paths, list):
            # 路径列表
            data_dict = {os.path.basename(path): path for path in data_paths}
        elif isinstance(data_paths, dict):
            # 路径字典
            data_dict = data_paths
        else:
            raise ValueError("data_paths must be str, list, or dict")
        
        if not data_dict:
            raise ValueError("No valid data paths found")
        
        # 处理标签
        if labels is None:
            labels = list(data_dict.keys())
        elif len(labels) != len(data_dict):
            warnings.warn("Labels length doesn't match data paths, using auto-generated labels")
            labels = list(data_dict.keys())
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制每个条件的曲线
        for i, (condition, path) in enumerate(data_dict.items()):
            label = labels[i] if i < len(labels) else condition
            
            # 加载数据
            data = data_processor.load_folder_data(path)
            if data is None:
                print(f"Warning: Could not load data from {path}")
                continue
            
            steps = data['steps']
            means = data['means']
            stds = data['stds']
            
            # 应用步数限制
            if max_steps is not None:
                mask = steps <= max_steps
                steps = steps[mask]
                means = means[mask]
                stds = stds[mask]
            
            if len(steps) == 0:
                continue
            
            # 数据平滑
            if smooth:
                means = data_processor.smooth_data(means, method='ema')
                stds = data_processor.smooth_data(stds, method='ema')
            
            # 获取颜色和线型
            color = style_manager.get_color(label)
            linestyle = style_manager.get_linestyle(label)
            
            # 绘制主线
            ax.plot(steps, means, label=label, color=color, linestyle=linestyle,
                   linewidth=LINE_CONFIG["width"])
            
            # 绘制标准差阴影
            if show_std and np.any(stds > 0):
                ax.fill_between(steps, means - stds, means + stds,
                              color=color, alpha=ALPHA_CONFIG["shade"])
        
        # 应用样式
        style_manager.apply_academic_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
        
        # 设置图例
        if legend_bbox is not None:
            # 使用精确坐标定位
            ax.legend(fontsize=FONT_SIZES["legend"], loc=legend_loc, 
                     bbox_to_anchor=legend_bbox)
        else:
            # 使用预设位置
            ax.legend(fontsize=FONT_SIZES["legend"], loc=legend_loc)
        
        # 设置科学计数法
        style_manager.setup_scientific_notation(ax)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_performance_bars(self,
                            data_paths: Union[List[str], Dict[str, str]],
                            labels: Optional[List[str]] = None,
                            title: str = "",
                            xlabel: str = "Methods",
                            ylabel: str = "Max Episode Return",
                            save_path: Optional[str] = None,
                            figsize: Tuple[float, float] = FIGURE_SIZES["default"],
                            show_legend: bool = False,
                            legend_loc: str = 'upper right') -> plt.Figure:
        """
        绘制性能对比柱状图
        
        Args:
            data_paths: 数据路径列表或字典
            labels: 方法标签列表
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            save_path: 保存路径
            figsize: 图表尺寸
            show_legend: 是否显示图例（柱状图通常不需要图例）
            legend_loc: 图例位置
        
        Returns:
            matplotlib Figure对象
        """
        # 处理输入参数
        if isinstance(data_paths, list):
            data_dict = {os.path.basename(path): path for path in data_paths}
        elif isinstance(data_paths, dict):
            data_dict = data_paths
        else:
            raise ValueError("data_paths must be list or dict")
        
        if labels is None:
            labels = list(data_dict.keys())
        
        # 提取最大分数
        means = []
        stds = []
        valid_labels = []
        
        for i, (condition, path) in enumerate(data_dict.items()):
            label = labels[i] if i < len(labels) else condition
            max_scores = data_processor.extract_max_scores(path)
            
            if max_scores:
                means.append(np.mean(max_scores))
                stds.append(np.std(max_scores))
                valid_labels.append(label)
            else:
                print(f"Warning: No valid scores found for {condition}")
        
        if not means:
            raise ValueError("No valid data found for plotting")
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制柱状图
        x_pos = np.arange(len(valid_labels))
        colors = [style_manager.get_color(label) for label in valid_labels]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=colors, alpha=0.8, 
                     error_kw={'ecolor': 'black', 'elinewidth': 1})
        
        # 设置标签和标题
        ax.set_xlabel(xlabel, fontsize=FONT_SIZES["label"])
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES["label"], fontweight='bold')
        if title:
            ax.set_title(title, fontsize=FONT_SIZES["title"])
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_labels)
        
        # 添加图例（如果需要）
        if show_legend:
            ax.legend(valid_labels, fontsize=FONT_SIZES["legend"], loc=legend_loc)
        
        # 应用网格和样式
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_box_comparison(self,
                          data_paths: Union[List[str], Dict[str, str]],
                          labels: Optional[List[str]] = None,
                          title: str = "",
                          xlabel: str = "Max Episode Return",
                          save_path: Optional[str] = None,
                          figsize: Tuple[float, float] = FIGURE_SIZES["default"]) -> plt.Figure:
        """
        绘制箱线图对比
        
        Args:
            data_paths: 数据路径列表或字典
            labels: 方法标签列表
            title: 图表标题
            xlabel: X轴标签
            save_path: 保存路径
            figsize: 图表尺寸
        
        Returns:
            matplotlib Figure对象
        """
        # 处理输入参数
        if isinstance(data_paths, list):
            data_dict = {os.path.basename(path): path for path in data_paths}
        elif isinstance(data_paths, dict):
            data_dict = data_paths
        else:
            raise ValueError("data_paths must be list or dict")
        
        if labels is None:
            labels = list(data_dict.keys())
        
        # 收集所有分数数据
        all_scores = []
        valid_labels = []
        
        for i, (condition, path) in enumerate(data_dict.items()):
            label = labels[i] if i < len(labels) else condition
            max_scores = data_processor.extract_max_scores(path)
            
            if max_scores:
                all_scores.append(max_scores)
                valid_labels.append(label)
            else:
                print(f"Warning: No valid scores found for {condition}")
        
        if not all_scores:
            raise ValueError("No valid data found for plotting")
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制箱线图
        colors = [style_manager.get_color(label) for label in valid_labels]
        
        bp = ax.boxplot(all_scores, vert=False, patch_artist=True,
                       showmeans=True, meanline=True, showfliers=True,
                       labels=valid_labels, widths=0.6)
        
        # 设置颜色
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
        
        # 设置标签和标题
        ax.set_xlabel(xlabel, fontsize=FONT_SIZES["label"])
        if title:
            ax.set_title(title, fontsize=FONT_SIZES["title"])
        
        # 应用网格和样式
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def quick_plot(self, data_path: str, **kwargs) -> plt.Figure:
        """
        快速绘图 - 自动选择最合适的图表类型
        
        Args:
            data_path: 数据路径
            **kwargs: 其他绘图参数
        
        Returns:
            matplotlib Figure对象
        """
        # 自动发现条件
        conditions = data_processor.auto_discover_conditions(data_path)
        
        if len(conditions) == 1:
            # 单条件，绘制单一训练曲线
            return self.plot_training_curves(data_path, **kwargs)
        elif len(conditions) > 1:
            # 多条件，绘制对比图
            return self.plot_training_curves(conditions, **kwargs)
        else:
            raise ValueError(f"No valid conditions found in {data_path}")
    
    def save_all_formats(self, fig: plt.Figure, base_path: str):
        """
        保存图表为多种格式
        
        Args:
            fig: matplotlib Figure对象
            base_path: 基础保存路径（不含扩展名）
        """
        formats = ['png', 'pdf', 'svg']
        for fmt in formats:
            save_path = f"{base_path}.{fmt}"
            fig.savefig(save_path, dpi=300, bbox_inches='tight', format=fmt)
            print(f"Saved {save_path}")

# 便捷函数
def quick_plot(data_path: str, **kwargs) -> plt.Figure:
    """快速绘图函数"""
    plotter = MLPlotter()
    return plotter.quick_plot(data_path, **kwargs)

def plot_comparison(data_paths: Union[List[str], Dict[str, str]], 
                   plot_type: str = "curves", **kwargs) -> plt.Figure:
    """绘制对比图"""
    plotter = MLPlotter()
    
    if plot_type == "curves":
        return plotter.plot_training_curves(data_paths, **kwargs)
    elif plot_type == "bars":
        return plotter.plot_performance_bars(data_paths, **kwargs)
    elif plot_type == "box":
        return plotter.plot_box_comparison(data_paths, **kwargs)
    else:
        raise ValueError("plot_type must be 'curves', 'bars', or 'box'")