"""
ML Plotter - 简化的机器学习实验结果可视化库

这个库提供了简洁易用的API来生成专业的学术风格图表，
保持与原画图程序一致的视觉风格，但大大简化了使用方式。

主要特性：
- 🎨 专业的学术风格图表
- 📊 自动数据处理和格式识别  
- 🔧 简洁的API接口
- 📈 支持多种图表类型
- 🎯 批量处理能力

基本用法：
    from ml_plotter import MLPlotter, quick_plot
    
    # 最简单的用法
    quick_plot("experiment_folder")
    
    # 更多控制
    plotter = MLPlotter()
    fig = plotter.plot_training_curves(
        data_paths=["vanilla", "redo", "regrama"],
        labels=["Vanilla", "ReDo", "ReGraMa"],
        title="Training Performance Comparison"
    )
"""

from .ml_plotter import MLPlotter, quick_plot, plot_comparison
from .styles import style_manager, COLOR_MAP, FONT_SIZES
from .data_utils import data_processor

__version__ = "1.0.0"
__author__ = "ML Plotter Team"

# 导出主要接口
__all__ = [
    'MLPlotter',
    'quick_plot', 
    'plot_comparison',
    'style_manager',
    'data_processor',
    'COLOR_MAP',
    'FONT_SIZES'
]

# 设置默认配置
import matplotlib.pyplot as plt
plt.style.use('default')  # 确保使用默认样式作为基础