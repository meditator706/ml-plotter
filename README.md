# ML Plotter | 简化的机器学习实验可视化库

[English](#english) | [中文](#中文)

---

## English

A simplified machine learning experiment visualization library that maintains professional academic style while dramatically simplifying usage.

### ✨ Features

- 🎨 **Professional Academic Style** - Publication-quality charts for top-tier conferences/journals
- 📊 **Automatic Data Processing** - Smart CSV format recognition and data cleaning
- 🔧 **Simple API** - Generate complex charts with just a few lines of code
- 📈 **Multiple Chart Types** - Training curves, bar charts, box plots, etc.
- 🎯 **Batch Processing** - Auto-discover and process multiple experimental conditions
- 🌈 **Consistent Visual Style** - Predefined color mappings and style configurations

### 🚀 Quick Start

#### Installation

```bash
pip install matplotlib pandas numpy scipy
```

#### Basic Usage

```python
from ml_plotter import quick_plot

# Generate professional charts with one line
quick_plot("experiment_folder")
```

#### Advanced Usage

```python
from ml_plotter import MLPlotter

plotter = MLPlotter()

# Training curves with legend control
fig = plotter.plot_training_curves(
    data_paths=["vanilla", "redo", "regrama"],
    labels=["Vanilla", "ReDo", "ReGraMa"],
    title="Training Performance Comparison",
    legend_loc='upper right'
)

# Performance bar chart
fig = plotter.plot_performance_bars(
    data_paths=["method1", "method2", "method3"],
    title="Final Performance Comparison"
)
```

### 📊 Supported Chart Types

- **Training Curves**: Multi-seed data with confidence intervals
- **Performance Bars**: Final performance comparison with error bars  
- **Box Plots**: Performance distribution analysis

### 📁 Data Format

Supports standard CSV format with automatic column recognition:

```csv
Step,env_name: task - episode_return
0,1000.5
1000,1050.2
2000,1100.8
...
```

### 🔧 Advanced Features

#### Legend Position Control

```python
# Basic position control
fig = plotter.plot_training_curves(
    data_paths=data_paths,
    legend_loc='upper right'  # 'upper left', 'lower right', 'best', etc.
)

# Precise positioning
fig = plotter.plot_training_curves(
    data_paths=data_paths,
    legend_loc='upper left',
    legend_bbox=(0.02, 0.98)  # (x, y) coordinates
)
```

#### Custom Styling

```python
from ml_plotter import style_manager

# Add custom colors
style_manager.COLOR_MAP["my_method"] = "blue"

# Custom plotter
plotter = MLPlotter(smooth_window=300)
```

### 📖 Examples

Run the complete example:

```bash
python example.py
```

---

## 中文

专门为机器学习/强化学习实验结果设计的简洁画图库，保留专业学术风格，但大大简化使用方式。

### ✨ 特性

- 🎨 **专业学术风格** - 符合顶级会议/期刊标准的图表质量
- 📊 **自动数据处理** - 智能识别CSV格式，自动清洗和对齐数据
- 🔧 **简洁API** - 几行代码生成复杂图表
- 📈 **多种图表类型** - 训练曲线、柱状图、箱线图等
- 🎯 **批量处理** - 自动发现和处理多个实验条件
- 🌈 **一致的视觉风格** - 预设的颜色映射和样式配置

### 🚀 快速开始

#### 安装依赖

```bash
pip install matplotlib pandas numpy scipy
```

#### 基本用法

```python
from ml_plotter import quick_plot

# 一行代码生成专业图表
quick_plot("experiment_folder")
```

#### 高级用法

```python
from ml_plotter import MLPlotter

plotter = MLPlotter()

# 训练曲线对比 - 带图例位置控制
fig = plotter.plot_training_curves(
    data_paths=["vanilla", "redo", "regrama"],
    labels=["Vanilla", "ReDo", "ReGraMa"],
    title="Training Performance Comparison",
    legend_loc='upper right'
)

# 性能柱状图
fig = plotter.plot_performance_bars(
    data_paths=["method1", "method2", "method3"],
    title="Final Performance Comparison"
)
```

### 📊 支持的图表类型

- **训练曲线**: 多种子数据，带置信区间
- **性能柱状图**: 最终性能对比，含误差棒
- **箱线图**: 性能分布分析

### 📁 数据格式

支持标准CSV格式，自动识别列名：

```csv
Step,env_name: task - episode_return
0,1000.5
1000,1050.2
2000,1100.8
...
```

### 🔧 高级功能

#### 图例位置控制

```python
# 基本位置控制
fig = plotter.plot_training_curves(
    data_paths=data_paths,
    legend_loc='upper right'  # 'upper left', 'lower right', 'best' 等
)

# 精确定位
fig = plotter.plot_training_curves(
    data_paths=data_paths,
    legend_loc='upper left',
    legend_bbox=(0.02, 0.98)  # (x, y) 坐标
)
```

#### 自定义样式

```python
from ml_plotter import style_manager

# 添加自定义颜色
style_manager.COLOR_MAP["my_method"] = "blue"

# 自定义绘图器
plotter = MLPlotter(smooth_window=300)
```

### 📖 完整示例

运行完整示例：

```bash
python example.py
```

---

## 📄 License

MIT License

## 🙏 Acknowledgments

Based on professional plotting programs, maintaining the same visual style and quality standards.