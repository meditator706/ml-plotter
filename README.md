<div align="center">

# ML Plotter

*A simplified machine learning experiment visualization library*

[![English](https://img.shields.io/badge/Language-English-blue)](#english) [![中文](https://img.shields.io/badge/语言-中文-red)](#中文)

</div>

---

## English

A simplified machine learning experiment visualization library that maintains professional academic style while dramatically simplifying usage.

### ✨ Features

- 🎨 **Professional Academic Style** - Publication-quality charts for top-tier conferences/journals
- 📊 **Automatic Data Processing** - Smart CSV format recognition and data cleaning
- 🔧 **Simple API** - Generate complex charts with just a few lines of code
- 📈 **Multiple Chart Types** - Training curves, bar charts, box plots
- 🎯 **Batch Processing** - Auto-discover and process multiple experimental conditions
- 🌈 **Consistent Visual Style** - Predefined color mappings and style configurations
- 🗃️ **Logger Integration** - Seamless integration with experiment logging systems

### 🚀 Quick Start

#### Installation

```bash
pip install matplotlib pandas numpy scipy
```

#### Simple Logger (Recommended) ⭐

```python
import simple_logger as logger

# Start experiment - like wandb.init()
logger.init(project="my_experiment", config={"lr": 0.001})

# Log metrics - like wandb.log()
for step in range(1000):
    logger.log({"loss": loss, "accuracy": acc})

# Generate plots - one line
logger.plot("my_experiment", "loss", group_by="lr")
logger.summary("my_experiment")
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

Run the complete examples:

```bash
# Basic plotting examples
python example.py

# Simple logger examples (Recommended)
python example_simple.py

# Advanced logger integration
python example_integrated_logger.py
```

### 📄 License

MIT License

---

## 中文

专门为机器学习/强化学习实验结果设计的简洁画图库，保留专业学术风格，但大大简化使用方式。

### ✨ 特性

- 🎨 **专业学术风格** - 符合顶级会议/期刊标准的图表质量
- 📊 **自动数据处理** - 智能识别CSV格式，自动清洗和对齐数据
- 🔧 **简洁API** - 几行代码生成复杂图表
- 📈 **多种图表类型** - 训练曲线、柱状图、箱线图
- 🎯 **批量处理** - 自动发现和处理多个实验条件
- 🌈 **一致的视觉风格** - 预设的颜色映射和样式配置
- 🗃️ **日志集成** - 与实验日志记录系统无缝集成

### 🚀 快速开始

#### 安装依赖

```bash
pip install matplotlib pandas numpy scipy
```

#### 极简日志记录 (推荐) ⭐

```python
import simple_logger as logger

# 开始实验 - 像 wandb.init()
logger.init(project="my_experiment", config={"lr": 0.001})

# 记录指标 - 像 wandb.log()
for step in range(1000):
    logger.log({"loss": loss, "accuracy": acc})

# 生成图表 - 一行代码
logger.plot("my_experiment", "loss", group_by="lr")
logger.summary("my_experiment")
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
# 基本绘图示例
python example.py

# 极简日志示例 (推荐)
python example_simple.py

# 高级日志集成
python example_integrated_logger.py
```

### 📄 许可证

MIT License