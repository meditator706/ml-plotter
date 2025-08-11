# ML Plotter 使用指南

## 🚀 快速开始

### 1. 基本安装和设置

确保安装了必要的依赖：
```bash
pip install matplotlib pandas numpy scipy
```

### 2. 最简单的使用方式

```python
from ml_plotter import quick_plot

# 假设你有一个包含多个实验条件的文件夹
# experiment_folder/
#   ├── Vanilla/
#   │   ├── seed_0.csv
#   │   ├── seed_1.csv
#   │   └── seed_2.csv
#   ├── ReDo/
#   │   ├── seed_0.csv
#   │   └── seed_1.csv
#   └── ReGraMa/
#       ├── seed_0.csv
#       └── seed_1.csv

# 一行代码生成专业图表
quick_plot("experiment_folder")
```

### 3. 更多控制的使用方式

```python
from ml_plotter import MLPlotter

# 创建绘图器
plotter = MLPlotter(smooth_window=500)

# 训练曲线对比
fig = plotter.plot_training_curves(
    data_paths={
        "Vanilla Agent": "experiment_folder/Vanilla",
        "ReDo (Ours)": "experiment_folder/ReDo", 
        "ReGraMa": "experiment_folder/ReGraMa"
    },
    title="Training Performance Comparison",
    xlabel="Training Steps",
    ylabel="Episode Return",
    smooth=True,
    show_std=True,
    max_steps=1000000,
    save_path="training_curves.png"
)
```

## 📊 支持的数据格式

### CSV文件格式要求

你的CSV文件应该包含以下列：

**步数列（自动识别）**：
- `Step`, `step`, `global_step`, `_step`
- `TotalSteps`, `total_steps`, `Timestep`, `timestep`
- `Epoch`, `epoch`, `Iteration`, `iteration`

**性能列（自动识别）**：
- `- episode_return` (最常见)
- `charts/episodic_return`
- `eval/avg_reward`
- `rollout/ep_rew_mean`
- `Value`

### 示例CSV格式

```csv
Step,env_name: HalfCheetah-v4 - episode_return
0,1000.5
1000,1050.2
2000,1100.8
3000,1150.1
...
```

## 🎨 图表类型详解

### 1. 训练曲线 (Training Curves)

```python
plotter = MLPlotter()

fig = plotter.plot_training_curves(
    data_paths=["method1", "method2", "method3"],
    labels=["Method 1", "Method 2", "Method 3"],
    title="Training Performance",
    smooth=True,        # 是否平滑数据
    show_std=True,      # 是否显示标准差阴影
    max_steps=500000    # 最大步数限制
)
```

**特点**：
- 自动处理多个种子的数据
- 支持EMA平滑
- 智能颜色映射
- 标准差阴影显示

### 2. 性能柱状图 (Performance Bars)

```python
fig = plotter.plot_performance_bars(
    data_paths=["method1", "method2", "method3"],
    labels=["Method 1", "Method 2", "Method 3"],
    title="Final Performance Comparison",
    ylabel="Max Episode Return"
)
```

**特点**：
- 显示每个方法的最大性能
- 包含误差棒（标准差）
- 自动颜色映射

### 3. 箱线图对比 (Box Plots)

```python
fig = plotter.plot_box_comparison(
    data_paths=["method1", "method2", "method3"],
    labels=["Method 1", "Method 2", "Method 3"],
    title="Performance Distribution",
    xlabel="Max Episode Return"
)
```

**特点**：
- 显示性能分布
- 自动检测异常值
- 水平布局便于标签显示

## 🎨 样式自定义

### 颜色映射

```python
from ml_plotter.styles import style_manager

# 查看当前颜色映射
print(style_manager.COLOR_MAP)

# 添加自定义颜色
style_manager.COLOR_MAP["my_method"] = "blue"
```

### 字体大小

```python
from ml_plotter.styles import FONT_SIZES

# 查看当前字体设置
print(FONT_SIZES)

# 修改字体大小
FONT_SIZES["title"] = 18
FONT_SIZES["label"] = 16
```

## 🔧 高级功能

### 便捷函数

```python
from ml_plotter import plot_comparison

# 快速生成不同类型的图表
plot_comparison(data_paths, plot_type="curves")  # 训练曲线
plot_comparison(data_paths, plot_type="bars")    # 柱状图
plot_comparison(data_paths, plot_type="box")     # 箱线图
```

### 批量保存多种格式

```python
fig = plotter.plot_training_curves(data_paths)
plotter.save_all_formats(fig, "my_plot")  # 保存为PNG, PDF, SVG
```

### 数据预处理

```python
from ml_plotter.data_utils import data_processor

# 自定义平滑窗口
data_processor.smooth_window = 300

# 手动加载和处理数据
data = data_processor.load_folder_data("experiment_folder/Method1")
print(f"加载了 {data['n_runs']} 个种子的数据")
```

## 🐛 常见问题

### 1. 找不到合适的列名

**问题**：`Warning: Could not find suitable columns`

**解决**：检查你的CSV文件列名，确保包含步数列和性能列。可以手动指定：

```python
# 在data_utils.py中添加你的列名
X_COLUMN_CANDIDATES.append("your_step_column")
Y_COLUMN_CANDIDATES.append("your_performance_column")
```

### 2. 数据为空

**问题**：`Warning: No valid data loaded`

**解决**：
- 检查文件路径是否正确
- 确保CSV文件包含数值数据
- 检查是否有NaN值

### 3. 图表样式不符合预期

**解决**：
- 检查标签名称是否匹配预设的颜色映射
- 使用自定义颜色映射
- 调整字体大小和图表尺寸

## 📝 完整示例

查看项目中的 `example.py` 文件，包含了所有功能的完整演示。

运行示例：
```bash
python example.py
```

这将生成多个示例图表，展示库的各种功能。