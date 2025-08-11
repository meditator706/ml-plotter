# ML Plotter - 简化的机器学习实验可视化库

一个专门为机器学习/强化学习实验结果设计的简洁画图库，保留专业学术风格，但大大简化使用方式。

## ✨ 特性

- 🎨 **专业学术风格** - 符合顶级会议/期刊标准的图表质量
- 📊 **自动数据处理** - 智能识别CSV格式，自动清洗和对齐数据
- 🔧 **简洁API** - 几行代码生成复杂图表
- 📈 **多种图表类型** - 训练曲线、柱状图、箱线图等
- 🎯 **批量处理** - 自动发现和处理多个实验条件
- 🌈 **一致的视觉风格** - 预设的颜色映射和样式配置

## 🚀 快速开始

### 安装依赖

```bash
pip install matplotlib pandas numpy scipy
```

### 最简单的用法

```python
from ml_plotter import quick_plot

# 一行代码生成专业图表
quick_plot("experiment_folder")
```

### 更多控制

```python
from ml_plotter import MLPlotter

plotter = MLPlotter()

# 训练曲线对比 - 带图例位置控制
fig = plotter.plot_training_curves(
    data_paths=["vanilla", "redo", "regrama"],
    labels=["Vanilla", "ReDo", "ReGraMa"],
    title="Training Performance Comparison",
    legend_loc='upper right',  # 图例位置
    legend_bbox=(0.98, 0.98)   # 精确坐标（可选）
)

# 性能柱状图 - 带图例控制
fig = plotter.plot_performance_bars(
    data_paths=["method1", "method2", "method3"],
    title="Final Performance Comparison",
    show_legend=True,          # 是否显示图例
    legend_loc='upper right'   # 图例位置
)

# 箱线图分布对比
fig = plotter.plot_box_comparison(
    data_paths=["method1", "method2", "method3"],
    title="Performance Distribution"
)
```

## 📊 支持的图表类型

### 1. 训练曲线 (Training Curves)
- 自动处理多个种子的数据
- 支持数据平滑和标准差阴影
- 智能颜色和线型映射

### 2. 性能柱状图 (Performance Bars)  
- 显示最终性能对比
- 包含误差棒显示标准差
- 支持自定义颜色

### 3. 箱线图对比 (Box Plots)
- 展示性能分布情况
- 自动检测异常值
- 水平布局便于标签显示

## 🎨 视觉风格

库保持与原画图程序一致的专业风格：

- **颜色映射**: Vanilla(灰色), ReDo(紫红色), ReGraMa(橙色)
- **字体大小**: 标题16pt, 标签15pt, 图例12pt
- **线条样式**: 主线6pt宽度，虚线表示剪枝版本
- **网格样式**: 浅灰色网格，3pt边框宽度

## 📁 数据格式

支持标准的CSV格式，自动识别常见列名：

**步数列**: Step, global_step, _step, total_steps, Timestep...
**性能列**: episode_return, episodic_return, avg_reward, Value...

示例CSV格式：
```csv
Step,env_name: task - episode_return
0,1000.5
1000,1050.2
2000,1100.8
...
```

## 🔧 高级用法

### 图例位置控制

```python
from ml_plotter import MLPlotter

plotter = MLPlotter()

# 基本图例位置控制
fig = plotter.plot_training_curves(
    data_paths=data_paths,
    legend_loc='upper right'  # 'upper left', 'upper right', 'lower left', 
                             # 'lower right', 'best', 'center'
)

# 精确图例位置控制
fig = plotter.plot_training_curves(
    data_paths=data_paths,
    legend_loc='upper left',
    legend_bbox=(0.02, 0.98)  # (x, y) 精确坐标
)

# 柱状图图例控制
fig = plotter.plot_performance_bars(
    data_paths=data_paths,
    show_legend=True,         # 是否显示图例
    legend_loc='upper right'  # 图例位置
)
```

### 自定义样式

```python
from ml_plotter import MLPlotter, style_manager

# 修改颜色映射
style_manager.COLOR_MAP["my_method"] = "blue"

# 创建自定义绘图器
plotter = MLPlotter(smooth_window=300)
```

### 批量保存多种格式

```python
fig = plotter.plot_training_curves(data_paths)
plotter.save_all_formats(fig, "my_plot")  # 保存为PNG, PDF, SVG
```

### 便捷函数

```python
from ml_plotter import plot_comparison

# 快速对比不同类型的图表（支持图例控制）
plot_comparison(data_paths, plot_type="curves", legend_loc='best')
plot_comparison(data_paths, plot_type="bars", show_legend=True) 
plot_comparison(data_paths, plot_type="box")
```

## 📖 完整示例

查看 `example.py` 文件获取完整的使用示例，包括：

1. 快速绘图
2. 自定义训练曲线
3. 性能对比柱状图
4. 箱线图分布对比
5. 便捷函数使用

运行示例：
```bash
python example.py
```

## 🤝 与原程序的对比

| 特性 | 原程序 | ML Plotter |
|------|--------|------------|
| 代码行数 | 1000+ | 10-20 |
| 配置复杂度 | 高 | 低 |
| 学习成本 | 高 | 低 |
| 视觉质量 | 专业 | 专业 |
| 功能完整性 | 完整 | 核心功能 |

## 📄 许可证

MIT License

## 🙏 致谢

基于原有的专业画图程序简化而来，保持了相同的视觉风格和质量标准。