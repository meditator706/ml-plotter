# Simple Logger - 极简实验记录和可视化

## 🎯 设计理念

像 wandb 一样简单，但专注于本地实验管理和专业图表生成。

## 🚀 核心API

只需要4个函数就能完成所有操作：

```python
import simple_logger as logger

# 1. 开始实验
logger.init(project="my_project", config={"lr": 0.001})

# 2. 记录指标
logger.log({"loss": 0.5, "accuracy": 0.9})

# 3. 生成图表
logger.plot(project="my_project", metric="loss")

# 4. 查看摘要
logger.summary("my_project")
```

## 📊 完整使用示例

### 基本用法

```python
import simple_logger as logger

# 开始实验 - 像 wandb.init()
logger.init(
    project="my_experiment",
    name="baseline_run", 
    config={"lr": 0.001, "batch_size": 32}
)

# 训练循环
for epoch in range(100):
    # ... 训练代码 ...
    
    # 记录指标 - 像 wandb.log()
    logger.log({
        "loss": train_loss,
        "accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

# 结束实验
logger.finish()
```

### 超参数扫描

```python
# 多个实验配置
configs = [
    {"lr": 0.001, "optimizer": "Adam"},
    {"lr": 0.01, "optimizer": "Adam"}, 
    {"lr": 0.001, "optimizer": "SGD"},
]

for i, config in enumerate(configs):
    # 开始新运行
    logger.init(
        project="hyperparameter_sweep",
        name=f"run_{i}",
        config=config
    )
    
    # 训练...
    for step in range(1000):
        logger.log({"loss": loss, "accuracy": acc})
    
    logger.finish()

# 一键生成所有可视化
logger.plot("hyperparameter_sweep", "loss", group_by="optimizer")
logger.plot("hyperparameter_sweep", "accuracy", group_by="lr")
logger.summary("hyperparameter_sweep", group_by="optimizer")
```

## 🔄 与其他方案对比

### vs wandb

| 特性 | wandb | Simple Logger |
|------|-------|---------------|
| **网络依赖** | 需要联网 | **完全本地** |
| **数据隐私** | 上传到云端 | **本地存储** |
| **图表质量** | 标准 | **学术级专业** |
| **学习成本** | 中等 | **极低** |
| **API复杂度** | 较复杂 | **4个函数** |

### vs 原ML Plotter

| 操作 | 原ML Plotter | Simple Logger |
|------|-------------|---------------|
| **记录数据** | 手动保存CSV | `logger.log()` |
| **开始实验** | 创建文件夹 | `logger.init()` |
| **生成图表** | 复杂配置 | `logger.plot()` |
| **查看结果** | 手动分析 | `logger.summary()` |
| **代码行数** | 20+ 行 | **4 行** |

### vs 复杂集成版本

| 特性 | 复杂版本 | Simple Logger |
|------|----------|---------------|
| **类和方法** | 10+ 个类/方法 | **4 个函数** |
| **配置复杂度** | 高 | **零配置** |
| **学习时间** | 30分钟+ | **5分钟** |
| **文件大小** | 19KB | **8KB** |

## 📈 使用场景

### 1. 日常实验

```python
# 开始实验
logger.init("daily_experiments", config={"model": "ResNet50"})

# 训练循环中
logger.log({"loss": loss, "accuracy": acc})

# 立即查看结果
logger.plot("daily_experiments", "loss")
```

### 2. 论文实验

```python
# 多个基线对比
baselines = ["Vanilla", "ReDo", "ReGraMa"]

for baseline in baselines:
    logger.init("paper_experiments", name=baseline, 
                config={"method": baseline})
    # 训练...
    logger.finish()

# 生成论文图表
logger.plot("paper_experiments", "reward", group_by="method")
```

### 3. 超参数调优

```python
# 网格搜索
for lr in [0.1, 0.01, 0.001]:
    for bs in [16, 32, 64]:
        logger.init("hyperparameter_tuning", 
                   config={"lr": lr, "batch_size": bs})
        # 训练...
        logger.finish()

# 找到最佳参数
logger.summary("hyperparameter_tuning", group_by="lr")
```

## 🎨 图表特性

- **专业学术风格** - 符合顶级会议标准
- **自动分组** - 按参数智能分组
- **置信区间** - 多运行自动显示误差
- **高分辨率** - 300 DPI，适合论文
- **一致配色** - 预设的专业配色方案

## 📁 文件结构

```
logs/
├── my_project/
│   ├── experiments.db      # 实验数据
│   └── plots/             # 生成的图表
│       ├── loss.png
│       └── accuracy_by_lr.png
└── another_project/
    ├── experiments.db
    └── plots/
```

## 🔧 高级用法

### 指定特定运行

```python
# 只绘制特定运行
logger.plot("my_project", "loss", runs=["run_1", "run_3"])
```

### 自定义图表

```python
# 自定义标题和保存
fig = logger.plot("my_project", "accuracy", 
                 title="Custom Title", save=False)
fig.show()
```

### 批量分析

```python
# 分析所有项目
projects = ["exp1", "exp2", "exp3"]
for project in projects:
    logger.summary(project)
```

## 💡 最佳实践

1. **项目命名** - 使用描述性名称：`"resnet_cifar10"` 而不是 `"exp1"`
2. **配置记录** - 记录所有重要超参数
3. **指标命名** - 使用标准名称：`"loss"`, `"accuracy"`, `"val_loss"`
4. **定期可视化** - 实验结束后立即生成图表
5. **分组分析** - 使用 `group_by` 参数进行对比分析

## 🎯 总结

Simple Logger 提供了：

- ✅ **极简API** - 只需4个函数
- ✅ **零配置** - 开箱即用
- ✅ **专业图表** - 学术级质量
- ✅ **本地存储** - 数据隐私安全
- ✅ **智能分组** - 自动参数对比
- ✅ **快速上手** - 5分钟学会

完美平衡了简单性和功能性！