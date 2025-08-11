"""
Simple Logger 使用示例
展示如何用类似wandb的简洁接口记录和可视化实验
"""

import numpy as np
import simple_logger as logger

def simulate_training(config):
    """模拟训练过程"""
    lr = config.get('lr', 0.001)
    algorithm = config.get('algorithm', 'SGD')
    
    # 根据配置设置基础性能
    base_loss = 2.0
    if algorithm == 'Adam':
        base_loss = 1.5
    elif algorithm == 'AdamW':
        base_loss = 1.2
    
    if lr == 0.01:
        base_loss += 0.5
    elif lr == 0.0001:
        base_loss -= 0.2
    
    # 模拟训练过程
    for step in range(100):
        # 损失随时间减少
        loss = base_loss * np.exp(-step/50) + np.random.normal(0, 0.1)
        
        # 准确率随时间增加
        accuracy = 1 - loss/3 + np.random.normal(0, 0.02)
        accuracy = max(0, min(1, accuracy))  # 限制在[0,1]
        
        # 记录指标 - 就像wandb.log()一样简单！
        logger.log({
            'loss': max(0, loss),
            'accuracy': accuracy,
            'learning_rate': lr
        })

def example_1_basic_usage():
    """示例1: 基本用法 - 像wandb一样简单"""
    print("\n" + "="*50)
    print("示例1: 基本用法")
    print("="*50)
    
    # 开始实验 - 像 wandb.init()
    logger.init(
        project="simple_ml_project",
        name="baseline_run",
        config={"lr": 0.001, "algorithm": "Adam", "batch_size": 32}
    )
    
    # 模拟训练
    simulate_training({"lr": 0.001, "algorithm": "Adam"})
    
    # 结束实验
    logger.finish()
    
    print("✅ 基本用法完成！")

def example_2_multiple_runs():
    """示例2: 多个运行对比"""
    print("\n" + "="*50)
    print("示例2: 多个运行对比")
    print("="*50)
    
    # 不同配置的实验
    configs = [
        {"lr": 0.001, "algorithm": "Adam"},
        {"lr": 0.001, "algorithm": "AdamW"},
        {"lr": 0.01, "algorithm": "Adam"},
        {"lr": 0.0001, "algorithm": "Adam"},
    ]
    
    for i, config in enumerate(configs):
        # 开始新的运行
        logger.init(
            project="optimizer_comparison",
            name=f"{config['algorithm']}_lr{config['lr']}",
            config=config
        )
        
        # 训练
        simulate_training(config)
        
        # 结束运行
        logger.finish()
        
        print(f"✅ 完成运行 {i+1}/{len(configs)}")
    
    print("✅ 多运行实验完成！")

def example_3_visualization():
    """示例3: 可视化结果"""
    print("\n" + "="*50)
    print("示例3: 可视化结果")
    print("="*50)
    
    # 绘制不同指标
    print("📊 生成可视化图表...")
    
    # 绘制损失曲线 - 按算法分组
    fig1 = logger.plot(
        project="optimizer_comparison",
        metric="loss",
        group_by="algorithm",
        title="Loss Comparison by Algorithm"
    )
    
    # 绘制准确率曲线 - 按学习率分组
    fig2 = logger.plot(
        project="optimizer_comparison", 
        metric="accuracy",
        group_by="lr",
        title="Accuracy Comparison by Learning Rate"
    )
    
    # 显示项目摘要
    print("\n📋 项目摘要:")
    logger.summary("optimizer_comparison", group_by="algorithm")
    
    print("✅ 可视化完成！")

def example_4_hyperparameter_sweep():
    """示例4: 超参数扫描"""
    print("\n" + "="*50)
    print("示例4: 超参数扫描")
    print("="*50)
    
    # 超参数网格
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    algorithms = ["Adam", "AdamW", "SGD"]
    
    run_count = 0
    for lr in learning_rates:
        for algo in algorithms:
            run_count += 1
            
            # 开始运行
            logger.init(
                project="hyperparameter_sweep",
                name=f"sweep_run_{run_count}",
                config={
                    "lr": lr,
                    "algorithm": algo,
                    "batch_size": 64,
                    "epochs": 100
                }
            )
            
            # 训练
            simulate_training({"lr": lr, "algorithm": algo})
            
            # 结束运行
            logger.finish()
    
    print(f"✅ 完成 {run_count} 个超参数组合！")
    
    # 可视化结果
    print("\n📊 生成超参数扫描结果...")
    
    # 按学习率分组
    logger.plot(
        project="hyperparameter_sweep",
        metric="loss", 
        group_by="lr",
        title="Hyperparameter Sweep: Loss by Learning Rate"
    )
    
    # 按算法分组
    logger.plot(
        project="hyperparameter_sweep",
        metric="accuracy",
        group_by="algorithm", 
        title="Hyperparameter Sweep: Accuracy by Algorithm"
    )
    
    # 显示最佳结果
    print("\n🏆 最佳结果:")
    logger.summary("hyperparameter_sweep", group_by="lr")

def example_5_quick_comparison():
    """示例5: 快速对比不同项目"""
    print("\n" + "="*50)
    print("示例5: 快速项目对比")
    print("="*50)
    
    # 对比不同项目的结果
    projects = ["simple_ml_project", "optimizer_comparison", "hyperparameter_sweep"]
    
    for project in projects:
        try:
            print(f"\n📊 项目: {project}")
            logger.summary(project)
        except FileNotFoundError:
            print(f"   项目 {project} 不存在，跳过...")
    
    print("✅ 项目对比完成！")

def main():
    """运行所有示例"""
    print("🎯 Simple Logger 使用示例")
    print("类似wandb的极简接口")
    
    try:
        # 运行所有示例
        example_1_basic_usage()
        example_2_multiple_runs()
        example_3_visualization()
        example_4_hyperparameter_sweep()
        example_5_quick_comparison()
        
        print("\n" + "="*50)
        print("🎉 所有示例运行完成！")
        print("\n📁 查看生成的文件:")
        print("   - 实验数据: logs/*/experiments.db")
        print("   - 图表文件: logs/*/plots/")
        print("\n💡 使用方法:")
        print("   1. logger.init() - 开始实验")
        print("   2. logger.log() - 记录指标")
        print("   3. logger.plot() - 生成图表")
        print("   4. logger.summary() - 查看摘要")
        print("="*50)
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()