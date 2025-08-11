"""
ML Plotter Logger 集成使用示例
展示如何使用集成的实验日志记录和可视化功能
"""

import numpy as np
import time
from ml_plotter_logger import MLPlotterLogger, create_integrated_logger, quick_experiment_plot

def simulate_training_run(config: dict, steps: int = 1000) -> dict:
    """模拟一个训练过程"""
    results = {'steps': [], 'episodic_return': [], 'actor_loss': [], 'critic_loss': []}
    
    # 根据配置设置基础性能
    base_reward = 100
    if config.get('algorithm') == 'TD3':
        base_reward += 50
    elif config.get('algorithm') == 'SAC':
        base_reward += 30
    
    if config.get('lr') == 0.001:
        base_reward += 20
    elif config.get('lr') == 0.0001:
        base_reward -= 10
    
    # 模拟训练过程
    for step in range(0, steps, 50):
        # 奖励随时间增长，但有噪声
        reward = base_reward + step * 0.1 + np.random.normal(0, 10)
        
        # 损失随时间减少
        actor_loss = 2.0 * np.exp(-step/300) + np.random.normal(0, 0.1)
        critic_loss = 5.0 * np.exp(-step/200) + np.random.normal(0, 0.2)
        
        results['steps'].append(step)
        results['episodic_return'].append(max(0, reward))  # 确保非负
        results['actor_loss'].append(max(0, actor_loss))
        results['critic_loss'].append(max(0, critic_loss))
    
    return results

def example_1_basic_logging():
    """示例1: 基本的日志记录和可视化"""
    print("\n" + "="*60)
    print("示例1: 基本的日志记录和可视化")
    print("="*60)
    
    # 创建集成日志记录器
    logger = create_integrated_logger("demo_logs")
    
    # 配置实验
    experiment_id = "basic_rl_experiment"
    configs = [
        {'algorithm': 'TD3', 'lr': 0.001, 'seed': 1},
        {'algorithm': 'TD3', 'lr': 0.001, 'seed': 2},
        {'algorithm': 'SAC', 'lr': 0.001, 'seed': 1},
        {'algorithm': 'SAC', 'lr': 0.001, 'seed': 2},
    ]
    
    # 运行实验
    for i, config in enumerate(configs):
        run_id = f"run_{i}"
        
        # 开始运行
        logger.start_run(experiment_id, run_id, config, 
                        tags=[f"algo:{config['algorithm']}", f"lr:{config['lr']}"])
        
        # 模拟训练
        results = simulate_training_run(config)
        
        # 记录数据
        for j, step in enumerate(results['steps']):
            logger.log(run_id, step, {
                'episodic_return': results['episodic_return'][j],
                'actor_loss': results['actor_loss'][j],
                'critic_loss': results['critic_loss'][j]
            })
        
        # 结束运行
        logger.finish_run(run_id)
    
    # 生成可视化
    print("\n📊 生成可视化图表...")
    
    # 按算法分组绘制训练曲线
    fig1 = logger.plot_training_curves(
        experiment_id, 'episodic_return', 
        group_by='algorithm',
        title="Training Performance by Algorithm"
    )
    
    # 绘制损失曲线
    fig2 = logger.plot_training_curves(
        experiment_id, 'actor_loss',
        group_by='algorithm', 
        title="Actor Loss by Algorithm"
    )
    
    # 绘制性能对比柱状图
    fig3 = logger.plot_comparison_bars(
        experiment_id, 'episodic_return',
        group_by='algorithm',
        title="Final Performance Comparison"
    )
    
    print("✅ 基本示例完成！")

def example_2_auto_plotting():
    """示例2: 自动绘图功能"""
    print("\n" + "="*60)
    print("示例2: 自动绘图功能")
    print("="*60)
    
    logger = create_integrated_logger("demo_logs")
    
    # 配置更复杂的实验
    experiment_id = "hyperparameter_sweep"
    configs = [
        {'algorithm': 'TD3', 'lr': 0.001, 'batch_size': 256, 'seed': 1},
        {'algorithm': 'TD3', 'lr': 0.001, 'batch_size': 256, 'seed': 2},
        {'algorithm': 'TD3', 'lr': 0.0001, 'batch_size': 256, 'seed': 1},
        {'algorithm': 'TD3', 'lr': 0.0001, 'batch_size': 256, 'seed': 2},
        {'algorithm': 'TD3', 'lr': 0.001, 'batch_size': 128, 'seed': 1},
        {'algorithm': 'TD3', 'lr': 0.001, 'batch_size': 128, 'seed': 2},
    ]
    
    # 运行实验
    for i, config in enumerate(configs):
        run_id = f"sweep_run_{i}"
        
        logger.start_run(experiment_id, run_id, config)
        
        # 模拟训练
        results = simulate_training_run(config, steps=800)
        
        # 记录数据
        for j, step in enumerate(results['steps']):
            logger.log(run_id, step, {
                'episodic_return': results['episodic_return'][j],
                'actor_loss': results['actor_loss'][j],
                'critic_loss': results['critic_loss'][j]
            })
        
        logger.finish_run(run_id)
    
    # 自动生成所有图表
    print("\n🎨 自动生成所有图表...")
    figures = logger.auto_plot_experiment(
        experiment_id,
        group_by='lr',  # 按学习率分组
        plot_types=['curves', 'bars', 'summary']
    )
    
    print(f"✅ 自动生成了 {len(figures)} 个图表")

def example_3_experiment_comparison():
    """示例3: 实验对比"""
    print("\n" + "="*60)
    print("示例3: 实验对比")
    print("="*60)
    
    logger = create_integrated_logger("demo_logs")
    
    # 创建两个不同的实验
    experiments = {
        'td3_experiment': [
            {'algorithm': 'TD3', 'lr': 0.001, 'seed': 1},
            {'algorithm': 'TD3', 'lr': 0.001, 'seed': 2},
            {'algorithm': 'TD3', 'lr': 0.001, 'seed': 3},
        ],
        'sac_experiment': [
            {'algorithm': 'SAC', 'lr': 0.001, 'seed': 1},
            {'algorithm': 'SAC', 'lr': 0.001, 'seed': 2},
            {'algorithm': 'SAC', 'lr': 0.001, 'seed': 3},
        ]
    }
    
    # 运行两个实验
    for exp_id, configs in experiments.items():
        print(f"\n🧪 运行实验: {exp_id}")
        
        for i, config in enumerate(configs):
            run_id = f"{exp_id}_run_{i}"
            
            logger.start_run(exp_id, run_id, config)
            
            # 模拟训练
            results = simulate_training_run(config)
            
            # 记录数据
            for j, step in enumerate(results['steps']):
                logger.log(run_id, step, {
                    'episodic_return': results['episodic_return'][j],
                    'actor_loss': results['actor_loss'][j]
                })
            
            logger.finish_run(run_id)
    
    # 比较两个实验
    print("\n🔄 比较实验...")
    fig_comparison = logger.compare_experiments(
        ['td3_experiment', 'sac_experiment'],
        'episodic_return',
        title="TD3 vs SAC Performance Comparison"
    )
    
    print("✅ 实验对比完成！")

def example_4_quick_plot():
    """示例4: 快速绘图"""
    print("\n" + "="*60)
    print("示例4: 快速绘图功能")
    print("="*60)
    
    # 使用便捷函数快速生成图表
    figures = quick_experiment_plot(
        'basic_rl_experiment',  # 使用示例1的实验
        log_dir="demo_logs",
        group_by='algorithm',
        plot_types=['curves', 'bars']
    )
    
    print(f"✅ 快速生成了 {len(figures)} 个图表")

def main():
    """运行所有示例"""
    print("🎨 ML Plotter Logger 集成示例")
    print("结合实验日志记录和专业可视化")
    
    try:
        # 运行所有示例
        example_1_basic_logging()
        example_2_auto_plotting()
        example_3_experiment_comparison()
        example_4_quick_plot()
        
        print("\n" + "="*60)
        print("🎉 所有示例运行完成！")
        print("📁 查看生成的文件:")
        print("   - 日志数据: demo_logs/experiments.db")
        print("   - 图表文件: demo_logs/plots/")
        print("   - 摘要报告: demo_logs/plots/*_summary.md")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()