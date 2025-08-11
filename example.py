"""
ML Plotter 使用示例
展示如何用几行代码生成专业的机器学习实验图表
"""

import os
import numpy as np
import pandas as pd
from ml_plotter import MLPlotter, quick_plot, plot_comparison
import matplotlib.pyplot as plt

def create_sample_data():
    """创建示例数据用于演示"""
    print("创建示例数据...")
    
    # 创建示例数据文件夹
    methods = ["Vanilla", "ReDo", "ReGraMa"]
    base_dir = "sample_data"
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for method in methods:
        method_dir = os.path.join(base_dir, method)
        if not os.path.exists(method_dir):
            os.makedirs(method_dir)
        
        # 为每个方法创建3个种子的数据
        for seed in range(3):
            # 生成模拟的训练数据
            steps = np.arange(0, 1000000, 1000)
            
            # 不同方法有不同的性能特征
            if method == "Vanilla":
                base_performance = 1000 + 500 * np.log(steps + 1) / np.log(1000000)
                noise_scale = 100
            elif method == "ReDo":
                base_performance = 1200 + 600 * np.log(steps + 1) / np.log(1000000)
                noise_scale = 80
            else:  # ReGraMa
                base_performance = 1400 + 700 * np.log(steps + 1) / np.log(1000000)
                noise_scale = 60
            
            # 添加噪声
            np.random.seed(seed + hash(method) % 1000)
            noise = np.random.normal(0, noise_scale, len(steps))
            episode_return = base_performance + noise
            
            # 确保性能不为负
            episode_return = np.maximum(episode_return, 0)
            
            # 保存为CSV
            df = pd.DataFrame({
                'Step': steps,
                f'{method.lower()} - episode_return': episode_return
            })
            
            csv_path = os.path.join(method_dir, f"seed_{seed}.csv")
            df.to_csv(csv_path, index=False)
    
    print(f"示例数据已创建在 {base_dir} 文件夹中")
    return base_dir

def example_1_quick_plot():
    """示例1: 最简单的快速绘图"""
    print("\n=== 示例1: 快速绘图 ===")
    
    # 创建示例数据
    data_dir = create_sample_data()
    
    # 一行代码生成专业图表
    fig = quick_plot(data_dir, title="Quick Plot Example", show_std=True)
    plt.savefig("example_1_quick_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图表已保存为 example_1_quick_plot.png")

def example_2_custom_training_curves():
    """示例2: 自定义训练曲线"""
    print("\n=== 示例2: 自定义训练曲线 ===")
    
    data_dir = create_sample_data()
    
    # 创建绘图器
    plotter = MLPlotter(smooth_window=300)
    
    # 指定具体的数据路径和标签
    data_paths = {
        "Vanilla Agent": os.path.join(data_dir, "Vanilla"),
        "ReDo (Ours)": os.path.join(data_dir, "ReDo"), 
        "ReGraMa": os.path.join(data_dir, "ReGraMa")
    }
    
    fig = plotter.plot_training_curves(
        data_paths=data_paths,
        title="Training Performance Comparison",
        xlabel="Training Steps",
        ylabel="Episode Return",
        smooth=True,
        show_std=True,
        max_steps=800000,
        figsize=(8, 5)
    )
    
    plt.savefig("example_2_training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图表已保存为 example_2_training_curves.png")

def example_3_performance_bars():
    """示例3: 性能对比柱状图"""
    print("\n=== 示例3: 性能对比柱状图 ===")
    
    data_dir = create_sample_data()
    
    plotter = MLPlotter()
    
    data_paths = [
        os.path.join(data_dir, "Vanilla"),
        os.path.join(data_dir, "ReDo"),
        os.path.join(data_dir, "ReGraMa")
    ]
    
    labels = ["Vanilla", "ReDo", "ReGraMa (Ours)"]
    
    fig = plotter.plot_performance_bars(
        data_paths=data_paths,
        labels=labels,
        title="Final Performance Comparison",
        ylabel="Max Episode Return",
        figsize=(8, 5)
    )
    
    plt.savefig("example_3_performance_bars.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图表已保存为 example_3_performance_bars.png")

def example_4_box_comparison():
    """示例4: 箱线图对比"""
    print("\n=== 示例4: 箱线图对比 ===")
    
    data_dir = create_sample_data()
    
    plotter = MLPlotter()
    
    data_paths = {
        "Vanilla": os.path.join(data_dir, "Vanilla"),
        "ReDo": os.path.join(data_dir, "ReDo"),
        "ReGraMa": os.path.join(data_dir, "ReGraMa")
    }
    
    fig = plotter.plot_box_comparison(
        data_paths=data_paths,
        title="Performance Distribution Comparison",
        xlabel="Max Episode Return",
        figsize=(8, 5)
    )
    
    plt.savefig("example_4_box_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图表已保存为 example_4_box_comparison.png")

def example_5_convenience_functions():
    """示例5: 便捷函数使用"""
    print("\n=== 示例5: 便捷函数 ===")
    
    data_dir = create_sample_data()
    
    data_paths = [
        os.path.join(data_dir, "Vanilla"),
        os.path.join(data_dir, "ReDo"),
        os.path.join(data_dir, "ReGraMa")
    ]
    
    # 使用便捷函数绘制不同类型的图表
    
    # 训练曲线
    fig1 = plot_comparison(data_paths, plot_type="curves", 
                          title="Training Curves", show_std=False)
    plt.savefig("example_5_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 柱状图
    fig2 = plot_comparison(data_paths, plot_type="bars",
                          title="Performance Bars")
    plt.savefig("example_5_bars.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 箱线图
    fig3 = plot_comparison(data_paths, plot_type="box",
                          title="Performance Distribution")
    plt.savefig("example_5_box.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("所有图表已保存")

def main():
    """运行所有示例"""
    print("ML Plotter 使用示例")
    print("=" * 50)
    
    try:
        # 运行所有示例
        example_1_quick_plot()
        example_2_custom_training_curves()
        example_3_performance_bars()
        example_4_box_comparison()
        example_5_convenience_functions()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("生成的图表文件：")
        for i in range(1, 6):
            if i == 5:
                print(f"  - example_5_curves.png")
                print(f"  - example_5_bars.png")
                print(f"  - example_5_box.png")
            else:
                print(f"  - example_{i}_*.png")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()