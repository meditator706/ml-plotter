"""
ML Plotter Logger Integration - 实验日志记录与可视化集成
结合 ExperimentLogger 和 ML Plotter，实现一体化的实验管理和可视化
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import matplotlib.pyplot as plt

# 导入现有的模块
from experiment_logger import ExperimentLogger, get_logger
from ml_plotter import MLPlotter, style_manager

class MLPlotterLogger:
    """集成的实验日志记录和可视化器"""
    
    def __init__(self, log_dir: str = "experiment_logs"):
        self.logger = ExperimentLogger(log_dir)
        self.plotter = MLPlotter()
        self.log_dir = Path(log_dir)
        
        # 创建可视化输出目录
        self.plots_dir = self.log_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        print(f"🎨 ML Plotter Logger 初始化完成")
        print(f"   日志目录: {self.log_dir}")
        print(f"   图表目录: {self.plots_dir}")
    
    def start_run(self, experiment_id: str, run_id: str, params: Dict[str, Any], tags: List[str] = None):
        """开始实验运行"""
        return self.logger.start_run(experiment_id, run_id, params, tags)
    
    def log(self, run_id: str, step: int, metrics: Dict[str, float]):
        """记录实验指标"""
        return self.logger.log(run_id, step, metrics)
    
    def finish_run(self, run_id: str):
        """结束实验运行"""
        return self.logger.finish_run(run_id)
    
    def export_to_csv(self, experiment_id: str, output_dir: Optional[str] = None) -> str:
        """
        将实验数据导出为CSV格式，兼容ML Plotter
        
        Args:
            experiment_id: 实验ID
            output_dir: 输出目录，默认为 log_dir/csv_exports
            
        Returns:
            导出目录路径
        """
        if output_dir is None:
            output_dir = self.log_dir / "csv_exports" / experiment_id
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有运行
        runs = self.logger.get_runs(experiment_id)
        
        print(f"📤 导出实验数据: {experiment_id}")
        print(f"   发现 {len(runs)} 个运行")
        
        for run_info in runs:
            run_id = run_info['run_id']
            
            # 从数据库获取该运行的所有数据
            conn = sqlite3.connect(self.logger.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT step, metric_name, metric_value 
                FROM logs 
                WHERE experiment_id = ? AND run_id = ?
                ORDER BY step, metric_name
            ''', (experiment_id, run_id))
            
            # 转换为DataFrame
            data = {}
            for row in cursor.fetchall():
                step, metric_name, metric_value = row
                if step not in data:
                    data[step] = {'Step': step}
                data[step][f'{experiment_id} - {metric_name}'] = metric_value
            
            conn.close()
            
            if data:
                # 创建DataFrame并保存
                df = pd.DataFrame(list(data.values()))
                df = df.sort_values('Step').reset_index(drop=True)
                
                csv_path = output_dir / f"{run_id}.csv"
                df.to_csv(csv_path, index=False)
                print(f"   ✅ {run_id} -> {csv_path}")
        
        print(f"📁 CSV文件已导出到: {output_dir}")
        return str(output_dir)
    
    def plot_training_curves(self, 
                           experiment_id: str,
                           metric_name: str = 'episodic_return',
                           group_by: Optional[Union[str, List[str]]] = None,
                           title: Optional[str] = None,
                           save_plot: bool = True,
                           **plot_kwargs) -> plt.Figure:
        """
        直接从日志数据绘制训练曲线
        
        Args:
            experiment_id: 实验ID
            metric_name: 要绘制的指标名称
            group_by: 分组参数
            title: 图表标题
            save_plot: 是否保存图表
            **plot_kwargs: 传递给MLPlotter的其他参数
            
        Returns:
            matplotlib Figure对象
        """
        # 获取指标数据
        metrics_data = self.logger.get_metrics(experiment_id, metric_name, group_by)
        
        if not metrics_data:
            raise ValueError(f"未找到实验 {experiment_id} 的指标 {metric_name}")
        
        print(f"📊 绘制训练曲线: {experiment_id}/{metric_name}")
        print(f"   发现 {len(metrics_data)} 个分组")
        
        # 准备数据
        data_dict = {}
        for group_key, data in metrics_data.items():
            # 创建临时CSV数据
            temp_data = pd.DataFrame({
                'Step': data['steps'],
                f'{experiment_id} - {metric_name}': data['values']
            })
            data_dict[group_key] = temp_data
        
        # 设置默认标题
        if title is None:
            if group_by:
                if isinstance(group_by, str):
                    title = f"{experiment_id}: {metric_name} (grouped by {group_by})"
                else:
                    title = f"{experiment_id}: {metric_name} (grouped by {', '.join(group_by)})"
            else:
                title = f"{experiment_id}: {metric_name}"
        
        # 使用MLPlotter绘制
        fig = self._plot_from_data_dict(data_dict, title, metric_name, **plot_kwargs)
        
        # 保存图表
        if save_plot:
            plot_filename = f"{experiment_id}_{metric_name}"
            if group_by:
                if isinstance(group_by, str):
                    plot_filename += f"_by_{group_by}"
                else:
                    plot_filename += f"_by_{'_'.join(group_by)}"
            plot_filename += ".png"
            
            plot_path = self.plots_dir / plot_filename
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"💾 图表已保存: {plot_path}")
        
        return fig
    
    def _plot_from_data_dict(self, data_dict: Dict[str, pd.DataFrame], 
                           title: str, ylabel: str, **plot_kwargs) -> plt.Figure:
        """从数据字典绘制图表"""
        # 创建图表
        figsize = plot_kwargs.get('figsize', (10, 6))
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制每个分组
        for group_key, df in data_dict.items():
            steps = df['Step'].values
            values = df.iloc[:, 1].values  # 第二列是指标值
            
            # 获取颜色
            color = style_manager.get_color(group_key)
            linestyle = style_manager.get_linestyle(group_key)
            
            # 数据平滑
            if plot_kwargs.get('smooth', True):
                from ml_plotter.data_utils import data_processor
                values = data_processor.smooth_data(values, method='ema')
            
            # 绘制曲线
            ax.plot(steps, values, label=group_key, color=color, 
                   linestyle=linestyle, linewidth=6)
        
        # 应用样式
        style_manager.apply_academic_style(ax, title=title, 
                                         xlabel="Steps", ylabel=ylabel)
        
        # 设置图例
        legend_loc = plot_kwargs.get('legend_loc', 'upper left')
        legend_bbox = plot_kwargs.get('legend_bbox', None)
        
        if legend_bbox is not None:
            ax.legend(fontsize=12, loc=legend_loc, bbox_to_anchor=legend_bbox)
        else:
            ax.legend(fontsize=12, loc=legend_loc)
        
        # 设置科学计数法
        style_manager.setup_scientific_notation(ax)
        
        plt.tight_layout()
        return fig
    
    def plot_comparison_bars(self,
                           experiment_id: str,
                           metric_name: str = 'episodic_return',
                           group_by: Optional[Union[str, List[str]]] = None,
                           title: Optional[str] = None,
                           save_plot: bool = True,
                           **plot_kwargs) -> plt.Figure:
        """
        绘制性能对比柱状图
        
        Args:
            experiment_id: 实验ID
            metric_name: 指标名称
            group_by: 分组参数
            title: 图表标题
            save_plot: 是否保存图表
            **plot_kwargs: 其他绘图参数
            
        Returns:
            matplotlib Figure对象
        """
        # 获取汇总统计
        stats = self.logger.get_summary_stats(experiment_id, metric_name, group_by)
        
        if not stats:
            raise ValueError(f"未找到实验 {experiment_id} 的指标 {metric_name}")
        
        print(f"📊 绘制性能对比: {experiment_id}/{metric_name}")
        print(f"   发现 {len(stats)} 个分组")
        
        # 准备数据
        labels = list(stats.keys())
        means = [stat['final'] for stat in stats.values()]  # 使用最终值
        stds = [stat['std'] for stat in stats.values()]
        
        # 设置默认标题
        if title is None:
            title = f"{experiment_id}: Final {metric_name}"
        
        # 创建图表
        figsize = plot_kwargs.get('figsize', (10, 6))
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制柱状图
        x_pos = np.arange(len(labels))
        colors = [style_manager.get_color(label) for label in labels]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=colors, alpha=0.8, 
                     error_kw={'ecolor': 'black', 'elinewidth': 1})
        
        # 设置标签和标题
        ax.set_xlabel("Groups", fontsize=15)
        ax.set_ylabel(f"Final {metric_name}", fontsize=15, fontweight='bold')
        ax.set_title(title, fontsize=16)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # 应用网格和样式
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图表
        if save_plot:
            plot_filename = f"{experiment_id}_{metric_name}_bars"
            if group_by:
                if isinstance(group_by, str):
                    plot_filename += f"_by_{group_by}"
                else:
                    plot_filename += f"_by_{'_'.join(group_by)}"
            plot_filename += ".png"
            
            plot_path = self.plots_dir / plot_filename
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"💾 图表已保存: {plot_path}")
        
        return fig
    
    def auto_plot_experiment(self, 
                           experiment_id: str,
                           metrics: Optional[List[str]] = None,
                           group_by: Optional[Union[str, List[str]]] = None,
                           plot_types: List[str] = ['curves', 'bars'],
                           save_plots: bool = True) -> Dict[str, plt.Figure]:
        """
        自动为实验生成所有相关图表
        
        Args:
            experiment_id: 实验ID
            metrics: 要绘制的指标列表，None表示自动检测
            group_by: 分组参数
            plot_types: 图表类型列表 ['curves', 'bars', 'summary']
            save_plots: 是否保存图表
            
        Returns:
            图表字典 {plot_name: figure}
        """
        print(f"🎨 自动生成实验图表: {experiment_id}")
        
        # 自动检测指标
        if metrics is None:
            conn = sqlite3.connect(self.logger.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT metric_name 
                FROM logs 
                WHERE experiment_id = ?
            ''', (experiment_id,))
            metrics = [row[0] for row in cursor.fetchall()]
            conn.close()
        
        print(f"   检测到指标: {metrics}")
        
        figures = {}
        
        for metric in metrics:
            try:
                # 训练曲线
                if 'curves' in plot_types:
                    fig_curves = self.plot_training_curves(
                        experiment_id, metric, group_by, 
                        save_plot=save_plots
                    )
                    figures[f"{metric}_curves"] = fig_curves
                
                # 性能对比柱状图
                if 'bars' in plot_types:
                    fig_bars = self.plot_comparison_bars(
                        experiment_id, metric, group_by,
                        save_plot=save_plots
                    )
                    figures[f"{metric}_bars"] = fig_bars
                    
            except Exception as e:
                print(f"   ⚠️ 绘制 {metric} 时出错: {e}")
        
        # 生成实验摘要
        if 'summary' in plot_types:
            self.generate_experiment_summary(experiment_id, group_by, save_plots)
        
        print(f"✅ 完成！生成了 {len(figures)} 个图表")
        return figures
    
    def generate_experiment_summary(self, 
                                  experiment_id: str,
                                  group_by: Optional[Union[str, List[str]]] = None,
                                  save_summary: bool = True) -> str:
        """
        生成实验摘要报告
        
        Args:
            experiment_id: 实验ID
            group_by: 分组参数
            save_summary: 是否保存摘要
            
        Returns:
            摘要文本
        """
        print(f"📋 生成实验摘要: {experiment_id}")
        
        # 获取所有指标
        conn = sqlite3.connect(self.logger.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT metric_name 
            FROM logs 
            WHERE experiment_id = ?
        ''', (experiment_id,))
        metrics = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # 生成摘要文本
        summary_lines = []
        summary_lines.append(f"# 实验摘要报告: {experiment_id}")
        summary_lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        if group_by:
            if isinstance(group_by, str):
                summary_lines.append(f"分组参数: {group_by}")
            else:
                summary_lines.append(f"分组参数: {', '.join(group_by)}")
            summary_lines.append("")
        
        # 为每个指标生成统计
        for metric in metrics:
            summary_lines.append(f"## 指标: {metric}")
            summary_lines.append("")
            
            stats = self.logger.get_summary_stats(experiment_id, metric, group_by)
            
            # 按最终性能排序
            sorted_stats = sorted(stats.items(), key=lambda x: x[1]['final'], reverse=True)
            
            summary_lines.append("| 分组 | 最终值 | 平均值 | 标准差 | 最小值 | 最大值 | 数据点 |")
            summary_lines.append("|------|--------|--------|--------|--------|--------|--------|")
            
            for group_key, stat in sorted_stats:
                summary_lines.append(
                    f"| {group_key} | {stat['final']:.3f} | {stat['mean']:.3f} | "
                    f"{stat['std']:.3f} | {stat['min']:.3f} | {stat['max']:.3f} | {stat['count']} |"
                )
            
            summary_lines.append("")
        
        summary_text = "\n".join(summary_lines)
        
        # 保存摘要
        if save_summary:
            summary_filename = f"{experiment_id}_summary"
            if group_by:
                if isinstance(group_by, str):
                    summary_filename += f"_by_{group_by}"
                else:
                    summary_filename += f"_by_{'_'.join(group_by)}"
            summary_filename += ".md"
            
            summary_path = self.plots_dir / summary_filename
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            print(f"💾 摘要已保存: {summary_path}")
        
        return summary_text
    
    def compare_experiments(self,
                          experiment_ids: List[str],
                          metric_name: str = 'episodic_return',
                          title: Optional[str] = None,
                          save_plot: bool = True,
                          **plot_kwargs) -> plt.Figure:
        """
        比较多个实验的性能
        
        Args:
            experiment_ids: 实验ID列表
            metric_name: 要比较的指标
            title: 图表标题
            save_plot: 是否保存图表
            **plot_kwargs: 其他绘图参数
            
        Returns:
            matplotlib Figure对象
        """
        print(f"🔄 比较实验: {experiment_ids}")
        print(f"   指标: {metric_name}")
        
        # 收集所有实验的数据
        all_data = {}
        
        for exp_id in experiment_ids:
            metrics_data = self.logger.get_metrics(exp_id, metric_name, group_by=None)
            
            # 聚合同一实验的所有运行
            all_steps = []
            all_values = []
            
            for run_data in metrics_data.values():
                all_steps.extend(run_data['steps'])
                all_values.extend(run_data['values'])
            
            if all_steps:
                # 创建DataFrame
                df = pd.DataFrame({
                    'Step': all_steps,
                    f'{exp_id} - {metric_name}': all_values
                })
                all_data[exp_id] = df
        
        if not all_data:
            raise ValueError("未找到任何有效的实验数据")
        
        # 设置默认标题
        if title is None:
            title = f"Experiment Comparison: {metric_name}"
        
        # 绘制对比图
        fig = self._plot_from_data_dict(all_data, title, metric_name, **plot_kwargs)
        
        # 保存图表
        if save_plot:
            plot_filename = f"comparison_{'_vs_'.join(experiment_ids)}_{metric_name}.png"
            plot_path = self.plots_dir / plot_filename
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"💾 对比图已保存: {plot_path}")
        
        return fig


# 便捷函数
def create_integrated_logger(log_dir: str = "experiment_logs") -> MLPlotterLogger:
    """创建集成的实验日志记录和可视化器"""
    return MLPlotterLogger(log_dir)

def quick_experiment_plot(experiment_id: str, 
                         log_dir: str = "experiment_logs",
                         **kwargs) -> Dict[str, plt.Figure]:
    """快速为实验生成所有图表"""
    logger = MLPlotterLogger(log_dir)
    return logger.auto_plot_experiment(experiment_id, **kwargs)