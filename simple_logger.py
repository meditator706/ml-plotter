"""
Simple ML Logger - 类似wandb的简洁接口
极简的实验记录和可视化系统
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
from ml_plotter import MLPlotter

# 全局状态
_current_run = None
_db_path = None
_project_name = None

def init(project: str = "ml_experiments", name: Optional[str] = None, config: Optional[Dict] = None):
    """
    初始化实验记录 - 类似 wandb.init()
    
    Args:
        project: 项目名称
        name: 运行名称，默认自动生成
        config: 实验配置参数
    """
    global _current_run, _db_path, _project_name
    
    import time
    
    _project_name = project
    
    # 创建项目目录
    project_dir = Path(f"logs/{project}")
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据库路径
    _db_path = project_dir / "experiments.db"
    
    # 初始化数据库
    _init_database()
    
    # 生成运行名称
    if name is None:
        name = f"run_{int(time.time())}"
    
    # 设置当前运行
    _current_run = {
        'project': project,
        'name': name,
        'config': config or {},
        'step': 0
    }
    
    print(f"🚀 Started run: {project}/{name}")
    if config:
        print(f"   Config: {config}")

def log(metrics: Dict[str, float], step: Optional[int] = None):
    """
    记录指标 - 类似 wandb.log()
    
    Args:
        metrics: 指标字典 {"loss": 0.5, "accuracy": 0.9}
        step: 步数，默认自动递增
    """
    global _current_run
    
    if _current_run is None:
        raise RuntimeError("请先调用 init() 初始化实验")
    
    # 自动递增步数
    if step is None:
        step = _current_run['step']
        _current_run['step'] += 1
    
    # 保存到数据库
    conn = sqlite3.connect(_db_path)
    cursor = conn.cursor()
    
    for metric_name, value in metrics.items():
        cursor.execute('''
            INSERT INTO logs (project, run_name, step, metric_name, metric_value, config)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            _current_run['project'],
            _current_run['name'], 
            step,
            metric_name,
            float(value),
            json.dumps(_current_run['config'])
        ))
    
    conn.commit()
    conn.close()

def finish():
    """结束当前运行"""
    global _current_run
    
    if _current_run:
        print(f"✅ Finished run: {_current_run['project']}/{_current_run['name']}")
        _current_run = None

def plot(project: str, 
         metric: str = "loss",
         runs: Optional[List[str]] = None,
         group_by: Optional[str] = None,
         title: Optional[str] = None,
         save: bool = True) -> plt.Figure:
    """
    绘制实验结果 - 极简接口
    
    Args:
        project: 项目名称
        metric: 要绘制的指标名称
        runs: 指定运行名称列表，None表示所有运行
        group_by: 按配置参数分组
        title: 图表标题
        save: 是否保存图表
        
    Returns:
        matplotlib Figure对象
    """
    db_path = Path(f"logs/{project}/experiments.db")
    
    if not db_path.exists():
        raise FileNotFoundError(f"项目 {project} 不存在")
    
    # 从数据库读取数据
    conn = sqlite3.connect(db_path)
    
    # 构建查询
    query = '''
        SELECT run_name, step, metric_value, config
        FROM logs 
        WHERE project = ? AND metric_name = ?
    '''
    params = [project, metric]
    
    if runs:
        placeholders = ','.join(['?' for _ in runs])
        query += f' AND run_name IN ({placeholders})'
        params.extend(runs)
    
    query += ' ORDER BY run_name, step'
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        raise ValueError(f"未找到项目 {project} 的指标 {metric}")
    
    # 解析配置
    df['config'] = df['config'].apply(json.loads)
    
    # 分组逻辑
    if group_by:
        # 按指定参数分组
        df['group'] = df['config'].apply(lambda x: f"{group_by}={x.get(group_by, 'unknown')}")
    else:
        # 按运行名称分组
        df['group'] = df['run_name']
    
    # 创建图表
    plotter = MLPlotter()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制每个分组
    for group_name in df['group'].unique():
        group_data = df[df['group'] == group_name]
        
        # 计算平均值和标准差（如果有多个运行）
        if group_by:
            # 按步数聚合
            agg_data = group_data.groupby('step')['metric_value'].agg(['mean', 'std']).reset_index()
            steps = agg_data['step']
            means = agg_data['mean']
            stds = agg_data['std'].fillna(0)
            
            # 绘制主线
            ax.plot(steps, means, label=group_name, linewidth=3)
            
            # 绘制置信区间
            if len(group_data['run_name'].unique()) > 1:  # 多个运行才显示置信区间
                ax.fill_between(steps, means - stds, means + stds, alpha=0.2)
        else:
            # 单个运行，直接绘制
            ax.plot(group_data['step'], group_data['metric_value'], 
                   label=group_name, linewidth=3)
    
    # 设置样式
    ax.set_xlabel('Steps', fontsize=14)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
    
    if title is None:
        if group_by:
            title = f"{project}: {metric} (grouped by {group_by})"
        else:
            title = f"{project}: {metric}"
    
    ax.set_title(title, fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    if save:
        save_dir = Path(f"logs/{project}/plots")
        save_dir.mkdir(exist_ok=True)
        
        filename = f"{metric}"
        if group_by:
            filename += f"_by_{group_by}"
        filename += ".png"
        
        save_path = save_dir / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 图表已保存: {save_path}")
    
    return fig

def summary(project: str, group_by: Optional[str] = None):
    """
    显示项目摘要
    
    Args:
        project: 项目名称
        group_by: 按参数分组
    """
    db_path = Path(f"logs/{project}/experiments.db")
    
    if not db_path.exists():
        raise FileNotFoundError(f"项目 {project} 不存在")
    
    conn = sqlite3.connect(db_path)
    
    # 获取所有指标
    metrics_df = pd.read_sql_query('''
        SELECT DISTINCT metric_name FROM logs WHERE project = ?
    ''', conn, params=[project])
    
    print(f"\n📊 项目摘要: {project}")
    print("=" * 50)
    
    for metric in metrics_df['metric_name']:
        print(f"\n📈 指标: {metric}")
        
        # 获取该指标的数据
        df = pd.read_sql_query('''
            SELECT run_name, metric_value, config
            FROM logs 
            WHERE project = ? AND metric_name = ?
            ORDER BY run_name, step
        ''', conn, params=[project, metric])
        
        if df.empty:
            continue
            
        # 解析配置
        df['config'] = df['config'].apply(json.loads)
        
        # 计算每个运行的最终值
        final_values = df.groupby('run_name')['metric_value'].last()
        
        if group_by:
            # 按参数分组
            df['group'] = df['config'].apply(lambda x: f"{group_by}={x.get(group_by, 'unknown')}")
            
            # 计算分组统计
            group_stats = []
            for group_name in df['group'].unique():
                group_runs = df[df['group'] == group_name]['run_name'].unique()
                group_values = final_values[final_values.index.isin(group_runs)]
                
                group_stats.append({
                    'group': group_name,
                    'mean': group_values.mean(),
                    'std': group_values.std(),
                    'count': len(group_values),
                    'best': group_values.max()
                })
            
            # 按最佳性能排序
            group_stats.sort(key=lambda x: x['best'], reverse=True)
            
            for stat in group_stats:
                print(f"  🔹 {stat['group']}")
                print(f"     最佳: {stat['best']:.4f}")
                print(f"     平均: {stat['mean']:.4f} ± {stat['std']:.4f}")
                print(f"     运行数: {stat['count']}")
        else:
            # 显示所有运行
            sorted_runs = final_values.sort_values(ascending=False)
            for run_name, value in sorted_runs.head(5).items():
                print(f"  🔹 {run_name}: {value:.4f}")
    
    conn.close()

def _init_database():
    """初始化数据库"""
    conn = sqlite3.connect(_db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT,
            run_name TEXT,
            step INTEGER,
            metric_name TEXT,
            metric_value REAL,
            config TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_project_metric 
        ON logs(project, metric_name, run_name, step)
    ''')
    
    conn.commit()
    conn.close()

# 便捷别名
start = init
save = plot