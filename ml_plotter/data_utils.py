"""
ML Plotter - 数据处理工具模块
包含自动CSV加载、列名识别、数据清洗、平滑处理等功能
"""

import os
import glob
import pandas as pd
import numpy as np
import scipy.signal as sig
from typing import List, Dict, Tuple, Optional, Union
import re

# 常见的列名候选 - 基于原程序的经验
X_COLUMN_CANDIDATES = [
    'Step', 'step', 'global_step', '_step', 
    'TotalSteps', 'total_steps', 'Timestep', 'timestep', 
    'Epoch', 'epoch', 'Iteration', 'iteration', 'Frame', 'frame',
    'time', 'index'
]

Y_COLUMN_CANDIDATES = [
    '- episode_return',                     # 最常见的格式
    'charts/episodic_return',               
    'eval/avg_reward',
    ' - episode_return',
    'rollout/ep_rew_mean',
    'Value',
    'eval_episodes/mean_reward',
    'eval_episode_reward',
    'AverageReturn',
    'MeanReturn',
    'performance/mean_episode_reward',
    'Train/mean_reward',
    'charts/zero_grad_ratio',
    'zero_vectors'
]

class DataProcessor:
    """数据处理器 - 统一处理各种数据格式和操作"""
    
    def __init__(self, smooth_window: int = 500):
        self.smooth_window = smooth_window
    
    def find_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """自动识别步数列和性能列"""
        x_col, y_col = None, None
        df_columns_original = list(df.columns)
        df_columns_lower = [col.lower() for col in df.columns]
        
        # 查找X列（步数列）
        for candidate in X_COLUMN_CANDIDATES:
            try:
                idx = df_columns_lower.index(candidate.lower())
                x_col = df_columns_original[idx]
                break
            except ValueError:
                continue
        
        # 查找Y列（性能列）- 支持精确匹配和后缀匹配
        for candidate_pattern in Y_COLUMN_CANDIDATES:
            candidate_lower = candidate_pattern.lower()
            # 精确匹配
            try:
                idx = df_columns_lower.index(candidate_lower)
                y_col = df_columns_original[idx]
                break
            except ValueError:
                # 后缀匹配
                for i, col_name_lower in enumerate(df_columns_lower):
                    if col_name_lower.endswith(candidate_lower):
                        y_col = df_columns_original[i]
                        break
                if y_col:
                    break
        
        return x_col, y_col
    
    def load_csv_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """加载单个CSV文件并进行基本清洗"""
        try:
            df = pd.read_csv(file_path)
            x_col, y_col = self.find_columns(df)
            
            if x_col is None or y_col is None:
                print(f"Warning: Could not find suitable columns in {file_path}")
                return None
            
            # 选择相关列并重命名
            df_processed = df[[x_col, y_col]].copy()
            df_processed.rename(columns={x_col: 'steps', y_col: 'values'}, inplace=True)
            
            # 转换为数值类型
            df_processed['steps'] = pd.to_numeric(df_processed['steps'], errors='coerce')
            df_processed['values'] = pd.to_numeric(df_processed['values'], errors='coerce')
            
            # 删除NaN值
            df_processed.dropna(inplace=True)
            
            # 按步数排序
            df_processed.sort_values(by='steps', inplace=True)
            
            # 删除重复的步数，保留第一个
            df_processed.drop_duplicates(subset=['steps'], keep='first', inplace=True)
            
            return df_processed
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_folder_data(self, folder_path: str) -> Optional[Dict[str, np.ndarray]]:
        """加载文件夹中的所有CSV文件并处理"""
        if not os.path.isdir(folder_path):
            print(f"Warning: {folder_path} is not a directory")
            return None
        
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not csv_files:
            print(f"Warning: No CSV files found in {folder_path}")
            return None
        
        all_runs = []
        for csv_file in csv_files:
            df = self.load_csv_data(csv_file)
            if df is not None and not df.empty:
                all_runs.append(df)
        
        if not all_runs:
            print(f"Warning: No valid data loaded from {folder_path}")
            return None
        
        return self.align_and_aggregate_runs(all_runs)
    
    def align_and_aggregate_runs(self, runs: List[pd.DataFrame]) -> Dict[str, np.ndarray]:
        """对齐多个运行的数据并计算统计量"""
        try:
            # 使用外连接对齐所有运行的数据
            aligned_data = []
            for run_df in runs:
                run_df.set_index('steps', inplace=True)
                aligned_data.append(run_df['values'])
            
            # 合并所有数据
            combined_df = pd.concat(aligned_data, axis=1, join='outer')
            combined_df.sort_index(inplace=True)
            
            # 线性插值填充缺失值
            combined_df.interpolate(method='linear', limit_direction='both', inplace=True)
            
            # 删除仍然为NaN的行
            combined_df.dropna(how='all', inplace=True)
            
            if combined_df.empty:
                return None
            
            # 计算统计量
            steps = combined_df.index.to_numpy()
            means = combined_df.mean(axis=1).to_numpy()
            stds = combined_df.std(axis=1, ddof=0).to_numpy() if combined_df.shape[1] > 1 else np.zeros_like(means)
            
            return {
                'steps': steps,
                'means': means,
                'stds': stds,
                'n_runs': combined_df.shape[1]
            }
            
        except Exception as e:
            print(f"Error aligning runs: {e}")
            return None
    
    def smooth_data(self, data: np.ndarray, method: str = 'ema') -> np.ndarray:
        """数据平滑处理"""
        if len(data) <= 1:
            return data
        
        if method == 'ema':
            # 指数移动平均
            if len(data) > self.smooth_window and self.smooth_window > 1:
                return pd.Series(data).ewm(span=self.smooth_window, adjust=False).mean().to_numpy()
        elif method == 'savgol':
            # Savitzky-Golay滤波
            window_length = min(31, len(data) if len(data) % 2 == 1 else len(data) - 1)
            if window_length >= 3:
                return sig.savgol_filter(data, window_length=window_length, polyorder=2)
        
        return data
    
    def normalize_data(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """数据归一化"""
        if method == 'minmax':
            min_val, max_val = np.min(data), np.max(data)
            if max_val > min_val:
                return (data - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val, std_val = np.mean(data), np.std(data)
            if std_val > 0:
                return (data - mean_val) / std_val
        
        return data
    
    def extract_max_scores(self, folder_path: str) -> List[float]:
        """提取文件夹中所有运行的最大分数"""
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        max_scores = []
        
        for csv_file in csv_files:
            df = self.load_csv_data(csv_file)
            if df is not None and not df.empty:
                max_score = df['values'].max()
                max_scores.append(max_score)
        
        return max_scores
    
    def auto_discover_conditions(self, base_path: str, 
                                condition_names: Optional[List[str]] = None) -> Dict[str, str]:
        """自动发现实验条件文件夹"""
        if not os.path.isdir(base_path):
            return {}
        
        discovered = {}
        
        # 如果指定了条件名称，直接查找
        if condition_names:
            for condition in condition_names:
                condition_path = os.path.join(base_path, condition)
                if os.path.isdir(condition_path):
                    csv_files = glob.glob(os.path.join(condition_path, "*.csv"))
                    if csv_files:
                        discovered[condition] = condition_path
        else:
            # 自动发现包含CSV文件的子文件夹
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path):
                    csv_files = glob.glob(os.path.join(item_path, "*.csv"))
                    if csv_files:
                        discovered[item] = item_path
        
        return discovered

# 全局数据处理器实例
data_processor = DataProcessor()