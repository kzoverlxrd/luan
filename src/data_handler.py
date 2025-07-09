import pandas as pd
import numpy as np
from typing import Tuple

class DataHandler:
    """
    数据处理类，用于加载和预处理生产数据
    """
    
    def __init__(self, csv_path: str):
        """
        初始化DataHandler
        
        Args:
            csv_path (str): CSV文件路径
        """
        self.csv_path = csv_path
        self.data = None
        self.X = None
        self.y = None
        
    def load_data(self) -> pd.DataFrame:
        """
        加载CSV数据
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"成功加载数据，形状: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"错误: 找不到文件 {self.csv_path}")
            return None
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None
    
    def process(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        处理数据，创建特征和目标变量
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: 特征DataFrame和目标Series
        """
        if self.data is None:
            self.load_data()
        
        if self.data is None:
            return None, None
        
        # 创建数据副本以避免修改原始数据
        df = self.data.copy()
        
        # 将'date'列设为索引
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 创建比率特征
        df['coke_ratio'] = df['coke_consumption_total'] / df['iron_output_total']
        df['coal_ratio'] = df['luan_coal_injection_total'] / df['iron_output_total']
        
        # 创建体现潞安煤特性的交叉特征
        df['volatility_effect'] = df['coal_ratio'] * df['luan_coal_vd_avg']  # coal_ratio * Vd
        df['ash_burden_effect'] = df['coal_ratio'] * df['luan_coal_ash_avg']  # coal_ratio * Ad
        df['effective_carbon_input'] = df['coal_ratio'] * df['luan_coal_fcad_avg']  # coal_ratio * Fcad
        
        # 定义特征列（包含原始特征和所有新创建的交叉特征）
        feature_columns = [
            # 原始特征
            'blast_temp_avg',
            'blast_volume_total', 
            'coke_consumption_total',
            'luan_coal_injection_total',
            'luan_coal_ash_avg',
            'luan_coal_vd_avg',
            'luan_coal_fcad_avg',
            'iron_output_total',
            # 新创建的比率特征
            'coke_ratio',
            'coal_ratio',
            # 新创建的交叉特征
            'volatility_effect',
            'ash_burden_effect', 
            'effective_carbon_input'
        ]
        
        # 定义目标列
        target_column = 'carbon_emission_co2'
        
        # 创建特征DataFrame和目标Series
        self.X = df[feature_columns]
        self.y = df[target_column]
        
        print(f"数据处理完成:")
        print(f"- 特征形状: {self.X.shape}")
        print(f"- 目标形状: {self.y.shape}")
        print(f"- 特征列: {list(self.X.columns)}")
        print(f"- 目标列: {target_column}")
        
        return self.X, self.y
    
    def get_feature_info(self) -> dict:
        """
        获取特征信息
        
        Returns:
            dict: 特征信息字典
        """
        if self.X is None:
            print("请先调用process()方法处理数据")
            return {}
        
        info = {
            'feature_count': len(self.X.columns),
            'sample_count': len(self.X),
            'features': list(self.X.columns),
            'feature_types': self.X.dtypes.to_dict(),
            'missing_values': self.X.isnull().sum().to_dict(),
            'feature_ranges': {}
        }
        
        for col in self.X.columns:
            info['feature_ranges'][col] = {
                'min': self.X[col].min(),
                'max': self.X[col].max(),
                'mean': self.X[col].mean(),
                'std': self.X[col].std()
            }
        
        return info
    
    def get_target_info(self) -> dict:
        """
        获取目标变量信息
        
        Returns:
            dict: 目标变量信息字典
        """
        if self.y is None:
            print("请先调用process()方法处理数据")
            return {}
        
        info = {
            'target_name': self.y.name,
            'sample_count': len(self.y),
            'target_type': str(self.y.dtype),
            'missing_values': self.y.isnull().sum(),
            'statistics': {
                'min': self.y.min(),
                'max': self.y.max(),
                'mean': self.y.mean(),
                'std': self.y.std(),
                'median': self.y.median()
            }
        }
        
        return info


# 使用示例
if __name__ == "__main__":
    # 创建DataHandler实例
    data_handler = DataHandler("data/daily_production_data.csv")
    
    # 处理数据
    X, y = data_handler.process()
    
    if X is not None and y is not None:
        print("\n=== 特征信息 ===")
        feature_info = data_handler.get_feature_info()
        print(f"特征数量: {feature_info['feature_count']}")
        print(f"样本数量: {feature_info['sample_count']}")
        
        print("\n=== 目标变量信息 ===")
        target_info = data_handler.get_target_info()
        print(f"目标变量: {target_info['target_name']}")
        print(f"统计信息: {target_info['statistics']}")
        
        print("\n=== 数据预览 ===")
        print("特征数据前5行:")
        print(X.head())
        print("\n目标数据前5行:")
        print(y.head()) 