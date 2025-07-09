import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import os
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from src.feature_name_map import en2zh

class ModelTrainer:
    """
    模型训练类，用于训练和评估XGBoost回归模型
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化ModelTrainer
        
        Args:
            test_size (float): 测试集比例，默认0.2
            random_state (int): 随机种子，默认42
        """
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
        """
        训练XGBoost回归模型
        
        Args:
            X (pd.DataFrame): 特征数据
            y (pd.Series): 目标变量
            
        Returns:
            xgb.XGBRegressor: 训练好的模型
        """
        print("开始模型训练...")
        
        # 保存特征名称
        self.feature_names = list(X.columns)
        
        # 分割训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        
        # 初始化XGBoost回归模型
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 训练模型
        print("训练模型中...")
        try:
            self.model.fit(
                self.X_train, 
                self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
        except TypeError:
            # 兼容低版本XGBoost
            self.model.fit(
                self.X_train, 
                self.y_train,
                eval_set=[(self.X_test, self.y_test)]
            )
        
        # 评估模型
        self._evaluate_model()
        
        # 保存模型
        self._save_model()
        
        # 生成特征重要性图
        self._plot_feature_importance()
        
        return self.model
    
    def _evaluate_model(self) -> Dict[str, float]:
        """
        评估模型性能
        
        Returns:
            Dict[str, float]: 评估指标字典
        """
        # 预测
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # 计算评估指标
        metrics = {
            'train_r2': r2_score(self.y_train, y_train_pred),
            'test_r2': r2_score(self.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred)
        }
        
        print("\n=== 模型评估结果 ===")
        print(f"训练集 R² 分数: {metrics['train_r2']:.4f}")
        print(f"测试集 R² 分数: {metrics['test_r2']:.4f}")
        print(f"训练集 RMSE: {metrics['train_rmse']:.2f}")
        print(f"测试集 RMSE: {metrics['test_rmse']:.2f}")
        print(f"训练集 MAE: {metrics['train_mae']:.2f}")
        print(f"测试集 MAE: {metrics['test_mae']:.2f}")
        
        return metrics
    
    def _save_model(self) -> None:
        """
        保存模型到文件
        """
        # 确保models目录存在
        os.makedirs('models', exist_ok=True)
        
        # 保存模型
        model_path = "models/xgboost_carbon_model.json"
        self.model.save_model(model_path)
        
        # 保存模型信息
        model_info = {
            'model_type': 'XGBoost',
            'feature_names': self.feature_names,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'model_params': self.model.get_params(),
            'feature_importance': self.model.feature_importances_.tolist()
        }
        
        info_path = "models/model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n模型已保存到: {model_path}")
        print(f"模型信息已保存到: {info_path}")
    
    def _plot_feature_importance(self) -> None:
        """
        绘制特征重要性图
        """
        # 确保models目录存在
        os.makedirs('models', exist_ok=True)
        
        # 获取特征重要性
        importance = self.model.feature_importances_
        feature_names = self.feature_names
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), [en2zh.get(f, f) for f in importance_df['feature']])
        plt.xlabel('特征重要性')
        plt.title('XGBoost 特征重要性')
        plt.tight_layout()
        
        # 保存图片
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("特征重要性图已保存到: models/feature_importance.png")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        Args:
            X (pd.DataFrame): 特征数据
            
        Returns:
            np.ndarray: 预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        Returns:
            pd.DataFrame: 特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_predictions(self) -> None:
        """
        绘制预测结果对比图
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        # 预测
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 训练集预测对比
        ax1.scatter(self.y_train, y_train_pred, alpha=0.6)
        ax1.plot([self.y_train.min(), self.y_train.max()], 
                [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        ax1.set_xlabel('实际值')
        ax1.set_ylabel('预测值')
        ax1.set_title('训练集预测对比')
        ax1.grid(True, alpha=0.3)
        
        # 测试集预测对比
        ax2.scatter(self.y_test, y_test_pred, alpha=0.6)
        ax2.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax2.set_xlabel('实际值')
        ax2.set_ylabel('预测值')
        ax2.set_title('测试集预测对比')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/prediction_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("预测对比图已保存到: models/prediction_comparison.png")


# 使用示例
if __name__ == "__main__":
    # 导入DataHandler
    from data_handler import DataHandler
    
    # 加载和处理数据
    data_handler = DataHandler("data/daily_production_data.csv")
    X, y = data_handler.process()
    
    if X is not None and y is not None:
        # 创建模型训练器
        trainer = ModelTrainer(test_size=0.2, random_state=42)
        
        # 训练模型
        model = trainer.train(X, y)
        
        # 获取特征重要性
        importance_df = trainer.get_feature_importance()
        print("\n=== 特征重要性 ===")
        importance_df_show = importance_df.copy()
        importance_df_show['feature'] = importance_df_show['feature'].map(lambda x: en2zh.get(x, x))
        print(importance_df_show)
        
        # 绘制预测对比图
        trainer.plot_predictions()
        
        print("\n模型训练完成！") 