#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
潞安煤碳排放预测模型训练主程序
"""

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_handler import DataHandler
from model_trainer import ModelTrainer

def main():
    """
    主函数：执行完整的数据处理和模型训练流程
    """
    print("=" * 50)
    print("潞安煤碳排放预测模型训练")
    print("=" * 50)
    
    # 1. 数据加载和预处理
    print("\n1. 数据加载和预处理...")
    data_handler = DataHandler("data/daily_production_data.csv")
    X, y = data_handler.process()
    
    if X is None or y is None:
        print("错误：数据加载失败")
        return
    
    # 2. 模型训练
    print("\n2. 模型训练...")
    trainer = ModelTrainer(test_size=0.2, random_state=42)
    model = trainer.train(X, y)
    
    # 3. 获取特征重要性
    print("\n3. 特征重要性分析...")
    importance_df = trainer.get_feature_importance()
    print("\n特征重要性排序:")
    print(importance_df.to_string(index=False))
    
    # 4. 生成可视化图表
    print("\n4. 生成可视化图表...")
    trainer.plot_predictions()
    
    # 5. 输出总结
    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)
    print(f"特征数量: {len(X.columns)}")
    print(f"样本数量: {len(X)}")
    print(f"训练集大小: {trainer.X_train.shape[0]}")
    print(f"测试集大小: {trainer.X_test.shape[0]}")
    print(f"模型已保存到: models/xgboost_carbon_model.json")
    print(f"模型信息已保存到: models/model_info.json")
    print(f"特征重要性图已保存到: models/feature_importance.png")
    print(f"预测对比图已保存到: models/prediction_comparison.png")

if __name__ == "__main__":
    main() 