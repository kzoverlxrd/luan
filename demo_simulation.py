#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真与优化功能演示脚本
"""

import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_handler import DataHandler
from model_trainer import ModelTrainer

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def demo_simulation():
    """
    演示仿真与优化功能
    """
    print("=" * 60)
    print("🎛️ 仿真与优化功能演示")
    print("=" * 60)
    
    # 加载数据
    print("\n1. 加载数据...")
    data_handler = DataHandler("data/daily_production_data.csv")
    X, y = data_handler.process()
    
    if X is None or y is None:
        print("❌ 数据加载失败")
        return
    
    print(f"✅ 数据加载成功，形状: {X.shape}")
    
    # 加载模型
    print("\n2. 加载模型...")
    model_path = "models/xgboost_carbon_model.json"
    if not os.path.exists(model_path):
        print("❌ 模型文件不存在，请先运行 main.py")
        return
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    print("✅ 模型加载成功")
    
    # 选择演示日期
    selected_date = X.index[50]  # 选择第50天的数据
    print(f"\n3. 选择演示日期: {selected_date.strftime('%Y-%m-%d')}")
    
    # 获取原始数据
    original_data = X.loc[selected_date:selected_date]
    original_prediction = model.predict(original_data)[0]
    
    print(f"原始预测值: {original_prediction:.2f} 吨")
    
    # 演示参数调整
    print("\n4. 参数调整演示:")
    
    # 煤比率调整
    coal_ratios = [0.05, 0.06, 0.07, 0.08, 0.09]
    coal_predictions = []
    
    for ratio in coal_ratios:
        simulation_data = original_data.copy()
        simulation_data['coal_ratio'] = ratio
        simulation_data['volatility_effect'] = ratio * simulation_data['luan_coal_vd_avg'].iloc[0]
        simulation_data['ash_burden_effect'] = ratio * simulation_data['luan_coal_ash_avg'].iloc[0]
        simulation_data['effective_carbon_input'] = ratio * simulation_data['luan_coal_fcad_avg'].iloc[0]
        
        prediction = model.predict(simulation_data)[0]
        coal_predictions.append(prediction)
        
        change = prediction - original_prediction
        print(f"  煤比率 {ratio:.3f}: {prediction:.2f} 吨 (变化: {change:+.2f})")
    
    # 鼓风温度调整
    temperatures = [1150, 1200, 1250, 1300, 1350]
    temp_predictions = []
    
    for temp in temperatures:
        simulation_data = original_data.copy()
        simulation_data['blast_temp_avg'] = temp
        
        prediction = model.predict(simulation_data)[0]
        temp_predictions.append(prediction)
        
        change = prediction - original_prediction
        print(f"  鼓风温度 {temp}°C: {prediction:.2f} 吨 (变化: {change:+.2f})")
    
    # 创建可视化图表
    print("\n5. 生成可视化图表...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 煤比率敏感性图
    ax1.plot(coal_ratios, coal_predictions, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=original_prediction, color='r', linestyle='--', alpha=0.7, label='原始预测')
    ax1.set_xlabel('煤比率')
    ax1.set_ylabel('预测碳排放量 (吨)')
    ax1.set_title('煤比率敏感性分析')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 鼓风温度敏感性图
    ax2.plot(temperatures, temp_predictions, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=original_prediction, color='r', linestyle='--', alpha=0.7, label='原始预测')
    ax2.set_xlabel('鼓风温度 (°C)')
    ax2.set_ylabel('预测碳排放量 (吨)')
    ax2.set_title('鼓风温度敏感性分析')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/simulation_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 图表已保存到: models/simulation_demo.png")
    
    # 分析结果
    print("\n6. 分析结果:")
    
    # 找到最优煤比率
    min_coal_idx = np.argmin(coal_predictions)
    optimal_coal_ratio = coal_ratios[min_coal_idx]
    min_coal_prediction = coal_predictions[min_coal_idx]
    
    # 找到最优温度
    min_temp_idx = np.argmin(temp_predictions)
    optimal_temp = temperatures[min_temp_idx]
    min_temp_prediction = temp_predictions[min_temp_idx]
    
    print(f"最优煤比率: {optimal_coal_ratio:.3f} (预测: {min_coal_prediction:.2f} 吨)")
    print(f"最优鼓风温度: {optimal_temp}°C (预测: {min_temp_prediction:.2f} 吨)")
    
    # 计算改进潜力
    coal_improvement = original_prediction - min_coal_prediction
    temp_improvement = original_prediction - min_temp_prediction
    
    print(f"煤比率优化潜力: {coal_improvement:.2f} 吨")
    print(f"温度优化潜力: {temp_improvement:.2f} 吨")
    
    # 综合建议
    print("\n7. 优化建议:")
    if coal_improvement > temp_improvement:
        print(f"🎯 优先调整煤比率到 {optimal_coal_ratio:.3f}，可减少碳排放 {coal_improvement:.2f} 吨")
    else:
        print(f"🎯 优先调整鼓风温度到 {optimal_temp}°C，可减少碳排放 {temp_improvement:.2f} 吨")
    
    print("\n✅ 仿真与优化功能演示完成！")
    print("现在可以运行 Streamlit 应用体验完整的交互式功能。")

if __name__ == "__main__":
    demo_simulation() 