#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP功能测试脚本
"""

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_shap_import():
    """测试SHAP导入"""
    try:
        import shap
        print("✅ SHAP库导入成功")
        print(f"SHAP版本: {shap.__version__}")
        return True
    except ImportError as e:
        print(f"❌ SHAP库导入失败: {e}")
        return False

def test_shap_functionality():
    """测试SHAP功能"""
    try:
        import shap
        import pandas as pd
        import numpy as np
        import xgboost as xgb
        
        # 创建简单的测试数据
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        y_test = np.array([10, 20, 30, 40, 50])
        
        # 训练简单模型
        model = xgb.XGBRegressor(n_estimators=10, random_state=42)
        model.fit(X_test, y_test)
        
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(X_test.iloc[0:1])
        
        print("✅ SHAP功能测试成功")
        print(f"SHAP值形状: {shap_values.shape}")
        print(f"基础值: {explainer.expected_value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ SHAP功能测试失败: {e}")
        return False

def test_data_and_model():
    """测试数据和模型"""
    try:
        import xgboost as xgb
        from data_handler import DataHandler
        from model_trainer import ModelTrainer
        
        # 检查数据文件
        data_path = "data/daily_production_data.csv"
        if not os.path.exists(data_path):
            print("❌ 数据文件不存在，请先运行 create_dummy_data.py")
            return False
        
        # 加载数据
        data_handler = DataHandler(data_path)
        X, y = data_handler.process()
        
        if X is None or y is None:
            print("❌ 数据加载失败")
            return False
        
        print(f"✅ 数据加载成功，形状: {X.shape}")
        
        # 检查模型文件
        model_path = "models/xgboost_carbon_model.json"
        if not os.path.exists(model_path):
            print("❌ 模型文件不存在，请先运行 main.py")
            return False
        
        # 加载模型
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        
        print("✅ 模型加载成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据和模型测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("SHAP功能测试")
    print("=" * 50)
    
    # 测试SHAP导入
    print("\n1. 测试SHAP库导入...")
    shap_import_ok = test_shap_import()
    
    # 测试SHAP功能
    print("\n2. 测试SHAP功能...")
    shap_func_ok = test_shap_functionality()
    
    # 测试数据和模型
    print("\n3. 测试数据和模型...")
    data_model_ok = test_data_and_model()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("=" * 50)
    
    if shap_import_ok and shap_func_ok and data_model_ok:
        print("🎉 所有测试通过！SHAP功能可以正常使用。")
        print("现在可以运行 Streamlit 应用了。")
    else:
        print("⚠️ 部分测试失败，请检查以下问题:")
        if not shap_import_ok:
            print("- 请安装SHAP库: pip install shap")
        if not shap_func_ok:
            print("- SHAP库安装可能有问题，请重新安装")
        if not data_model_ok:
            print("- 请确保数据和模型文件存在")

if __name__ == "__main__":
    main() 