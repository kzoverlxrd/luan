#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
潞安煤高炉碳排放监控与管理系统启动脚本
"""

import os
import sys
import subprocess
import importlib.util

def check_package(package_name):
    """检查包是否已安装"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_requirements():
    """安装依赖包"""
    print("正在安装依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依赖包安装完成！")
        return True
    except subprocess.CalledProcessError:
        print("依赖包安装失败，请手动运行: pip install -r requirements.txt")
        return False

def check_data():
    """检查数据文件是否存在"""
    data_path = "data/daily_production_data.csv"
    if not os.path.exists(data_path):
        print("数据文件不存在，正在生成虚拟数据...")
        try:
            subprocess.check_call([sys.executable, "create_dummy_data.py"])
            print("虚拟数据生成完成！")
            return True
        except subprocess.CalledProcessError:
            print("虚拟数据生成失败！")
            return False
    return True

def check_model():
    """检查模型文件是否存在"""
    model_path = "models/xgboost_carbon_model.json"
    if not os.path.exists(model_path):
        print("模型文件不存在，正在训练模型...")
        try:
            subprocess.check_call([sys.executable, "main.py"])
            print("模型训练完成！")
            return True
        except subprocess.CalledProcessError:
            print("模型训练失败！")
            return False
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("🏭 潞安煤高炉碳排放监控与管理系统")
    print("=" * 60)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("错误: 需要Python 3.8或更高版本")
        return
    
    # 检查依赖包
    required_packages = ['streamlit', 'pandas', 'numpy', 'xgboost', 'scikit-learn', 'matplotlib']
    missing_packages = [pkg for pkg in required_packages if not check_package(pkg)]
    
    if missing_packages:
        print(f"缺少依赖包: {', '.join(missing_packages)}")
        if input("是否自动安装依赖包? (y/n): ").lower() == 'y':
            if not install_requirements():
                return
        else:
            print("请手动安装依赖包后重试")
            return
    
    # 检查数据文件
    if not check_data():
        return
    
    # 检查模型文件
    if not check_model():
        return
    
    print("\n✅ 环境检查完成！")
    print("正在启动Web应用...")
    print("应用启动后，请在浏览器中访问: http://localhost:8501")
    print("按 Ctrl+C 停止应用")
    print("-" * 60)
    
    # 启动Streamlit应用
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n应用已停止")
    except Exception as e:
        print(f"启动应用时出错: {e}")

if __name__ == "__main__":
    main() 