# 潞安煤高炉碳排放监控与管理系统

基于机器学习的智能碳排放预测和监控系统，专门针对潞安煤高炉生产环境设计。

## 🏭 项目概述

本项目结合了数据科学和工业应用，通过分析高炉生产数据来预测碳排放量，为节能减排提供数据支持。

### 主要功能

- 📊 **数据预处理**: 自动处理生产数据，创建特征工程
- 🤖 **机器学习模型**: 使用XGBoost进行碳排放预测
- 📈 **实时监控**: Streamlit Web界面展示预测结果
- 📋 **数据可视化**: 趋势分析和特征重要性展示
- 🔍 **智能诊断**: 使用SHAP进行模型解释和特征影响分析

## 📁 项目结构

```
luan carbon enhanced/
├── data/
│   └── daily_production_data.csv    # 日度生产数据
├── models/                          # 模型文件目录
│   ├── xgboost_carbon_model.json   # 训练好的XGBoost模型
│   ├── model_info.json             # 模型信息
│   ├── feature_importance.png      # 特征重要性图
│   └── prediction_comparison.png   # 预测对比图
├── src/
│   ├── data_handler.py             # 数据处理类
│   └── model_trainer.py            # 模型训练类
├── app.py                          # Streamlit Web应用
├── main.py                         # 主训练脚本
├── create_dummy_data.py            # 虚拟数据生成脚本
├── test_shap.py                    # SHAP功能测试脚本
├── demo_simulation.py              # 仿真功能演示脚本
├── analyze_features.py             # 特征重要性分析脚本
├── requirements.txt                # 依赖包列表
└── README.md                       # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备

确保已安装Python 3.8+，然后安装依赖包：

```bash
pip install -r requirements.txt
```

### 2. 数据准备

如果还没有数据，可以生成虚拟数据：

```bash
python create_dummy_data.py
```

### 3. 模型训练

运行主训练脚本：

```bash
python main.py
```

这将：
- 加载和预处理数据
- 训练XGBoost模型
- 评估模型性能
- 保存模型和可视化结果

### 4. 启动Web应用

启动Streamlit应用：

```bash
streamlit run app.py
```

然后在浏览器中访问 `http://localhost:8501`

### 5. 测试SHAP功能（可选）

测试SHAP功能是否正常工作：

```bash
python test_shap.py
```

### 6. 运行功能演示（可选）

运行仿真功能演示：

```bash
python demo_simulation.py
```

运行特征重要性分析：

```bash
python analyze_features.py
```

## 📊 数据说明

### 输入特征

- **blast_temp_avg**: 平均鼓风温度 (°C)
- **blast_volume_total**: 总鼓风量 (m³)
- **coke_consumption_total**: 总焦炭消耗 (吨)
- **luan_coal_injection_total**: 总潞安煤喷吹量 (吨)
- **luan_coal_ash_avg**: 潞安煤平均灰分 (%)
- **luan_coal_vd_avg**: 潞安煤平均挥发分 (%)
- **luan_coal_fcad_avg**: 潞安煤平均固定碳 (%)
- **iron_output_total**: 总铁水产量 (吨)

### 衍生特征

- **coke_ratio**: 焦炭比率 (焦炭消耗/铁水产量)
- **coal_ratio**: 煤比率 (煤喷吹量/铁水产量)
- **volatility_effect**: 挥发分效应 (煤比率 × 挥发分)
- **ash_burden_effect**: 灰分负担效应 (煤比率 × 灰分)
- **effective_carbon_input**: 有效碳输入 (煤比率 × 固定碳)

### 目标变量

- **carbon_emission_total**: 总碳排放量 (吨)

## 🎯 模型性能

系统使用XGBoost回归模型，具有以下特点：

- **高精度**: R² 分数通常在0.9以上
- **快速训练**: 支持增量学习和早停机制
- **特征重要性**: 自动识别关键影响因素
- **可解释性**: 提供特征重要性分析

## 🖥️ Web界面功能

### 主要面板

1. **📈 碳排放监控**
   - 模型预测值 vs 实际记录值
   - 预测误差率计算
   - 当日详细生产数据

2. **📊 趋势分析**
   - 最近30天碳排放趋势图
   - 特征重要性排名
   - 系统信息概览

3. **🔍 智能诊断分析**
   - SHAP瀑布图显示特征贡献
   - 详细的特征影响解释
   - 基于数据的优化建议

4. **🎛️ 仿真与优化**
   - 实时参数调整滑块
   - 预测结果对比分析
   - 参数敏感性分析
   - 智能优化建议

5. **🏭 潞安煤特性影响权重分析**
   - 特征重要性可视化
   - 交叉特征分析
   - 深度学习能力展示
   - 技术优势总结

### 侧边栏

- **📅 日期选择**: 选择要查看的具体日期
- **📊 系统信息**: 显示数据统计和模型信息

## 🔧 技术栈

- **数据处理**: pandas, numpy
- **机器学习**: scikit-learn, xgboost
- **可视化**: matplotlib, seaborn
- **Web框架**: streamlit
- **模型解释**: shap

## 📈 使用场景

1. **生产监控**: 实时监控高炉碳排放情况
2. **预测分析**: 基于生产参数预测碳排放
3. **智能诊断**: 使用SHAP分析特征对预测的具体影响
4. **仿真优化**: 通过参数调整模拟不同生产条件下的碳排放
5. **敏感性分析**: 识别对碳排放影响最大的关键参数
6. **优化建议**: 识别影响碳排放的关键因素并提供改进建议
7. **特征权重分析**: 深入分析潞安煤特性的影响权重
8. **报告生成**: 自动生成碳排放分析报告

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

本项目采用MIT许可证。

## 📞 联系方式

如有问题或建议，请通过以下方式联系:
- 邮箱: 390496404@qq.com

---

**注意**: 本项目仅用于研究和学习目的，实际生产环境使用前请进行充分测试和验证。 