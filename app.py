#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
潞安煤高炉碳排放监控与管理系统 - Streamlit Web应用
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os
import matplotlib
import plotly
import plotly.graph_objects as go
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import os

# 字体文件路径
font_path = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansSC-Regular.otf")
if os.path.exists(font_path):
    my_font = fm.FontProperties(fname=font_path)
    # fm._rebuild()  # 强制重建字体缓存（新版matplotlib已无此方法，直接注释掉）
    plt.rcParams['font.sans-serif'] = [my_font.get_name()]
    plt.rcParams['axes.unicode_minus'] = False
else:
    my_font = None
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False 
from src.feature_name_map import en2zh
from create_dummy_data import calculate_carbon_emission_wsa

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_handler import DataHandler
from model_trainer import ModelTrainer

# 尝试导入SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP库未安装，智能诊断功能将不可用。请运行: pip install shap")

# 设置页面配置
st.set_page_config(
    page_title="潞安煤高炉碳排放监控与管理系统",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 插入自定义CSS美化tab横向分布
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 22px;
    }
    .stTabs [data-baseweb="tab"] {·
        font-size: 1.15em;
        font-weight: 500;
        color: #222;
        padding: 0.5em 1.5em;
    }
    .stTabs [aria-selected="true"] {
        color: #e53935 !important;
        border-bottom: 3px solid #e53935 !important;
        background: none !important;
    }
    .stTabs [aria-selected="false"]:hover {
        color: #1976d2 !important;
        background: #f5f5f5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 更新映射
if 'coal_ratio' in en2zh:
    en2zh['coal_ratio'] = '煤比'
if 'coke_ratio' in en2zh:
    en2zh['coke_ratio'] = '焦比'
if 'coke_consumption_total' in en2zh:
    en2zh['coke_consumption_total'] = '焦炭消耗'

@st.cache_data
def load_data():
    """
    缓存加载数据
    """
    try:
        data_handler = DataHandler("data/daily_production_data.csv")
        X, y = data_handler.process()
        
        if X is not None and y is not None and data_handler.data is not None:
            # 获取原始数据用于显示
            raw_data = data_handler.data.copy()
            raw_data['date'] = pd.to_datetime(raw_data['date'])
            raw_data.set_index('date', inplace=True)
            
            return X, y, raw_data, data_handler
        else:
            st.error("数据加载失败")
            return None, None, None, None
    except Exception as e:
        st.error(f"数据加载错误: {e}")
        return None, None, None, None

@st.cache_resource
def load_model():
    """
    缓存加载模型
    """
    try:
        # 检查模型文件是否存在
        model_path = "models/xgboost_carbon_model.json"
        if not os.path.exists(model_path):
            st.warning("模型文件不存在，正在训练新模型...")
            return train_new_model()
        
        # 加载已保存的模型
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        
        # 加载模型信息
        info_path = "models/model_info.json"
        if os.path.exists(info_path):
            import json
            with open(info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            return model, model_info
        else:
            return model, None
            
    except Exception as e:
        st.error(f"模型加载错误: {e}")
        return None, None

def train_new_model():
    """
    训练新模型
    """
    try:
        X, y, _, _ = load_data()
        if X is not None and y is not None:
            trainer = ModelTrainer(test_size=0.2, random_state=42)
            model = trainer.train(X, y)
            
            # 获取模型信息
            model_info = {
                'feature_names': trainer.feature_names,
                'test_size': trainer.test_size,
                'random_state': trainer.random_state
            }
            
            return model, model_info
        else:
            return None, None
    except Exception as e:
        st.error(f"模型训练错误: {e}")
        return None, None

def get_prediction_for_date(model, X, raw_data, selected_date):
    """
    获取指定日期的预测值和WSA公式实际值
    """
    try:
        if selected_date in X.index:
            features = X.loc[selected_date:selected_date]
            prediction = model.predict(features)[0]
            # 用WSA公式动态计算实际记录值
            actual = calculate_carbon_emission_wsa(features.iloc[0])
            return prediction, actual
        else:
            return None, None
    except Exception as e:
        st.error(f"预测错误: {e}")
        return None, None

def create_emission_chart(raw_data, selected_date):
    """
    创建碳排放趋势图
    """
    try:
        # 获取最近30天的数据
        end_date = selected_date
        start_date = end_date - timedelta(days=30)
        
        # 过滤数据
        chart_data = raw_data[(raw_data.index >= start_date) & (raw_data.index <= end_date)]
        
        if len(chart_data) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制碳排放趋势
            ax.plot(chart_data.index, chart_data['carbon_emission_co2'], 
                   marker='o', linewidth=2, markersize=6, label='实际碳排放')
            
            # 高亮显示选中日期
            if selected_date in chart_data.index:
                ax.scatter(selected_date, chart_data.loc[selected_date, 'carbon_emission_co2'], 
                          color='red', s=100, zorder=5, label='选中日期')
            
            ax.set_xlabel('日期', fontproperties=my_font)
            ax.set_ylabel('碳排放量 (吨CO2)', fontproperties=my_font)
            ax.set_title('最近30天碳排放趋势', fontproperties=my_font)
            ax.legend(prop=my_font)
            ax.grid(True, alpha=0.3)
            
            # 旋转x轴标签
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
        else:
            return None
    except Exception as e:
        st.error(f"图表创建错误: {e}")
        return None

def create_shap_waterfall(model, X, selected_date, feature_names):
    """
    创建SHAP瀑布图，特征名、标签、标题全部中文
    """
    try:
        import shap
        sample_data = X.loc[selected_date:selected_date]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_data)
        # 中文特征名
        feature_names_zh = [en2zh.get(f, f) for f in feature_names]
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=sample_data.iloc[0].values,
                feature_names=feature_names_zh
            ),
            show=False
        )
        plt.title(f'SHAP瀑布图 - {selected_date.strftime("%Y-%m-%d")} 碳排放预测解释', fontsize=14, fontweight='bold', fontproperties=my_font)
        plt.xlabel('SHAP值（对预测的影响）', fontsize=12, fontproperties=my_font)
        plt.tight_layout()
        return fig, None
    except Exception as e:
        return None, f"SHAP分析错误: {e}"

def get_shap_interpretation(shap_values, feature_names, sample_data):
    """
    获取SHAP解释文本
    """
    try:
        if not SHAP_AVAILABLE:
            return "SHAP库未安装，无法提供详细解释。"
        
        # 获取最重要的特征（按SHAP值绝对值排序）
        feature_importance = list(zip(feature_names, abs(shap_values[0])))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 获取前5个最重要的特征
        top_features = feature_importance[:5]
        
        interpretation = "## 🔍 智能诊断分析结果\n\n"
        interpretation += "### 主要影响因素分析：\n\n"
        
        for i, (feature, importance) in enumerate(top_features, 1):
            feature_zh = en2zh.get(feature, feature)
            feature_value = sample_data.iloc[0][feature]
            shap_value = shap_values[0][feature_names.index(feature)]
            
            # 根据SHAP值判断影响方向
            if shap_value > 0:
                direction = "增加"
                impact = "推高"
            else:
                direction = "减少"
                impact = "降低"
            
            interpretation += f"**{i}. {feature_zh}**\n"
            interpretation += f"- 当前值: {feature_value:.4f}\n"
            interpretation += f"- 影响程度: {abs(shap_value):.2f}\n"
            interpretation += f"- 影响方向: {direction}碳排放 ({impact}预测值)\n\n"
        
        # 总体解释
        total_impact = sum(shap_values[0])
        if total_impact > 0:
            overall_effect = "高于平均水平"
        else:
            overall_effect = "低于平均水平"
        
        interpretation += f"### 总体分析：\n"
        interpretation += f"当前生产条件下的碳排放预测值{overall_effect}。\n"
        interpretation += f"主要影响因素包括上述特征，其中{top_features[0][0]}的影响最为显著。\n\n"
        
        interpretation += "### 优化建议：\n"
        for feature, importance in top_features[:3]:
            feature_zh = en2zh.get(feature, feature)
            shap_value = shap_values[0][feature_names.index(feature)]
            if shap_value > 0:
                interpretation += f"- 考虑优化 **{feature_zh}** 以降低碳排放\n"
            else:
                interpretation += f"- 保持 **{feature_zh}** 在当前水平\n"
        
        return interpretation
        
    except Exception as e:
        return f"解释生成错误: {e}"

def create_simulation_data(X, selected_date, coal_ratio, luan_coal_ash_avg, blast_temp_avg):
    """
    创建仿真数据
    """
    try:
        if X is None or selected_date not in X.index:
            return None, None
        
        # 获取原始数据
        original_data = X.loc[selected_date:selected_date].copy()
        
        # 创建仿真数据
        simulation_data = original_data.copy()
        
        # 更新关键参数
        simulation_data['coal_ratio'] = coal_ratio
        simulation_data['luan_coal_ash_avg'] = luan_coal_ash_avg
        simulation_data['blast_temp_avg'] = blast_temp_avg
        
        # 更新相关的交叉特征
        simulation_data['volatility_effect'] = coal_ratio * simulation_data['luan_coal_vd_avg'].iloc[0]
        simulation_data['ash_burden_effect'] = coal_ratio * luan_coal_ash_avg
        simulation_data['effective_carbon_input'] = coal_ratio * simulation_data['luan_coal_fcad_avg'].iloc[0]
        
        return simulation_data, original_data
        
    except Exception as e:
        st.error(f"仿真数据创建错误: {e}")
        return None, None

def get_parameter_ranges(X):
    """
    获取参数范围
    """
    if X is None:
        return {
            'coal_ratio': {'min': 0.0, 'max': 1.0, 'current': 0.5},
            'luan_coal_ash_avg': {'min': 0.0, 'max': 50.0, 'current': 25.0},
            'blast_temp_avg': {'min': 800.0, 'max': 1200.0, 'current': 1000.0}
        }
    
    ranges = {
        'coal_ratio': {
            'min': float(X['coal_ratio'].min()),
            'max': float(X['coal_ratio'].max()),
            'current': float(X['coal_ratio'].mean())
        },
        'luan_coal_ash_avg': {
            'min': float(X['luan_coal_ash_avg'].min()),
            'max': float(X['luan_coal_ash_avg'].max()),
            'current': float(X['luan_coal_ash_avg'].mean())
        },
        'blast_temp_avg': {
            'min': float(X['blast_temp_avg'].min()),
            'max': float(X['blast_temp_avg'].max()),
            'current': float(X['blast_temp_avg'].mean())
        }
    }
    return ranges

def create_sensitivity_analysis(model, X, selected_date, param_name, param_range, steps=20):
    """
    创建参数敏感性分析
    """
    try:
        if X is None or selected_date not in X.index:
            return None, None
        # 获取原始数据
        original_data = X.loc[selected_date:selected_date].copy()
        # 创建参数变化序列
        param_values = np.linspace(param_range['min'], param_range['max'], steps)
        predictions = []
        for value in param_values:
            # 创建仿真数据
            simulation_data = original_data.copy()
            if param_name == 'coal_ratio':
                simulation_data[param_name] = value / 1000  # 还原为原始比值
                # 更新相关的交叉特征
                simulation_data['volatility_effect'] = (value / 1000) * simulation_data['luan_coal_vd_avg'].iloc[0]
                simulation_data['ash_burden_effect'] = (value / 1000) * simulation_data['luan_coal_ash_avg'].iloc[0]
                simulation_data['effective_carbon_input'] = (value / 1000) * simulation_data['luan_coal_fcad_avg'].iloc[0]
            elif param_name == 'luan_coal_ash_avg':
                simulation_data[param_name] = value
                simulation_data['ash_burden_effect'] = simulation_data['coal_ratio'].iloc[0] * value
            else:
                simulation_data[param_name] = value
            # 预测
            prediction = model.predict(simulation_data)[0]
            predictions.append(prediction)
        return param_values, predictions
    except Exception as e:
        st.error(f"敏感性分析错误: {e}")
        return None, None

def plot_sensitivity_analysis(param_values, predictions, param_name, original_value):
    """
    绘制敏感性分析图
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制敏感性曲线
        ax.plot(param_values, predictions, 'b-', linewidth=2, label='预测碳排放')
        
        # 标记原始值
        import numpy as np
        param_values_arr = np.array(param_values)
        original_prediction = predictions[np.argmin(np.abs(param_values_arr - original_value))]
        ax.axvline(x=original_value, color='r', linestyle='--', alpha=0.7, label='原始值')
        ax.axhline(y=original_prediction, color='g', linestyle='--', alpha=0.7, label='原始预测')
        
        # 设置标签和标题
        param_labels = {
            'coal_ratio': '煤比率',
            'luan_coal_ash_avg': '潞安煤灰分 (%)',
            'blast_temp_avg': '鼓风温度 (°C)'
        }
        
        xlabel = param_labels.get(param_name, str(param_name))
        ax.set_xlabel(xlabel, fontproperties=my_font)
        ax.set_ylabel('预测碳排放量 (吨CO2)', fontproperties=my_font)
        ax.set_title(f'{param_labels.get(param_name, param_name)}敏感性分析', fontproperties=my_font)
        ax.legend(prop=my_font)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"图表创建错误: {e}")
        return None

def main():
    """
    主函数
    """
    # 标题
    st.title("高炉碳排放监控与预测系统")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "碳排放监控",
        "智能诊断分析",
        "仿真与优化",
        "潞安煤特性影响权重分析",
        "能力分析及技术总结"
    ])

    with tab1:
        # 侧边栏：添加导入数据功能
        st.sidebar.header("📂 数据导入与管理")
        uploaded_file = st.sidebar.file_uploader("上传生产数据CSV文件（字段需与模板一致）", type=["csv"], help="请上传包含所有必需字段的CSV文件")
        if uploaded_file is not None:
            try:
                # 自动尝试utf-8和gbk编码
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='gbk')
                # 字段校验
                required_cols = [
                    'date', 'blast_temp_avg', 'blast_volume_total', 'coke_consumption_total',
                    'luan_coal_injection_total', 'luan_coal_ash_avg', 'luan_coal_vd_avg',
                    'luan_coal_fcad_avg', 'iron_output_total', 'carbon_emission_co2'
                ]
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    st.sidebar.error(f"缺少字段: {', '.join(missing)}，请检查模板！")
                else:
                    # 保存到data目录
                    save_path = os.path.join("data", "daily_production_data.csv")
                    df.to_csv(save_path, index=False)

                    # 自动清洗和补全carbon_emission_co2（使用全局import的WSA公式）
                    df = df.dropna(how='all', subset=df.columns[1:])
                    # 覆盖所有碳排放量为WSA公式计算值
                    df['carbon_emission_co2'] = df.apply(calculate_carbon_emission_wsa, axis=1)
                    df = df.dropna(subset=['carbon_emission_co2'])
                    df.to_csv(save_path, index=False)
                    st.sidebar.success("✅ 数据上传、清洗并补全成功！")
                    st.sidebar.info("⚠️ 如需加载新数据，请点击页面右上角菜单，选择'Clear cache'后刷新页面。")
                    # 选择是否重新训练模型
                    retrain = st.sidebar.checkbox("上传后立即重新训练模型", value=True)
                    if retrain:
                        with st.spinner("正在重新训练模型..."):
                            model, model_info = train_new_model()
                        st.sidebar.success("模型训练完成！请刷新页面体验新数据。")
                    else:
                        st.sidebar.info("数据已保存，如需生效请手动重新训练模型或刷新页面。")
            except Exception as e:
                st.sidebar.error(f"数据导入失败: {e}\n请确认文件编码为UTF-8或GBK，并检查字段格式是否正确。")
        
        # 侧边栏：添加强制刷新按钮
        if st.sidebar.button("🔄 数据刷新"):
            st.rerun()
    
        # 加载数据和模型
        with st.spinner("正在加载数据和模型..."):
            X, y, raw_data, data_handler = load_data()
            model, model_info = load_model()
        
        if X is None or model is None:
            st.error("系统初始化失败，请检查数据和模型文件")
            return
        
        # 侧边栏
        st.sidebar.header("📅 日期选择")
        
        # 获取可用日期范围
        if raw_data is not None:
            available_dates = raw_data.index.tolist()
            min_date = min(available_dates)
            max_date = max(available_dates)
        else:
            st.error("数据加载失败")
            return
        
        # 日期选择器
        selected_date = st.sidebar.date_input(
            "选择日期",
            value=max_date.date(),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
        
        # 转换为datetime
        selected_date = pd.Timestamp(selected_date)
        
        # 侧边栏显示系统信息
        st.sidebar.markdown("---")
        st.sidebar.header("📊 系统信息")
        if raw_data is not None:
            st.sidebar.metric("数据总天数", len(raw_data))
        if X is not None:
            st.sidebar.metric("特征数量", len(X.columns))
        st.sidebar.metric("模型类型", "XGBoost")
        
        # 主面板
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📈 碳排放监控")

            # 获取预测值和实际值（提前）
            prediction, actual = get_prediction_for_date(model, X, raw_data, selected_date)

            # 1. 健康度仪表盘（美化版）
            import plotly.graph_objects as go
            gauge_min = 8000
            gauge_green = 9100
            gauge_yellow = 9300
            gauge_max = 10000
            if prediction is not None:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    number={'font': {'size': 32, 'color': '#222'}, 'suffix': " 吨CO₂"},
                    title={'text': "<b>碳排放绩效</b>", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [gauge_min, gauge_max], 'tickwidth': 1, 'tickcolor': "#888"},
                        'bar': {'color': "#2b7cff", 'thickness': 0.25},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#eee",
                        'steps': [
                            {'range': [gauge_min, gauge_green], 'color': "#b6e3b6"},
                            {'range': [gauge_green, gauge_yellow], 'color': "#ffe699"},
                            {'range': [gauge_yellow, gauge_max], 'color': "#ffb3b3"}
                        ],
                        'threshold': {
                            'line': {'color': "#222", 'width': 4},
                            'thickness': 0.8,
                            'value': prediction
                        }
                    }
                ))
                fig.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10),
                    width=500,
                    height=350,
                    paper_bgcolor="white"
                )
                st.plotly_chart(fig, use_container_width=False)

            # 2. 指标卡片和当日生产数据表格（紧凑排版）
            prediction, actual = get_prediction_for_date(model, X, raw_data, selected_date)
            if prediction is not None and actual is not None:
                col1_1, col1_2, col1_3 = st.columns(3)
                with col1_1:
                    st.metric(
                        label="模型预测值（吨CO2）",
                        value=f"{prediction:.2f}",
                        delta=f"{prediction - actual:.2f}"
                    )
                with col1_2:
                    st.metric(
                        label="实际记录值（吨CO2）",
                        value=f"{actual:.2f}",
                        delta=None
                    )
                with col1_3:
                    error_rate = abs(prediction - actual) / actual * 100
                    st.metric(
                        label="预测误差率",
                        value=f"{error_rate:.2f}%",
                        delta=None
                    )
                # 当日生产数据表格（紧跟在指标卡片下方）
                st.subheader("📋 当日生产数据")
                if raw_data is not None and selected_date in raw_data.index:
                    daily_data = raw_data.loc[selected_date]
                    detail_items = [
                        ("鼓风温度 (°C)", f"{daily_data['blast_temp_avg']:.2f}"),
                        ("鼓风量 (m³)", f"{daily_data['blast_volume_total']:.2f}"),
                        ("焦炭消耗 (吨)", f"{daily_data['coke_consumption_total']:.2f}"),
                        ("潞安煤喷吹量 (吨)", f"{daily_data['luan_coal_injection_total']:.2f}"),
                        ("潞安煤灰分 (%)", f"{daily_data['luan_coal_ash_avg']:.2f}"),
                        ("潞安煤挥发分 (%)", f"{daily_data['luan_coal_vd_avg']:.2f}"),
                        ("潞安煤固定碳 (%)", f"{daily_data['luan_coal_fcad_avg']:.2f}"),
                        ("铁水产量 (吨)", f"{daily_data['iron_output_total']:.2f}")
                    ]
                    import numpy as np
                    if actual is not None and not (isinstance(actual, float) and np.isnan(actual)):
                        detail_items.append(("**碳排放量 (吨CO2)**", f"{actual:.2f}"))
                    detail_data = {
                        "指标": [item[0] for item in detail_items],
                        "数值": [item[1] for item in detail_items]
                    }
                    detail_df = pd.DataFrame(detail_data)
                    st.dataframe(detail_df, use_container_width=True)

    with col2:
        st.header("📊 趋势分析")
        
        # 创建趋势图
        fig = create_emission_chart(raw_data, selected_date)
        if fig is not None:
            st.pyplot(fig)
        
        # 3. 趋势分析一句话
        try:
            # 取最近7天碳排放量
            recent_dates = [selected_date - timedelta(days=i) for i in range(6, -1, -1)]
            recent_data = [raw_data.loc[d, 'carbon_emission_co2'] for d in recent_dates if d in raw_data.index]
            import numpy as np
            if len(recent_data) >= 2:
                trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
                if trend < 0:
                    trend_text = "过去7天，碳排放呈下降趋势。"
                else:
                    trend_text = "过去7天，碳排放呈上升趋势。"
                volatility = np.std(recent_data)
                if volatility > 100:
                    trend_text += "但波动较大。"
                else:
                    trend_text += "波动较小。"
                st.caption(trend_text)
        except Exception:
            pass
        
        # 显示特征重要性
        if model_info and 'feature_names' in model_info:
            st.subheader("🎯 特征重要性")
            
            # 获取特征重要性
            feature_importance = pd.DataFrame({
                'feature': model_info['feature_names'],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # 显示前5个重要特征
            top_features = feature_importance.head(5)
            for idx, row in top_features.iterrows():
                st.metric(
                        label=en2zh.get(row['feature'], row['feature']),
                    value=f"{row['importance']:.4f}"
                )
    
    with tab2:
        # 智能诊断分析模块
        st.markdown("---")
        st.header("🔍 智能诊断分析")
        
        # 检查SHAP是否可用
        if not SHAP_AVAILABLE:
            st.warning("⚠️ SHAP库未安装，智能诊断功能不可用。请运行: `pip install shap`")
        else:
            # 创建两列布局
            diag_col1, diag_col2 = st.columns([1, 1])
            
            with diag_col1:
                st.subheader("📊 SHAP瀑布图分析")
                # 智能诊断按钮
                if not st.button("🚀 开始智能诊断", type="primary", key="diagnose_btn_left"):
                    with st.spinner("正在进行智能诊断分析..."):
                        # 创建SHAP瀑布图
                        waterfall_fig, error_msg = create_shap_waterfall(
                            model, X, selected_date, X.columns.tolist()
                        )
                        if waterfall_fig is not None:
                            st.pyplot(waterfall_fig)
                            # 获取SHAP解释（增强版，前5个特征，含历史均值对比）
                            if selected_date in X.index:
                                sample_data = X.loc[selected_date:selected_date]
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer.shap_values(sample_data)
                                feature_names = X.columns.tolist()
                                # 取前5个重要特征
                                top_indices = np.argsort(np.abs(shap_values[0]))[::-1][:5]
                                diagnosis = []
                                for idx in top_indices:
                                    feature = feature_names[idx]
                                    current_value = sample_data.iloc[0][feature]
                                    shap_value = shap_values[0][idx]
                                    mean_value = X[feature].mean()
                                    # 对比文本
                                    if current_value > mean_value:
                                        compare_text = f"高于历史均值 {mean_value:.2f}"
                                    elif current_value < mean_value:
                                        compare_text = f"低于历史均值 {mean_value:.2f}"
                                    else:
                                        compare_text = "等于历史均值"
                                    # 影响方向
                                    if shap_value > 0:
                                        direction = "推高碳排放"
                                    else:
                                        direction = "降低碳排放"
                                    diagnosis.append(
                                        f"**{en2zh.get(feature, feature)}**：当前值 {current_value:.2f}，{compare_text}，"
                                        f"影响程度 {abs(shap_value):.2f}，方向：{direction}"
                                    )
                                st.markdown("### 主要影响因素详细解析：")
                                for item in diagnosis:
                                    st.markdown(item)
                        else:
                            st.error(f"诊断分析失败: {error_msg}")
            
            with diag_col2:
                if st.button("🚀 开始智能诊断", type="primary", key="diagnose_btn_right"):
                    st.subheader("📝 诊断说明")
                    st.markdown("""
                    **智能诊断功能说明：**
                    
                    🔍 **SHAP瀑布图**：显示每个特征对预测结果的具体贡献
                    
                    📊 **特征影响分析**：
                    - 正值：该特征推高了碳排放预测
                    - 负值：该特征降低了碳排放预测
                    
                    💡 **优化建议**：基于分析结果提供具体的改进建议
                    
                    ⚡ **点击"开始智能诊断"按钮开始分析**
                    """)
                    # 新增：右侧显示优化建议
                    if selected_date in X.index:
                        sample_data = X.loc[selected_date:selected_date]
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(sample_data)
                        feature_names = X.columns.tolist()
                        top_indices = np.argsort(np.abs(shap_values[0]))[::-1][:5]
                        st.markdown("### 优化建议：")
                        for idx in top_indices:
                            feature = feature_names[idx]
                            shap_value = shap_values[0][idx]
                            feature_zh = en2zh.get(feature, feature)
                            # 针对特定特征
                            if feature == "luan_coal_injection_total":
                                if shap_value < 0:
                                    st.info(f"增加{feature_zh}有助于降低碳排放")
                                # 如果shap_value为正，不显示任何与潞安煤喷吹量相关的建议
                            elif feature == "coke_consumption_total":
                                if shap_value > 0:
                                    st.warning(f"{feature_zh}增加会推高碳排放，建议优化")
                                else:
                                    st.info(f"降低{feature_zh}有助于减少碳排放")
                            else:
                                if shap_value > 0:
                                    st.info(f"建议适当降低{feature_zh}以减少碳排放")
                                else:
                                    st.info(f"当前{feature_zh}有助于降低碳排放，可保持")

    with tab3:
        # 仿真与优化模块
        st.markdown("---")
        st.header("🎛️ 仿真与优化")
        
        # 获取参数范围
        param_ranges = get_parameter_ranges(X)
        
        # 创建三列布局
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        
        with sim_col1:
            st.subheader("📊 参数调整")
            # 煤比（kg/吨铁水）滑块，范围110-210
            coal_ratio_min = 110
            coal_ratio_max = 210
            coal_ratio_current = int(param_ranges['coal_ratio']['current'] * 1000)
            coal_ratio_kg = st.slider(
                "煤比（kg/吨铁水）",
                min_value=coal_ratio_min,
                max_value=coal_ratio_max,
                value=int(X.loc[selected_date, 'coal_ratio'] * 1000) if selected_date in X.index else coal_ratio_current,
                step=1,
                help="调整煤喷吹量与铁水产量的比值，单位：kg/吨铁水"
            )
            coal_ratio = coal_ratio_kg / 1000  # 还原为原始比值
            # 焦比（kg/吨铁水）滑块，范围300-400
            coke_ratio_min = 300
            coke_ratio_max = 400
            coke_ratio_current = int(X.loc[selected_date, 'coke_ratio'] * 1000) if 'coke_ratio' in X.columns and selected_date in X.index else 350
            coke_ratio_kg = st.slider(
                "焦比（kg/吨铁水）",
                min_value=coke_ratio_min,
                max_value=coke_ratio_max,
                value=coke_ratio_current,
                step=1,
                help="调整焦炭消耗与铁水产量的比值，单位：kg/吨铁水"
            )
            coke_ratio = coke_ratio_kg / 1000
            # 灰分滑块，范围6-11
            luan_coal_ash_avg = st.slider(
                "潞安煤灰分 (%)",
                min_value=6.0,
                max_value=11.0,
                value=float(X.loc[selected_date, 'luan_coal_ash_avg']) if selected_date in X.index else param_ranges['luan_coal_ash_avg']['current'],
                step=0.1,
                help="调整潞安煤的平均灰分含量"
            )
            # 鼓风温度滑块，范围1150-1280
            blast_temp_avg = st.slider(
                "鼓风温度 (°C)",
                min_value=1150.0,
                max_value=1280.0,
                value=float(X.loc[selected_date, 'blast_temp_avg']) if selected_date in X.index else param_ranges['blast_temp_avg']['current'],
                step=1.0,
                help="调整平均鼓风温度"
            )
            # 预估成本变化（正负对称）
            # 煤比每减少1kg/吨铁水，成本-20元，增加+20元
            # 焦比每减少1kg/吨铁水，成本-30元，增加+30元
            # 风温每升高10度+5元，降低-5元
            coal_cost = (int(X.loc[selected_date, 'coal_ratio'] * 1000) - coal_ratio_kg) * 20
            coke_cost = (coke_ratio_kg - int(X.loc[selected_date, 'coke_ratio'] * 1000) if 'coke_ratio' in X.columns else 350) * 30
            temp_cost = (blast_temp_avg - float(X.loc[selected_date, 'blast_temp_avg'])) / 10 * 5
            total_cost = coal_cost + coke_cost + temp_cost
            st.markdown(f"**预估成本变化：** <span style='color:#2b7cff;font-size:1.2em;'>{total_cost:+.2f} 元</span> ", unsafe_allow_html=True)
        if st.button("🔄 重置为原始值"):
            st.rerun()
        
        with sim_col2:
            st.subheader("📈 预测对比")
            
            # 创建仿真数据
            if X is not None:
                simulation_data, original_data = create_simulation_data(
                    X, selected_date, coal_ratio, luan_coal_ash_avg, blast_temp_avg
                )
            else:
                simulation_data, original_data = None, None
            
            if simulation_data is not None and original_data is not None:
                # 获取原始预测值
                original_prediction = model.predict(original_data)[0]
                
                # 获取仿真预测值
                simulation_prediction = model.predict(simulation_data)[0]
                
                # 计算变化
                prediction_change = simulation_prediction - original_prediction
                change_percentage = (prediction_change / original_prediction) * 100
                
                # 显示预测结果
                st.metric(
                    label="原始预测值",
                    value=f"{original_prediction:.2f}",
                    delta=None
                )
                
                st.metric(
                    label="仿真预测值",
                    value=f"{simulation_prediction:.2f}",
                    delta=f"{prediction_change:.2f} ({change_percentage:+.2f}%)"
                )
                
                # 显示参数变化
                st.markdown("### 参数变化:")
                
                original_coal_ratio = float(original_data['coal_ratio'].iloc[0])
                original_ash = float(original_data['luan_coal_ash_avg'].iloc[0])
                original_temp = float(original_data['blast_temp_avg'].iloc[0])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("煤比（kg/吨铁水）", f"{coal_ratio_kg}", f"{coal_ratio_kg - int(original_coal_ratio * 1000):+d}")
                    st.metric("灰分 (%)", f"{luan_coal_ash_avg:.2f}", f"{luan_coal_ash_avg - original_ash:+.2f}")
                with col2:
                    st.metric("温度 (°C)", f"{blast_temp_avg:.1f}", f"{blast_temp_avg - original_temp:+.1f}")
        
        with sim_col3:
            st.subheader("💡 优化建议")
            
            if simulation_data is not None and original_data is not None:
                original_prediction = model.predict(original_data)[0]
                simulation_prediction = model.predict(simulation_data)[0]
                
                if simulation_prediction < original_prediction:
                    st.success("✅ 优化效果")
                    improvement = original_prediction - simulation_prediction
                    st.metric("碳排放减少", f"{improvement:.2f} 吨CO2")
                    
                    st.markdown("### 优化建议:")
                    st.markdown("""
                    🎯 **当前参数调整有效降低了碳排放**
                    
                    📋 **建议措施:**
                    - 保持当前参数设置
                    - 监控生产稳定性
                    - 评估成本效益
                    """)
                else:
                    st.warning("⚠️ 需要调整")
                    increase = simulation_prediction - original_prediction
                    st.metric("碳排放增加", f"{increase:.2f} 吨CO2")
                    
                    st.markdown("### 调整建议:")
                    st.markdown("""
                    🔧 **当前参数调整增加了碳排放**
                    
                    📋 **建议措施:**
                    - 降低煤比率
                    - 减少灰分含量
                    - 优化鼓风温度
                    - 寻找最佳平衡点
                    """)
            
            # 显示参数说明
            st.markdown("### 参数说明:")
            st.markdown("""
            **煤比率**: 影响燃料效率和燃烧特性
            
            **灰分含量**: 影响燃烧效率和炉渣形成
            
            **鼓风温度**: 影响燃烧反应和热效率
            """)
        
        # 敏感性分析
        st.markdown("---")
        st.subheader("📊 参数敏感性分析")
        
        # 敏感性分析参数去除焦炭消耗量
        st.markdown("---")
        sensitivity_param = st.selectbox(
            "选择要分析的参数",
            ["coal_ratio", "luan_coal_ash_avg", "blast_temp_avg"],
            format_func=lambda x: {
                    "coal_ratio": "煤比（kg/吨铁水）",
                "luan_coal_ash_avg": "潞安煤灰分",
                "blast_temp_avg": "鼓风温度"
            }[x]
        )
        if st.button("🔍 开始敏感性分析"):
            with st.spinner("正在进行敏感性分析..."):
                param_range = param_ranges[sensitivity_param] if sensitivity_param in param_ranges else None
                if sensitivity_param == 'coal_ratio':
                    param_range = {'min': 110, 'max': 210, 'current': int(X.loc[selected_date, 'coal_ratio'] * 1000) if selected_date in X.index else int(param_ranges['coal_ratio']['current'] * 1000)}
                # 其余参数不变
                param_values, predictions = create_sensitivity_analysis(
                    model, X, selected_date, sensitivity_param, param_range
                )
                if sensitivity_param == 'coal_ratio':
                    original_value = int(X.loc[selected_date, 'coal_ratio'] * 1000) if selected_date in X.index else param_range['current']
                else:
                    original_value = float(X.loc[selected_date, sensitivity_param]) if selected_date in X.index else param_range['current']
                # 后续绘图、指标等都用param_values
                if param_values is not None and predictions is not None:
                    # 绘制敏感性分析图
                    sensitivity_fig = plot_sensitivity_analysis(
                        param_values, predictions, sensitivity_param, original_value
                    )
                    if sensitivity_fig is not None:
                        st.pyplot(sensitivity_fig)
                        # 分析结果
                        st.markdown("### 📈 敏感性分析结果:")
                        # 找到最优值
                        min_prediction_idx = np.argmin(predictions)
                        optimal_value = param_values[min_prediction_idx]
                        min_prediction = predictions[min_prediction_idx]
                        # 计算敏感性
                        sensitivity = (max(predictions) - min(predictions)) / (param_range['max'] - param_range['min'])
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("最优值", f"{optimal_value:.1f}")
                        with col2:
                            st.metric("最低预测", f"{min_prediction:.2f}")
                        with col3:
                            st.metric("敏感性", f"{sensitivity:.2f}")
                        # 优化建议
                        st.markdown("### 💡 优化建议:")
                        import numpy as np
                        param_values_arr = np.array(param_values)
                        # 参数中文名和单位
                        param_zh = {
                            "coal_ratio": "煤比",
                            "luan_coal_ash_avg": "潞安煤灰分",
                            "blast_temp_avg": "鼓风温度"
                        }
                        param_unit = {
                            "coal_ratio": "kg/吨铁水",
                            "luan_coal_ash_avg": "%",
                            "blast_temp_avg": "°C"
                        }
                        # 合理范围
                        valid_range = {
                            "coal_ratio": (110, 210),
                            "luan_coal_ash_avg": (6.0, 11.0),
                            "blast_temp_avg": (1150.0, 1280.0)
                        }
                        if optimal_value != original_value and valid_range[sensitivity_param][0] <= optimal_value <= valid_range[sensitivity_param][1]:
                            improvement = predictions[np.argmin(np.abs(param_values_arr - original_value))] - min_prediction
                            st.success(f"🎯 将{param_zh[sensitivity_param]}调整到{optimal_value:.1f}{param_unit[sensitivity_param]}可减少碳排放{improvement:.2f}吨CO₂")
                        else:
                            st.info("✅ 当前参数设置已接近最优，无需调整")

    with tab4:
        # 潞安煤特性影响权重分析模块
        st.markdown("---")
        st.header("潞安煤特性影响权重分析")

        # 创建两列布局
        weight_col1, weight_col2 = st.columns([2, 1])

        with weight_col1:
            st.subheader("📊 特征重要性分析")
            # 获取特征重要性
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            # 中文名索引
            feature_importance['feature_zh'] = feature_importance['feature'].apply(lambda f: en2zh.get(f, f))
            # 用plotly绘制每个柱正上方有标签的条形图
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x=feature_importance['feature_zh'],
                y=feature_importance['importance'],
                marker_color='#2b7cff',
                text=[f"{v:.3f}" for v in feature_importance['importance']],
                textposition='outside'
            ))
            fig.update_layout(
                xaxis_title="特征",
                yaxis_title="重要性",
                margin=dict(l=10, r=10, t=60, b=30),
                height=350,
                yaxis_range=[0, max(feature_importance['importance']) * 1.15]
            )
            st.plotly_chart(fig, use_container_width=True)
            # 显示详细的特征重要性表格
            st.markdown("### 📋 特征重要性排名")
            # 为特征添加分类标签
            def categorize_feature(feature_name):
                if 'luan_coal' in feature_name:
                    return "潞安煤特性"
                elif 'ratio' in feature_name:
                    return "比率特征"
                elif 'effect' in feature_name or 'input' in feature_name:
                    return "交叉特征"
                elif 'blast' in feature_name:
                    return "鼓风参数"
                elif 'coke' in feature_name:
                    return "焦炭参数"
                elif 'iron' in feature_name:
                    return "产量参数"
                else:
                    return "其他参数"
            feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)
            # 显示前10个重要特征，全部用中文名
            top_features = feature_importance.head(10)
            for idx, row in top_features.iterrows():
                category_color = {
                    "潞安煤特性": "🔵",
                    "比率特征": "🟢", 
                    "交叉特征": "🟡",
                    "鼓风参数": "🟠",
                    "焦炭参数": "🔴",
                    "产量参数": "🟣",
                    "其他参数": "⚪"
                }
                category = str(row['category'])
                feature_zh = str(row['feature_zh'])
                st.markdown(f"{category_color.get(category, '⚪')} **{feature_zh}** ({category}) - 重要性: {row['importance']:.4f}")

        with weight_col2:
            st.subheader("🎯 潞安煤特性分析")
            # 计算潞安煤相关特征的重要性
            luan_features = feature_importance[feature_importance['category'].isin(['潞安煤特性', '交叉特征'])]
            luan_total_importance = luan_features['importance'].sum()
            total_importance = feature_importance['importance'].sum()
            luan_percentage = (luan_total_importance / total_importance) * 100
            # 权重说明问号
            st.markdown(
                f'<span style="font-size:1.1em;font-weight:bold;">潞安煤特性总权重</span>'
                f'<span title="此权重为所有与潞安煤相关的特征（包括原始特征和交叉特征）的重要性得分之和。" style="cursor: pointer; color: #888;">❓</span>',
                unsafe_allow_html=True
            )
            st.metric(
                label="",
                value=f"{luan_percentage:.1f}%",
                delta=f"{luan_total_importance:.4f}"
            )
            # 显示潞安煤特性详情，全部用中文名
            st.markdown("### 🔍 潞安煤特性详情:")
            for idx, row in luan_features.iterrows():
                importance_percent = (row['importance'] / total_importance) * 100
                st.markdown(f"**{row['feature_zh']}**: {importance_percent:.1f}%")
            # 交叉特征分析
            cross_features = feature_importance[feature_importance['category'] == '交叉特征']
            if len(cross_features) > 0:
                st.markdown("### 🧠 交叉特征分析:")
                cross_total = cross_features['importance'].sum()
                cross_percentage = (cross_total / total_importance) * 100
                st.metric(
                    label="交叉特征权重",
                    value=f"{cross_percentage:.1f}%",
                    delta=f"{cross_total:.4f}"
                )
                # 只展示一条中文公式及物理含义
                st.markdown("**交叉特征综合计算公式：**")
                st.markdown("- 交叉影响 = 煤比 × (挥发分 + 固定碳) - 煤比 × 灰分")
                st.markdown(
                    "> 物理含义：煤比提升时，煤中可燃组分（挥发分、固定碳）有助于降低碳排放，灰分则起稀释和负面作用。该公式综合反映煤质好坏对碳排放的正负影响。"
                )

    with tab5:
        # 能力分析及技术总结
        st.markdown("---")
        st.subheader("核心算法解析：从特征工程到智能决策")
        # 1. 创新特征工程：捕捉潞安煤的"指纹"
        st.markdown("### 1. 创新特征工程：捕捉潞安煤的'指纹'")
        col_feat1, col_feat2 = st.columns(2)
        with col_feat1:
            st.markdown("#### 基础特征的局限性")
            st.markdown("""
            - 仅用单一特征，无法捕捉它们之间的复杂交互作用。
            - 易导致模型误判，预测精度受限。
            """)
            st.image("data/feature_limit_demo.png")
            st.markdown("<span style='font-size:0.95em;color:#888;'>图片来源：scikit-learn官方文档，Permutation Importance vs Random Forest Feature Importance (MDI)，<a href='https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html' target='_blank'>[原文链接]</a></span>", unsafe_allow_html=True)
        with col_feat2:
            st.markdown("#### 引入冶金机理的交叉特征设计")
            st.markdown("为解决上述问题，本系统独创性地构建了以下交叉特征，将煤质特性与操作参数深度耦合：")
            st.latex(r'挥发分影响 = 煤比 \times 潞安煤挥发分')
            st.caption("量化挥发分在高喷吹下的燃烧促进效应")
            st.latex(r'灰分负荷 = 煤比 \times 潞安煤灰分')
            st.caption("量化灰分在高喷吹下带来的额外热耗和渣量负荷")
            st.latex(r'有效碳输入 = 煤比 \times 潞安煤固定碳')
            st.caption("量化固定碳输入对碳排放的直接贡献")
            st.markdown(
                "> 传统模型仅将煤质指标作为孤立输入，忽略了其在不同喷吹水平下的协同效应，导致预测精度瓶颈。"
            )
        # 2. 智能模型能力：从数据到洞察
        st.markdown("### 2. 智能模型能力：从数据到洞察")
        col_model1, col_model2 = st.columns(2)
        with col_model1:
            st.markdown("#### 模型选择与优势（XGBoost）")
            st.markdown("""
            我们选用业界领先的XGBoost模型，其核心优势在于：
            - 高效拟合高维、非线性工艺关系
            - 内生性处理多变量间的复杂交互作用
            - 输出可解释的特征贡献度，为归因分析提供依据
            """)
            st.markdown("证据：如右方特征重要性分析所示，本系统创建的灰分负荷效应等交叉特征在模型中占据主导地位，证明模型成功学习并验证了我们特征工程的有效性。")
        with col_model2:
            st.markdown("#### 学习效果验证")
            st.image("data/cross_feature_importance.png", caption="交叉特征在模型重要性排序中的主导地位", use_container_width=True)

        # 3. 转化应用价值：赋能生产决策
        st.markdown("### 3. 转化应用价值：赋能生产决策")
        st.markdown("""
        基于上述算法能力，本系统将复杂模型转化为对生产有直接指导意义的洞察：
        - 从数据中挖掘并量化关键工艺影响因素
        - 为操作优化提供量化依据和模拟推演
        - 赋能从经验驱动到数据驱动的决策模式转型
        """)
        # 核心技术优势
        st.markdown("---")
        st.subheader("核心技术优势")
        st.markdown("""
        1. 🏆 **深度机理理解 (Deep Mechanism Understanding)**  
        通过独创的交叉特征工程，系统不再是"黑箱"，而是能够深入理解并量化潞安煤特性（挥发分、灰分等）与生产参数（煤比）之间的复杂交互机理。  
        ➡️ 详见上方"创新特征工程"设计原理
        
        2. 📊 **智能归因分析 (Intelligent Attribution Analysis)**  
        基于XGBoost模型的可解释性，系统能够将抽象的碳排放波动，智能、准确地归因到具体的工艺参数上，实现了从"知其然"到"知其所以然"的跨越。  
        ➡️ 详见"智能诊断分析"中的SHAP瀑布图
        
        3. 🔮 **精准预测与优化 (Accurate Prediction & Optimization)**  
        得益于对机理的深度理解和强大的模型学习能力，系统不仅能实现高精度的碳排放预测，更能提供前瞻性的仿真优化，指导生产在成本和环保之间找到最佳平衡点。  
        ➡️ 详见"仿真与优化"模块的模拟结果
        """)

if __name__ == "__main__":
    main()
