import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置随机种子以确保结果可重现
np.random.seed(42)
BASE_IRON = 3000  # 每天铁水产量基准值（吨）
DAYS = 150        # 天数，与数据长度一致
# 生成150天的日期序列
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(150)]

# 创建基础数据
data = {
    'date': dates,
    'blast_temp_avg': np.random.normal(1200, 50, 150),  # 平均鼓风温度 (°C)
    'blast_volume_total': np.random.normal(3000, 200, 150),  # 总鼓风量 (m³)
    'coke_consumption_total': np.random.normal(0.4 * BASE_IRON, 0.04 * BASE_IRON, DAYS),  # 总焦炭消耗 (吨)
    'luan_coal_injection_total': np.random.normal(0.15 * BASE_IRON, 0.02 * BASE_IRON, DAYS),  # 总潞安煤喷吹量 (吨)
    'luan_coal_ash_avg': np.random.normal(10, 1.0, 150),  # 潞安煤平均灰分 (%)
    'luan_coal_vd_avg': np.random.normal(11, 2.0, 150),  # 潞安煤平均挥发分 (%)
    'luan_coal_fcad_avg': np.random.normal(75.0, 3.0, 150),  # 潞安煤平均固定碳 (%)
    'iron_output_total': np.random.normal(BASE_IRON, 0.05 * BASE_IRON, DAYS)  # 总铁水产量 (吨)
}

# 创建DataFrame
df = pd.DataFrame(data)

# 添加一些趋势和季节性变化
for i in range(150):
    # 添加时间趋势
    trend_factor = 1 + 0.001 * i
    
    # 添加周期性变化（每周7天）
    weekly_cycle = 0.1 * np.sin(2 * np.pi * i / 7)
    
    # 应用趋势和周期性变化
    df.loc[i, 'blast_temp_avg'] *= (trend_factor + weekly_cycle)
    df.loc[i, 'blast_volume_total'] *= (trend_factor + weekly_cycle)
    df.loc[i, 'coke_consumption_total'] *= (trend_factor + weekly_cycle)
    df.loc[i, 'luan_coal_injection_total'] *= (trend_factor + weekly_cycle)
    df.loc[i, 'iron_output_total'] *= (trend_factor + weekly_cycle)

# 基于特征生成碳排放总量（包含非线性和交互作用）
def calculate_carbon_emission(row):
    # 基础碳排放系数
    base_emission = 2.5
    
    # 非线性关系
    temp_factor = 0.001 * (row['blast_temp_avg'] - 1200) ** 2
    volume_factor = 0.0001 * row['blast_volume_total'] ** 1.5
    
    # 交互作用
    coke_volume_interaction = 0.0002 * row['coke_consumption_total'] * row['blast_volume_total']
    coal_ash_interaction = 0.05 * row['luan_coal_injection_total'] * row['luan_coal_ash_avg']
    
    # 固定碳含量的影响
    fcad_factor = 0.02 * (80 - row['luan_coal_fcad_avg'])  # 固定碳含量越低，排放越高
    
    # 挥发分的影响
    vd_factor = 0.01 * row['luan_coal_vd_avg']
    
    # 计算总碳排放
    total_emission = (
        base_emission * row['iron_output_total'] +
        temp_factor * row['iron_output_total'] +
        volume_factor +
        coke_volume_interaction +
        coal_ash_interaction +
        fcad_factor * row['luan_coal_injection_total'] +
        vd_factor * row['luan_coal_injection_total']
    )
    
    # 添加一些随机噪声
    noise = np.random.normal(0, 0.05 * total_emission)
    
    return max(0, total_emission + noise)

def calculate_carbon_emission_wsa(row: pd.Series) -> float:
    """
    按世界钢铁协会（WSA）推荐的碳平衡经验公式计算每日总碳排放量。
    1. 焦炭带入的碳：coke_consumption_total * 0.855
    2. 潞安煤带入的碳：luan_coal_injection_total * 0.75
    3. 铁水吸收的碳：iron_output_total * 0.044
    4. 炉尘带走的碳：iron_output_total * 0.01
    5. 碳排放 = (焦炭碳 + 煤碳) - (铁水吸碳 + 炉尘碳)
    返回：每日总碳排放量（吨CO₂）
    """
    C_coke_in = row['coke_consumption_total'] * 0.855
    C_coal_in = row['luan_coal_injection_total'] * 0.75
    C_in_iron = row['iron_output_total'] * 0.044
    C_in_dust = row['iron_output_total'] * 0.01
    carbon_gas = (C_coke_in + C_coal_in) - (C_in_iron + C_in_dust)
    # noise = np.random.normal(0, 0.01 * abs(carbon_gas))  # 去除噪声
    return max(0, carbon_gas * 3.67)

# 计算碳排放（吨CO₂）
df['carbon_emission_co2'] = df.apply(calculate_carbon_emission_wsa, axis=1)

# 保存到CSV文件
output_path = 'data/daily_production_data.csv'
df.to_csv(output_path, index=False)

print(f"数据已保存到 {output_path}")
print(f"数据形状: {df.shape}")
print("\n数据预览:")
print(df.head())
print("\n数据统计信息:")
print(df.describe())

