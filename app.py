import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === 1. 加载模型与编码器 ===
dose_model = joblib.load("C:/Users/wyk/Desktop/predict_web/best_model_dose_Gradient Boosting.pkl")  # 请根据训练结果改成对应文件
drug_model = joblib.load("C:/Users/wyk/Desktop/predict_web/best_model_drug_XGBoost.pkl")             # 请根据训练结果改成对应文件
drug_encoder = joblib.load("C:/Users/wyk/Desktop/predict_web/drug_encoder.pkl")

# === 2. 页面设置 ===
st.set_page_config(page_title="Gn促排方案智能推荐系统", layout="centered")
st.title("👩‍⚕️ Gn促排个体化推荐系统")
st.markdown("请填写以下指标，系统将智能预测 Gn 起始剂量与推荐启动药品类型：")

# === 3. 用户输入 ===
age = st.number_input("年龄（岁）", min_value=18, max_value=50, value=30)
bmi = st.number_input("体重指数（BMI）", min_value=14.0, max_value=40.0, value=21.0, format="%.2f")
fsh = st.number_input("基础FSH（IU/L）", min_value=0.0, max_value=20.0, value=6.5, format="%.2f")
e2 = st.number_input("基础E2（pg/mL）", min_value=0.0, max_value=200.0, value=40.0, format="%.2f")
amh = st.number_input("基础AMH（ng/mL）", min_value=0.0, max_value=20.0, value=3.2, format="%.2f")

if st.button("立即预测"):
    # === 4. 构造输入数据 ===
    input_data = pd.DataFrame([[age, bmi, fsh, e2, amh]],
                              columns=["Age", "BMI", "(基础内分泌)FSH", "(基础内分泌)E2", "(基础内分泌)AMH"])
    
    # === 5. 执行模型预测 ===
    dose_pred = np.round(dose_model.predict(input_data)[0], 1)
    drug_pred = drug_encoder.inverse_transform(drug_model.predict(input_data))[0]
    
    # === 6. 显示结果 ===
    st.success("🎯 智能预测结果如下：")
    st.markdown(f"- 🧬 推荐 Gn 起始剂量：**{dose_pred} IU**")
    st.markdown(f"- 💊 推荐启动药品：**{drug_pred}**")

    # 结果表格形式展示
    result_df = pd.DataFrame({
        "Gn起始剂量 (IU)": [dose_pred],
        "推荐药品": [drug_pred]
    })
    st.table(result_df)
