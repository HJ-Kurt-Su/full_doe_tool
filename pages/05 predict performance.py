import streamlit as st

import tools


def main():
    st.title("Predict")

    uploaded = st.sidebar.file_uploader("請上傳您的 CSV 檔案", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    df_raw = tools.load_table(uploaded, tools.DEMO_REAL_Y)
    st.dataframe(df_raw)

    cols = list(df_raw.columns)
    y = st.selectbox("Please select real value", cols)
    prediction = st.selectbox("Please select predict value", [c for c in cols if c != y])

    tools.show_perf(df_raw[y], df_raw[prediction], key="p05", default_p=1)


main()
