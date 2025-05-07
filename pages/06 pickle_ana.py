import pandas as pd 
import streamlit as st
import requests
from statsmodels.iolib.smpickle import load_pickle
import pickle
# import datetime
from io import BytesIO
import tools

def backend(uploaded_model):
    package = load_pickle(uploaded_model)
    pkl_key = package.keys()
    show_item = st.selectbox("請選擇要顯示的項目", pkl_key)
    if show_item == "model":
        st.write(package["model"])
    elif show_item == "features":
        st.write(package["features"])
    elif show_item == "df_nom":
        st.write(package["df_nom"])
    elif show_item == "df_result":
        st.write(package["df_result"])
    elif show_item == "df_imps":
        st.write(package["df_imps"])
    elif show_item == "df_perf":
        st.write(package["df_perf"])
    elif show_item == "fig_perf":
        st.plotly_chart(package["fig_perf"], use_container_width=True)
    elif show_item == "yscarler":
        st.write(package["yscarler"])
    else:
        st.markdown("所選項目尚未建立")



def main():
    st.title('Predict')

    # st.markdown("#### Author & License:")

    # st.markdown("**Kurt Su** (phononobserver@gmail.com)")

    # st.markdown("**This tool release under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license**")

    st.markdown("               ")
    st.markdown("               ")


    uploaded_model = st.sidebar.file_uploader("請上傳您的 Model 檔案", type=["pickle"])
    # uploaded_model
    # uploaded_raw = st.sidebar.file_uploader("請上傳您的 CSV 檔案", type=["csv", "xlsx"])

 

    if uploaded_model is not None:
        st.header('您所上傳的 Model 檔內容：')
        uploaded_model.name
    else:
        url_model = "https://raw.githubusercontent.com/HJ-Kurt-Su/full_doe_tool/main/2023-07-01_model.pickle"
        st.header('未上傳 Model，以下為 Demo：')
        uploaded_model = BytesIO(requests.get(url_model).content)
        # r

    st.markdown("---")

    df_predict = backend(uploaded_model)



if __name__ == '__main__':
    main()

