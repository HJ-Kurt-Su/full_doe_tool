#%% 匯入套件(包含我們寫好的function檔)
import pandas as pd 
import streamlit as st
import requests
from statsmodels.iolib.smpickle import load_pickle
import datetime
from io import BytesIO




@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

#%% 包成一整個 backend function: 主要資料處理及視覺化儀表板製作
def backend(uploaded_model, df_raw):
    model = load_pickle(uploaded_model)

    y_hat = model.predict(df_raw)
    df_raw["predict"] = y_hat

    return df_raw

#%% 頁面呈現
def main():
    st.title('Predict')

    # st.markdown("#### Author & License:")

    # st.markdown("**Kurt Su** (phononobserver@gmail.com)")

    # st.markdown("**This tool release under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license**")

    st.markdown("               ")
    st.markdown("               ")


    uploaded_model = st.sidebar.file_uploader("請上傳您的 Model 檔案", type=["pickle"])
    # uploaded_model
    uploaded_file = st.sidebar.file_uploader("請上傳您的 CSV 檔案", type=["csv"])

    


    if uploaded_file is not None:
        st.header('您所上傳的 CSV 檔內容：')
        df_raw = pd.read_csv(uploaded_file, encoding="utf-8")
    else:
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQjqLJWGJ46N6GkXkkMCcmthgALF9J28Bm1SGnwYUdmTpTn4Kq8tCzQ-G5FMddTZFF5YANEtPYese-g/pub?gid=0&single=true&output=csv"
        st.header('未上傳檔案，以下為 Demo：')
        df_raw = pd.read_csv(url, encoding="utf-8")
    
    st.dataframe(df_raw)
    
    if uploaded_model is not None:
        st.header('您所上傳的 Model 檔內容：')
        uploaded_model.name
    else:
        url_model = "https://raw.githubusercontent.com/HJ-Kurt-Su/full_doe_tool/main/2023-07-01_model.pickle"
        st.header('未上傳 Model，以下為 Demo：')
        uploaded_model = BytesIO(requests.get(url_model).content)
        # r
    
    df_predict = backend(uploaded_model, df_raw)

    st.header('預測結果：')
    st.dataframe(df_predict)

    csv = convert_df(df_predict)

    date = str(datetime.datetime.now()).split(" ")[0]
    result_file = date + "_predict.csv"

    st.download_button(label='Download predict result as CSV',  
                    data=csv, 
                    file_name=result_file,
                    mime='text/csv')


#%% Web App 頁面
if __name__ == '__main__':
    main()

# %%
