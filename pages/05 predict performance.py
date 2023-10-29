#%% 匯入套件(包含我們寫好的function檔)
import pandas as pd 
import streamlit as st
import requests
# from statsmodels.iolib.smpickle import load_pickle
import datetime
from io import BytesIO
from sklearn.metrics import r2_score, mean_absolute_percentage_error, max_error, mean_squared_error
import numpy as np




@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


def backend(y, prediction, p):
  # Target: calculate model metrics (r2, mape, rmse)
  # Input: actual Y, predict Y, factor number
  # Use sklearn metrice module to calculate metrics

  model_r2_score = r2_score(y, prediction)
  n = len(y)

  adj_r_squared = 1 - (1 - model_r2_score) * (n-1)/(n-p-1)

  model_mape_score = mean_absolute_percentage_error(y, prediction)

  epsilon = np.finfo(np.float64).eps
  mape = np.abs(prediction - y) / np.maximum(np.abs(y), epsilon)
  mape_series = pd.Series(mape)

  model_rmse_score = mean_squared_error(y, prediction, squared=False)

  return model_r2_score, adj_r_squared, model_rmse_score, model_mape_score, mape_series


#%% 包成一整個 backend function: 主要資料處理及視覺化儀表板製作
# def backend(uploaded_model, df_raw):
#     model = load_pickle(uploaded_model)

#     y_hat = model.predict(df_raw)
#     df_raw["predict"] = y_hat

#     return df_raw

#%% 頁面呈現
def main():
    st.title('Predict')

    # st.markdown("#### Author & License:")

    # st.markdown("**Kurt Su** (phononobserver@gmail.com)")

    # st.markdown("**This tool release under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license**")

    st.markdown("               ")
    st.markdown("               ")


    # uploaded_model = st.sidebar.file_uploader("請上傳您的 Model 檔案", type=["pickle"])
    # uploaded_model
    uploaded_file = st.sidebar.file_uploader("請上傳您的 CSV 檔案", type=["csv"])

    


    if uploaded_file is not None:
        st.header('您所上傳的 CSV 檔內容：')
        df_raw = pd.read_csv(uploaded_file, encoding="utf-8")
    else:
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5skYRbfVPGE6RFYIM6Gg9QurH8u3h_RLfjt-CG0z5YgyxWEUTOdvoKmVkfWCLc2ECAuSEKaHVYPOA/pub?gid=0&single=true&output=csv"
        st.header('未上傳檔案，以下為 Demo：')
        df_raw = pd.read_csv(url, encoding="utf-8")
    
    st.dataframe(df_raw)

    select_list = list(df_raw.columns)

    y = st.selectbox("Please select real value", select_list)

    predict_list = select_list.copy()
    predict_list.remove(y)

    prediction = st.selectbox("Please select predict value", predict_list)
    factor_number = st.number_input("Choose Factor Number", value=1, min_value=1)

    # factor_number = 3

    model_r2_score, adj_r_squared, model_rmse_score, model_mape_score, mape_series = backend(df_raw[y], df_raw[prediction], factor_number)
    
    st.markdown("#### $R^2$: %s" % round(model_r2_score, 3))
    st.markdown("               ")
    st.markdown("#### Adjusted $R^2$: %s" % round(adj_r_squared, 3))
    st.markdown("               ")
    st.markdown("#### RMSE: %s" % round(model_rmse_score, 3))
    st.markdown("               ")
    st.markdown("#### MAPE: %s %%" % round(model_mape_score*100, 1))

    # st.header('預測結果：')
    # st.dataframe(df_predict)

    # csv = convert_df(df_predict)

    # date = str(datetime.datetime.now()).split(" ")[0]
    # result_file = date + "_predict.csv"

    # st.download_button(label='Download predict result as CSV',  
    #                 data=csv, 
    #                 file_name=result_file,
    #                 mime='text/csv')


#%% Web App 頁面
if __name__ == '__main__':
    main()

# %%