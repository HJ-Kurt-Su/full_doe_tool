#%% 匯入套件(包含我們寫好的function檔)
import pandas as pd 
import streamlit as st
import requests
from statsmodels.iolib.smpickle import load_pickle
import pickle
# import datetime
from io import BytesIO
import tools
from sklearn.metrics import r2_score, mean_absolute_percentage_error, max_error, mean_squared_error
import numpy as np



# @st.cache_data
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv(index=False).encode('utf-8')

#%% 包成一整個 backend function: 主要資料處理及視覺化儀表板製作
def backend(uploaded_model, df_raw):
    package = load_pickle(uploaded_model)
    # with open(uploaded_model, "rb") as f:
    #     loaded_package = pickle.load(f)
    model = package["model"]
    model_type = type(model).__name__
    # model_type

    para_name = package["features"]
    df_nom = package["df_nom"]
     
    # model = loaded_package["model"]    
    # para_name = loaded_package["features"]
    # model = pickle.load(uploaded_model)["model"]
    # para_name = model.feature_names_in_
    # para_name = pickle.load(uploaded_model)["features"]
    # para_name = model.get_feature_names_out()
    # para_name = model.keys()
    st.markdown("#### Model Parameter")
    st.dataframe(para_name)

    

    if df_nom.shape[0] == 2:
        df_prd_nom = pd.DataFrame()
        for i in df_nom.columns:
            # i
            nom_mean = df_nom[i][0]
            nom_std = df_nom[i][1]

            df_prd_nom[i] = (df_raw[i] - nom_mean) / nom_std 
        df_input = df_prd_nom.copy()
    else:
        df_input = df_raw[para_name].copy()

    rslt_type =st.radio(
        "#### Choose Result Type", 
        ("Result Only", "Probability"))
    if rslt_type == "Probability":
        st.markdown("#### Probability Result")

        y_hat = model.predict_proba(df_input)
        df_raw["predict"] = y_hat[:, 1]
    else:
        # st.dataframe(df_input)
        y_hat = model.predict(df_input)
        if "GaussianProcessRegressor" in model_type:
            y_scale = package["yscarler"]
            y_delta = y_scale["Y_Delta"]
            y_min = y_scale["Y_Min"]

            # y_hat = y_hat[:, 1]
            df_raw["predict"] = y_hat * y_delta + y_min
        else:
            df_raw["predict"] = y_hat

    # fig_mod = package["fig_perf"]
    # st.plotly_chart(fig_mod, use_container_width=True)
    return df_raw


# def backend_pefmce(y, prediction, p):
#   # Target: calculate model metrics (r2, mape, rmse)
#   # Input: actual Y, predict Y, factor number
#   # Use sklearn metrice module to calculate metrics

#   model_r2_score = r2_score(y, prediction)
#   n = len(y)

#   adj_r_squared = 1 - (1 - model_r2_score) * (n-1)/(n-p-1)

#   model_mape_score = mean_absolute_percentage_error(y, prediction)

#   epsilon = np.finfo(np.float64).eps
#   mape = np.abs(prediction - y) / np.maximum(np.abs(y), epsilon)
#   mape_series = pd.Series(mape)

#   model_rmse_score = mean_squared_error(y, prediction, squared=False)

#   return model_r2_score, adj_r_squared, model_rmse_score, model_mape_score, mape_series


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
    uploaded_raw = st.sidebar.file_uploader("請上傳您的 CSV 檔案", type=["csv", "xlsx"])

    
    if uploaded_raw is not None:
        url = None
    else:
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQjqLJWGJ46N6GkXkkMCcmthgALF9J28Bm1SGnwYUdmTpTn4Kq8tCzQ-G5FMddTZFF5YANEtPYese-g/pub?gid=0&single=true&output=csv"

    df_raw = tools.upload_file(uploaded_raw, url)

    # if uploaded_file is not None:
    #     st.header('您所上傳的 CSV 檔內容：')
    #     df_raw = pd.read_csv(uploaded_file, encoding="utf-8")
    # else:
    #     url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQjqLJWGJ46N6GkXkkMCcmthgALF9J28Bm1SGnwYUdmTpTn4Kq8tCzQ-G5FMddTZFF5YANEtPYese-g/pub?gid=0&single=true&output=csv"
    #     st.header('未上傳檔案，以下為 Demo：')
    #     df_raw = pd.read_csv(url, encoding="utf-8")
    
    st.dataframe(df_raw)

    select_list = list(df_raw.columns)

    filter_req = st.checkbox('Filter Data')
    if filter_req == True:
        st.markdown("----------------")  
        st.markdown("#### Filter Parameter")
        filter_para = st.selectbox(
            "### Choose filter column", select_list)
        filter_sel = df_raw[filter_para].unique()
        filter_item = st.multiselect(
            "### Choose item", filter_sel,
        ) 

    if filter_req == True:
        df_prd = df_raw[df_raw[filter_para].isin(filter_item)].copy()
        st.markdown("----------------")  
        st.markdown("#### Filter DataFrame")
        df_prd
        st.markdown("----------------") 
    else:
        df_prd = df_raw.copy()
    
    if uploaded_model is not None:
        st.header('您所上傳的 Model 檔內容：')
        uploaded_model.name
    else:
        url_model = "https://raw.githubusercontent.com/HJ-Kurt-Su/full_doe_tool/main/2023-07-01_model.pickle"
        st.header('未上傳 Model，以下為 Demo：')
        uploaded_model = BytesIO(requests.get(url_model).content)
        # r

    # st.markdown("---")
    # st.subheader('Normalization')
    # nom_req = st.checkbox("Need Normalize Data")
    # if nom_req == True:
    #     nom_fac = st.file_uploader("請上傳您的 Normalization Factor 檔案", type=["csv", "xlsx"])
    #     if nom_fac is not None:
    #         url = None
    #     else:
    #         url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQjqLJWGJ46N6GkXkkMCcmthgALF9J28Bm1SGnwYUdmTpTn4Kq8tCzQ-G5FMddTZFF5YANEtPYese-g/pub?gid=0&single=true&output=csv"
    #     df_nom = tools.upload_file(nom_fac, url)
        
    #     st.dataframe(df_nom)

    #     df_prd_nom = pd.DataFrame()
    #     for i in df_nom.columns:
    #         # i
    #         nom_mean = df_nom[i][0]
    #         nom_std = df_nom[i][1]

    #         df_prd_nom[i] = (df_prd[i] - nom_mean) / nom_std 
    #     df_prd = df_prd_nom.copy()

    st.markdown("---")

    df_predict = backend(uploaded_model, df_prd)

    st.header('預測結果：')
    st.dataframe(df_predict)

    st.markdown("---")

    tools.download_file(name_label="Input Result File Name",
                      button_label='Download predict result as CSV',
                      file=df_predict,
                      file_type="csv",
                      gui_key="result_data"
                      )

    st.markdown("---")

    check_performance = st.checkbox("Check Predict Performance")
    
    if check_performance == True:

        real_y = st.radio(

            "**Real Y From:**",
            ["another upload", "original data"])

        
        uploaded_y = st.file_uploader('#### 選擇您要上傳的CSV檔', type=["csv", "xlsx"])
        if uploaded_y is None:
            
            url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5skYRbfVPGE6RFYIM6Gg9QurH8u3h_RLfjt-CG0z5YgyxWEUTOdvoKmVkfWCLc2ECAuSEKaHVYPOA/pub?gid=0&single=true&output=csv"
        else: 
            url = None
        df_tmp = tools.upload_file(uploaded_y, url)

        if uploaded_y is None:
            df_y = df_tmp["Y1"]
            df_yhat = df_tmp["yhat"]
            
        else:
            df_y = df_tmp.copy()
            df_yhat = df_predict["prediction"]
            


        factor_number = st.number_input("Choose Factor Number", value=2, min_value=2)

        model_r2_score, adj_r_squared, model_rmse_score, model_mape_score, mape_series = tools.backend_pefmce(df_y, df_yhat, factor_number)
            
        st.markdown("#### $R^2$: %s" % round(model_r2_score, 3))
        st.markdown("               ")
        st.markdown("#### Adjusted $R^2$: %s" % round(adj_r_squared, 3))
        st.markdown("               ")
        st.markdown("#### RMSE: %s" % round(model_rmse_score, 3))
        st.markdown("               ")
        st.markdown("#### MAPE: %s %%" % round(model_mape_score*100, 1))
            
        # st.header("Under constrcution performance too")



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
