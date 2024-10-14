"""
Common use tool for Streamlit UI & regression tool

"""


import streamlit as st
import pandas as pd
# import itertools

import datetime
import numpy as np
import io
import pickle


from sklearn.metrics import r2_score, mean_absolute_percentage_error, max_error, mean_squared_error



## UI download file/figure
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


def upload_file(uploaded_raw, url):

    if uploaded_raw is not None:
        up_file_type = uploaded_raw.name.split(".")[1]

        if up_file_type == "csv":
            df_raw = pd.read_csv(uploaded_raw, encoding="utf-8")
        elif up_file_type == "xlsx":
            df_raw = pd.read_excel(uploaded_raw)
        st.header('您所上傳的CSV檔內容：')

        # fac_n = df_fac.shape[1]
    else:
        st.header('未上傳檔案，以下為 Demo：')
        df_raw = pd.read_csv(url, encoding="utf-8")

    return df_raw


def convert_fig(fig):

    mybuff = io.StringIO()
   
    # fig_html = fig_pair.write_html(fig_file_name)
    fig.write_html(mybuff, include_plotlyjs='cdn')
    html_bytes = mybuff.getvalue().encode()

    return html_bytes


def download_file(name_label, button_label, file, file_type, gui_key):
    date = str(datetime.datetime.now()).split(" ")[0]
    if file_type == "csv":
        file = convert_df(file)
        mime_text = 'text/csv'
    elif file_type == "html":
        file = convert_fig(file)  
        mime_text = 'text/html'
    elif file_type == "pickle":
        file = pickle.dumps(file)
        mime_text = ""

    # file_name_col, button_col = st.columns(2)
    result_file = date + "_result"
    download_name = st.text_input(label=name_label, value=result_file, key=gui_key) 
    download_name = download_name + "." + file_type

    st.download_button(label=button_label,  
                    data=file, 
                    file_name=download_name,
                    mime=mime_text,
                    key=gui_key+"dl")
    


def reg_save(df_result, fig, model):
        st.markdown("---")

        download_file(name_label="Input Result File Name",
                      button_label='Download statistics result as CSV',
                      file=df_result,
                      file_type="csv",
                      gui_key="result_data"
                      )

        st.markdown("---")

        download_file(name_label="Input Figure File Name",
                      button_label='Download figure as HTML',
                      file=fig,
                      file_type="html",
                      gui_key="figure"
                      )
        
        st.markdown("---")

        download_file(name_label="Input Model File Name",
                      button_label='Download model as PICKLE',
                      file=model,
                      file_type="pickle",
                      gui_key="model"
                      )
        
        st.markdown("---")
## Model performance tool


def backend_pefmce(y, prediction, p):
  
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
