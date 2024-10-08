import streamlit as st
import pandas as pd
# import itertools

import datetime
import numpy as np
import io
import pickle



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
