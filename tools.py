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


from sklearn.metrics import r2_score, mean_absolute_percentage_error, max_error, mean_squared_error, root_mean_squared_error


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, average_precision_score

import plotly.express as px



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

    # Calculate R-squared score using the actual and predicted values
    model_r2_score = r2_score(y, prediction)
    n = len(y)  # Number of observations

    # Calculate adjusted R-squared score
    adj_r_squared = 1 - (1 - model_r2_score) * (n-1)/(n-p-1)

    # Calculate Mean Absolute Percentage Error (MAPE)
    model_mape_score = mean_absolute_percentage_error(y, prediction)

    # Calculate individual MAPE values, adjusted to avoid division by zero
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(prediction - y) / np.maximum(np.abs(y), epsilon)
    mape_series = pd.Series(mape)  # Convert MAPE values to a pandas Series

    # Calculate Root Mean Squared Error (RMSE)
    model_rmse_score = root_mean_squared_error(y, prediction)

    # Return all calculated metrics
    return model_r2_score, adj_r_squared, model_rmse_score, model_mape_score, mape_series




def clf_score(y, y_predict, gui_key=None):
    """
    Calculate model metrics (accuracy, risk, ROC figure) and provide visualizations.
    Input:
      - y: actual Y
      - y_predict: predict Y
      - gui_key: GUI-related keys for threshold input (optional)
    Uses:
      - sklearn metrics module to calculate metrics
      - Plotly for visualization
    """

    # Compute the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds using ROC curve
    fpr, tpr, threshold = roc_curve(y, y_predict)
    roc_auc = auc(fpr, tpr)  # Compute Area Under the Curve (AUC)

    # Display the AUC value
    st.markdown("### AUC is: %s" % round(roc_auc, 4))

    # Create a dictionary of FPR, TPR, and thresholds
    dict_sum = {"FPR": fpr, "TPR": tpr, "Threshold": threshold}

    # Convert the dictionary to a pandas DataFrame
    df_roc_data = pd.DataFrame.from_dict(dict_sum)
    st.subheader("ROC Data")
    st.dataframe(df_roc_data)  # Display the ROC Data DataFrame

    fig_size = 640  # Set the figure size

    # Create a line plot for the ROC curve using Plotly
    fig_roc = px.line(df_roc_data, x="FPR", y="TPR", markers=False, labels={'label': 'roc_auc'}, 
                        range_x=[0, 1], range_y=[0, 1.1], width=fig_size, height=fig_size)
    
    st.subheader("ROC Figure")
    st.plotly_chart(fig_roc, use_container_width=True)  # Display the ROC curve plot

    if gui_key:  # If GUI keys are provided, handle threshold selection
        st.subheader("Accuracy Judge")
        key_in = st.checkbox("Key-in threshold", key=gui_key["threshold_check"])
        if key_in:
            threshold_cut = st.number_input("Key-in threshold value", min_value=0.0001, max_value=0.9999, value=0.5)
        else:
            threshold_cut = st.selectbox("Choose Threshold", threshold, key=gui_key["threshold"])

        y_pred_code = y_predict.map(lambda x: 1 if x >= threshold_cut else 0)
        # Calculate accuracy and confusion matrix with the threshold code
        acc = accuracy_score(y, y_pred_code)
        cof_mx = confusion_matrix(y, y_pred_code)
    else:
        # Calculate accuracy and confusion matrix
        acc = accuracy_score(y, y_predict)
        cof_mx = confusion_matrix(y, y_predict)

    df_cof_mx = pd.DataFrame(cof_mx)  # Convert confusion matrix to DataFrame
    # Rename columns and index for better readability
    df_cof_mx = df_cof_mx.rename(columns={0: "Predict Pass", 1: "Predict Fail"}, index={0: "Real Pass", 1: "Real Fail"})
    
    # Calculate risk value
    risk_qty = cof_mx[1, 0]
    risk = risk_qty / y_predict.size

    # Display accuracy, risk, and confusion matrix
    st.markdown("### Accuracy is: %s" % round(acc, 4))
    st.markdown("### Risk is: %s" % round(risk, 4))
    st.markdown("### Confusion Matrix:")
    st.dataframe(df_cof_mx)

    st.markdown("---")  # Add a horizontal rule

    preci_recall = st.checkbox("Use Precision & Recall Method")

    if preci_recall == True:

        precision, recall, thresholds = precision_recall_curve(y, y_predict)
        ap = average_precision_score(y, y_predict)
        pr_auc = auc(recall, precision) 
        st.markdown("### Average Precision is: %s" % round(ap, 4))
        st.markdown("### Precision-Recall AUC is: %s" % round(pr_auc, 4))

        # dict_pr = {"Precision": precision, "Recall": recall, "Threshold": thresholds}
        dict_pr = {"Precision": precision, "Recall": recall}

        # Convert the dictionary to a pandas DataFrame
        df_pr_data = pd.DataFrame.from_dict(dict_pr)
        st.dataframe(df_pr_data)
        # st.dataframe(recall)
        st.dataframe(thresholds)
        fig_pr = px.line(df_pr_data, x="Recall", y="Precision", markers=False, labels={'label': 'Precision-Recall'}, 
                        range_x=[0, 1], range_y=[0, 1.1], width=fig_size, height=fig_size)
    
        st.subheader("Precision Recall Figure")
        st.plotly_chart(fig_pr, use_container_width=True)  # Display the Precision-Recall curve plot


    return df_roc_data, fig_roc  # Return the ROC Data DataFrame and ROC figure plot



# def backend_pefmce(y, prediction, p):
  
#   # Target: calculate model metrics (r2, mape, rmse)
#   # Input: actual Y, predict Y, factor number
#   # Use sklearn metrice module to calculate metrics

#     model_r2_score = r2_score(y, prediction)
#     n = len(y)

#     adj_r_squared = 1 - (1 - model_r2_score) * (n-1)/(n-p-1)

#     model_mape_score = mean_absolute_percentage_error(y, prediction)

#     epsilon = np.finfo(np.float64).eps
#     mape = np.abs(prediction - y) / np.maximum(np.abs(y), epsilon)
#     mape_series = pd.Series(mape)

#     model_rmse_score = mean_squared_error(y, prediction, squared=False)

#     return model_r2_score, adj_r_squared, model_rmse_score, model_mape_score, mape_series
