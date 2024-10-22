import streamlit as st
import pandas as pd

# import numpy as np

# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix

# from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance


# from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
# import graphviz
from sklearn.tree import export_graphviz

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

# import pickle
import tools





# color_sequence = ["#65BFA1", "#A4D6C1", "#D5EBE1", "#EBF5EC", "#00A0DF", "#81CDE4", "#BFD9E2"]
color_sequence = px.colors.qualitative.Pastel
template = "simple_white"



# def clf_score_sklearn(y, y_predict):
#     # Target: calculate model metrics (accuracy, risk, roc figure)
#     # Input: actual Y, predict Y
#     # Use sklearn metrice module to calculate metrics


#     # thresh_key = gui_key["threshold"]
#     # thresh_check_key = gui_key["threshold_check"]

#     # Compute the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds using ROC curve
#     fpr, tpr, threshold = roc_curve(y, y_predict)
#     roc_auc = auc(fpr, tpr)  # Compute Area Under the Curve (AUC)

#     # Display the AUC value
#     st.markdown("### AUC is: %s" % round(roc_auc,4))

#     # Create a dictionary of FPR, TPR, and thresholds
#     dict_sum = {"FPR": fpr, "TPR": tpr, "Threshold": threshold}
#     # Convert the dictionary to a pandas DataFrame
#     df_roc_data = pd.DataFrame.from_dict(dict_sum)
#     st.subheader("ROC Data")
#     df_roc_data  # Display the ROC Data DataFrame
    
#     fig_size = 640  # Set the figure size

#     # Create a line plot for the ROC curve using Plotly
#     fig_roc = px.line(df_roc_data, x="FPR", y="TPR", markers=False, labels={'label': 'roc_auc'}, 
#                         range_x=[0, 1], range_y=[0, 1.1], width=fig_size, height=fig_size)
    
#     st.subheader("ROC Figure")
#     st.plotly_chart(fig_roc, use_container_width=True)  # Display the ROC curve plot

#     # Calculate accuracy and confusion matrix
#     acc = accuracy_score(y, y_predict)
#     cof_mx = confusion_matrix(y, y_predict)
#     df_cof_mx = pd.DataFrame(cof_mx)  # Convert confusion matrix to DataFrame
#     # Rename columns and index for better readability
#     df_cof_mx = df_cof_mx.rename(columns={0: "Predict Pass", 1: "Predict Fail"}, index={0: "Real Pass", 1: "Real Fail"})
    
#     # Calculate risk value
#     risk_qty = cof_mx[1, 0]
#     risk = risk_qty / y_predict.size

#     # Display accuracy, risk, and confusion matrix
#     st.markdown("### Accuracy is: %s" % round(acc,4))
#     st.markdown("### Risk is: %s" % round(risk,4))
#     st.markdown("### Confusion Matrix:")
#     st.dataframe(df_cof_mx)

#     st.markdown("---")  # Add a horizontal rule
#     return df_roc_data, fig_roc  # Return the ROC Data DataFrame and ROC figure plot



  
def backend(df_reg, formula, fig_size):
    # df_reg = df_raw.copy()
    st.markdown("Under Construction")

    # result, df_result, model = ols_reg(formula, df_reg)

    # alpha = 0.05
    # # f_num = len(result.tvalues)-1
    # # dof = round(f_num/3, 0)
    # dof = result.df_resid
    # t_val = stats.t.ppf(1-alpha/2, dof)

    # df_pareto = result.tvalues[1:].abs()
    # df_pareto = df_pareto.sort_values(ascending=True)
    # df_pareto = pd.DataFrame(df_pareto).reset_index(level=0)
    # df_pareto.columns = ["factor", "t-value"]


    # SW, sw_p_val = shapiro(df_result["resid"])
    # df_qq = acquire_qq_data(df_result["resid"])



    # fig = make_subplots(
    #     rows=2, cols=2,
    #     subplot_titles=("yhat-residual-plot (random better)", "residual-histogram-plot (normal distribution better)", 
    #                     "redidual-sequence-plot (random better)", "pareto-plot (red line as criteria)"))

    # fig.add_trace(go.Scatter(x=df_result["yhat"], y=df_result["resid"], mode="markers", 
    #                         marker=dict(color='rgba(19, 166, 255, 0.6)')),
    #             row=1, col=1)

    # fig.add_trace(go.Histogram(x=df_result["resid"],
    #                         marker=dict(color='rgba(19, 166, 255, 0.6)')),
    #             row=1, col=2)

    # fig.add_trace(go.Scatter(y=df_result["resid"], mode="lines+markers",
    #                         marker=dict(color='rgba(19, 166, 255, 0.6)')),
    #             row=2, col=1)

    # fig.add_trace(go.Bar(x=df_pareto["t-value"], y=df_pareto["factor"], orientation='h', width=0.8,
    #                     marker=dict(color='rgba(19, 166, 255, 0.6)')
    #                     ),
    #             row=2, col=2)
    # fig.add_vline(x=t_val, line_width=2, line_dash='dash', line_color='red',
    #             row=2, col=2)

    # # fig.add_trace(go.Scatter(x=df_qq["x_line"], y=df_qq["y_line"], mode="lines"),
    # #               row=2, col=2)

    # fig.update_xaxes(title_text="Y-hat", row=1, col=1)
    # fig.update_yaxes(title_text="Residual", row=1, col=1)

    # fig.update_xaxes(title_text="Residual", row=1, col=2)
    # fig.update_yaxes(title_text="Count", row=1, col=2)

    # fig.update_xaxes(title_text="Sequence", row=2, col=1)
    # fig.update_yaxes(title_text="Residual", row=2, col=1)

    # fig.update_xaxes(title_text="Factor Importance", row=2, col=2)
    # fig.update_yaxes(title_text="Factor", row=2, col=2)

    # fig.update_layout(height=fig_size[1], width=fig_size[0],
    #                 title_text="Model Check Figure",
    #                 showlegend=False)
 
    # return result, df_result, fig, sw_p_val

def backend_clf(df_x, df_y, clf_type):


    # if clf_type == "Support Vector" or "Decision Tree" or "K-Means" or "Navie Bayes":
        # clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    nom_choice = st.checkbox("Normalize Factor", value=False)
    if nom_choice == True:
        scaler = StandardScaler()
        x = scaler.fit_transform(df_x)
        st.subheader("Normalize X")
        st.dataframe(x)
    else:
        x = df_x
    
    if clf_type == "Support Vector":
        clf = SVC(gamma='auto', kernel="linear")
    elif clf_type == "Decision Tree":
        max_tree_layer = st.number_input("Setup Max Tree Layer", min_value=1, max_value=10, value=2)
        clf = DecisionTreeClassifier(max_depth=max_tree_layer)
    elif clf_type == "K-Means":
        st.subheader("Under Construction")
        cluster_num = st.number_input("Setup Cluster Q'ty", min_value=1, max_value=10, value=2)
        clf = KNeighborsClassifier(n_neighbors=cluster_num)
    elif clf_type == "Navie Bayes":
        clf = GaussianNB()
    elif clf_type == "Gaussian Process":
        kernel = 1.0 * RBF(1.0)
        clf = GaussianProcessClassifier(kernel=kernel)
    # clf = make_pipeline(StandardScaler(), DecisionTreeClassifier())
    clf.fit(x, df_y)
    y_pred = clf.predict(x)
    # proba = clf.predict_proba(x)
    # proba
    # clf.fit(df_x, df_y)
    # y_pred = clf.predict(df_x)
    # feature_importances = clf.feature_importances_
    if clf_type == "Support Vector":
        st.subheader("SVC Model Coefficence")
        coef_svc = clf.coef_
        coef_svc
    if clf_type == "Decision Tree":
        feature_importances = clf.feature_importances_
        # tree_rules = export_text(clf, feature_names=df_x.columns)
        # tree_rules
    # elif clf_type == "K-Means":
    #     DistanceList = []
    #     for i in range(1,11): #測試將資料分為1~10群
    #         KM = KNeighborsClassifier(n_neighbors=i)
    #         KM.fit(x, df_y)
    #         DistanceList.append(KM.inertia_) #求出每個Cluster內的資料與其中心點之平方距離和，並用List記錄起來
    #     st.dataframe(DistanceList)
        # st.markdown("K-Means Center")
        # centers = clf.cluster_centers_
        # centers

    imps = permutation_importance(clf, x, df_y)
    imps_mean = imps.importances_mean
    # imps_std = imps.importances_std
    # imps_mean
    # imps_std
    # df_x.columns
    # imps_mean
    df_imps = pd.DataFrame(imps_mean,
                  index=list(df_x.columns))
                #   columns=("Imp_Mean", "Imp_Std"))
    st.subheader("Feature Importance")
    # imps_mean
    st.dataframe(df_imps)
    # print(imps.importances_mean)
    if clf_type == "Decision Tree":
            
        st.subheader("Tree Native Feature Importance")
        df_tree_imps = pd.Series(feature_importances,
                  index=list(df_x.columns))
        st.dataframe(df_tree_imps)
    
    # importances = clf_mod.feature_importances_ 
    # importances
    # clf.coef_()

    
    return clf, y_pred

def main():

    fig_size = [1280, 960]

    st.title('ML Regression Tool')

    st.markdown("               ")
    st.markdown("               ")


    uploaded_raw = st.sidebar.file_uploader('#### 選擇您要上傳的CSV檔', type=["csv", "xlsx"])
    # uploaded_csv = st.file_uploader('#### 選擇您要上傳的CSV檔')

    ana_type = st.selectbox("Choose Analysis Mehthd", ["Regression Method", "Classification"])

    if ana_type == "Regression Method":
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTnqEkuIqYHm1eDDF-wHHyQ-Jm_cvmJuyBT4otEFt0ZE0A6FEQyg1tqpWTU0PXIFA_jYRX8O-G6SzU8/pub?gid=0&single=true&output=csv"

    elif ana_type == "Classification":
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSAL6t_HNdjVhKPyDzt-fVMHqT7ZZnWPrKLIY-QveQxF9vMbR-HbRwcDBM1MEyUjnHkC0JWKbL2TjP0/pub?gid=0&single=true&output=csv"

    else:
        url = None

    df_raw = tools.upload_file(uploaded_raw, url)

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
        st.markdown("----------------")  
    # if filter_req == True:
        df_reg = df_raw[df_raw[filter_para].isin(filter_item)].copy()
        st.markdown("----------------") 
        st.markdown("#### Filter DataFrame")
        df_reg
        st.markdown("----------------") 
    else:
        df_reg = df_raw.copy()
    
    # select_list
    response = st.selectbox("### Choose Response(y)", select_list)
    # response
    factor_list = select_list.copy()
    factor_list.remove(response)
    factor = st.multiselect(
        "### Choose Factor(x)", factor_list)
    if not factor:
        st.error("Please select at least one factor.")

    if uploaded_raw is None:
        st.header('未上傳檔案，以下為 Demo：')
        response = "Y2"
        factor = ["R", "r", "t"]

    if ana_type == "Regression Method":

        st.header("Under Construction")

        # # select_list
        # response = st.selectbox("### Choose Response(y)", select_list)
        # # response
        # factor_list = select_list.copy()
        # factor_list.remove(response)
        # factor = st.multiselect(
        #     "### Choose Factor(x)", factor_list)
        # if not factor:
        #     st.error("Please select at least one factor.")


        # if uploaded_raw is None:
        #     st.header('未上傳檔案，以下為 Demo：')
        #     response = "Y1"
        #     factor = ["R", "r", "t"]



        # st.markdown("---")

        # tools.download_file(name_label="Input Result File Name",
        #               button_label='Download statistics result as CSV',
        #               file=df_result,
        #               file_type="csv",
        #               gui_key="result_data"
        #               )

        # st.markdown("---")

        # tools.download_file(name_label="Input Figure File Name",
        #               button_label='Download figure as HTML',
        #               file=fig,
        #               file_type="html",
        #               gui_key="figure" 
        #               )
        
        # st.markdown("---")

        # tools.download_file(name_label="Input Model File Name",
        #               button_label='Download model as PICKLE',
        #               file=result,
        #               file_type="pickle",
        #               gui_key="model"
        #               )
        
        # st.markdown("---")



    if ana_type == "Classification":

        clf_type = st.selectbox("### Choose Classification Method", ["Support Vector", "Decision Tree", "K-Means", "Navie Bayes", "Gaussian Process"])

        df_x = df_reg[factor]
        df_y = df_reg[response]
        
        clf, y_pred = backend_clf(df_x, df_y, clf_type)

        if clf_type == "Decision Tree":

            g = export_graphviz(clf,
                                feature_names=df_x.columns,
                                class_names=["0","1"],
                                filled=True)

            st.subheader("Decision Tree")
            st.graphviz_chart(g)

            # st.plotly_chart(g, use_container_width=True)



        # df_roc_data, fig_roc = clf_score_sklearn(df_y, y_pred)
        df_roc_data, fig_roc = tools.clf_score(df_y, y_pred)

        tools.reg_save(df_roc_data, fig_roc, clf)


        predict_performance = st.checkbox("Predict New Data & Check Performance", key="clf")

        if predict_performance == True:
            st.subheader("**Predict Portion**")
        
            uploaded_df = st.file_uploader('#### 選擇您要上傳的CSV檔', type=["csv", "xlsx"], key="predict")

            if uploaded_df is None:
                
                url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTnqEkuIqYHm1eDDF-wHHyQ-Jm_cvmJuyBT4otEFt0ZE0A6FEQyg1tqpWTU0PXIFA_jYRX8O-G6SzU8/pub?gid=0&single=true&output=csv"

            else: 
                url = None
            df_test = tools.upload_file(uploaded_df, url)
            df_pred_x = df_test[factor]

            y_hat = clf.predict(df_pred_x)
            st.markdown("Predict Result:")
            
            df_test["predict"] = y_hat
            st.dataframe(df_test)
            
            tools.download_file(name_label="Input Predict File Name",
                            button_label='Download predict result as CSV',
                            file=df_test,
                            file_type="csv",
                            gui_key="predict_data"
                            )

            st.markdown("---")

            check_pefmce = st.checkbox("Check Model Performance")

            if check_pefmce == True:
                st.subheader("**Model Perfomance Portion**")
                st.subheader("**Under Construction**")

                uploaded_y = st.file_uploader('#### 選擇您要上傳的CSV檔', type=["csv", "xlsx"], key="up_y")

                if uploaded_y is None:
                    
                    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5skYRbfVPGE6RFYIM6Gg9QurH8u3h_RLfjt-CG0z5YgyxWEUTOdvoKmVkfWCLc2ECAuSEKaHVYPOA/pub?gid=0&single=true&output=csv"

                else: 
                    url = None

                df_tmp = tools.upload_file(uploaded_y, url)
                # df_y
                select_list = list(df_raw.columns)
                y = st.selectbox("Please select real value", select_list)
                df_y = df_tmp[y]

                tools.clf_score(df_y, y_hat)


#%% Web App 頁面
if __name__ == '__main__':
    main()

# %%
