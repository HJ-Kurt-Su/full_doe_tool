import streamlit as st
import pandas as pd

# import numpy as np

# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix

# from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance


# from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, export_text, DecisionTreeRegressor
# import graphviz
from sklearn.tree import export_graphviz

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    RANSACRegressor,
    TheilSenRegressor,
)


import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

# import pickle
import tools





# color_sequence = ["#65BFA1", "#A4D6C1", "#D5EBE1", "#EBF5EC", "#00A0DF", "#81CDE4", "#BFD9E2"]
color_sequence = px.colors.qualitative.Pastel
template = "simple_white"


  
def backend(df_x, df_y, reg_type):
    # df_reg = df_raw.copy()
    # st.markdown("Under Construction")

    # nom_choice = st.checkbox("Normalize Factor", value=False)
    # if nom_choice == True:
    #     scaler = StandardScaler()
    #     x = scaler.fit_transform(df_x)
    #     st.subheader("Normalize X")
    #     st.dataframe(x)
    # else:
    #     x = df_x
    # df_x
    x = tools.nom_checkbox(df_x)
    # x

    if reg_type == "Support Vector":
        svr_ker = st.selectbox("Choose Kernal", ["linear", "rbf", "poly", "sigmoid"])
        reg = SVR(gamma='auto', kernel=svr_ker)
    elif reg_type == "Decision Tree":
        max_tree_layer = st.number_input("Setup Max Tree Layer", min_value=1, max_value=10, value=2)
        reg = DecisionTreeRegressor(max_depth=max_tree_layer)
    elif reg_type == "K-Means":
        
        cluster_num = st.number_input("Setup Neighbor Q'ty", min_value=1, max_value=10, value=2)
        reg = KNeighborsRegressor(n_neighbors=cluster_num)
    # elif reg_type == "Navie Bayes":
    #     st.subheader("Under Construction")
    #     reg = GaussianNB()
    elif reg_type == "Gaussian Process":
        kernel = 1.0 * RBF(1.0)
        reg = GaussianProcessRegressor(kernel=kernel)

    elif reg_type == "Linear Model":
        lin_type = st.selectbox("Choose Model", ["OLS", "Huber", "RANSAC", "TheilSen"])
        if lin_type == "OLS":
            reg = LinearRegression()

        elif lin_type == "Huber":
            reg = HuberRegressor()

        elif lin_type == "RANSAC":
            reg = RANSACRegressor()

        elif lin_type == "TheilSen":
            reg = TheilSenRegressor()


    # clf = make_pipeline(StandardScaler(), DecisionTreeClassifier())

    reg.fit(x, df_y)
    y_pred = reg.predict(x)

    if reg_type == "Linear Model":
        if lin_type != "RANSAC":
            coef = reg.coef_
            st.dataframe(coef)

        elif lin_type == "RANSAC":
            coef = reg.estimator_.coef_
            st.dataframe(coef)


    if reg_type == "Decision Tree":
        feature_importances = reg.feature_importances_

    imps = permutation_importance(reg, x, df_y)
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

    if reg_type == "Decision Tree":
            
        st.subheader("Tree Native Feature Importance")
        df_tree_imps = pd.Series(feature_importances,
                  index=list(df_x.columns))
        st.dataframe(df_tree_imps)
    
    return reg, y_pred


def backend_clf(df_x, df_y, clf_type):


    # if clf_type == "Support Vector" or "Decision Tree" or "K-Means" or "Navie Bayes":
        # clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    # nom_choice = st.checkbox("Normalize Factor", value=False)
    # if nom_choice == True:
    #     scaler = StandardScaler()
    #     x = scaler.fit_transform(df_x)
    #     st.subheader("Normalize X")
    #     st.dataframe(x)
    # else:
    #     x = df_x
    x = tools.nom_checkbox(df_x)

    if clf_type == "Support Vector":
        svc_ker = st.selectbox("Choose Kernal", ["linear", "rbf", "poly", "sigmoid"])
        clf = SVC(gamma='auto', kernel=svc_ker)
    elif clf_type == "Decision Tree":
        max_tree_layer = st.number_input("Setup Max Tree Layer", min_value=1, max_value=10, value=2)
        clf = DecisionTreeClassifier(max_depth=max_tree_layer)
    elif clf_type == "K-Means":
        # st.subheader("Under Construction")
        cluster_num = st.number_input("Setup Neighbor Q'ty", min_value=1, max_value=10, value=2)
        clf = KNeighborsClassifier(n_neighbors=cluster_num)
    elif clf_type == "Navie Bayes":
        nb_method = st.selectbox("Choose Method", ["Gaussian", "Bernoulli", "Multinomial"])
        if nb_method == "Gaussian":
            clf = GaussianNB()
        elif nb_method == "Bernoulli":
            clf = BernoulliNB()
        elif nb_method == "Multinomial":
            clf = MultinomialNB()
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
    if clf_type == "Support Vector" and svc_ker == "linear":
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

    df_x = df_reg[factor]
    df_y = df_reg[response]

    # nom_choice = st.checkbox("Normalize Factor", value=False, key="normalize")
    # if nom_choice == True:
    #     scaler = StandardScaler()
    #     x = scaler.fit_transform(df_x)
    #     st.subheader("Normalize X")
    #     st.dataframe(x)
    # else:
    #     x = df_x

    if ana_type == "Regression Method":

        # st.header("Under Construction")
        reg_type = st.selectbox("### Choose Regression Method", ["Support Vector", "Decision Tree", "K-Means", "Gaussian Process", "Linear Model"])


        
        reg, y_pred = backend(df_x, df_y, reg_type)

        if reg_type == "Decision Tree":

            g = export_graphviz(reg,
                                feature_names=df_x.columns,
                                class_names=["0","1"],
                                filled=True)

            st.subheader("Decision Tree")
            st.graphviz_chart(g)

        factor_number = df_x.shape[1]
        # factor_number

        model_r2_score, adj_r_squared, model_rmse_score, model_mape_score, mape_series = tools.backend_pefmce(df_y, y_pred, factor_number)

        st.markdown("#### $R^2$: %s" % round(model_r2_score, 3))
        st.markdown("               ")
        st.markdown("#### Adjusted $R^2$: %s" % round(adj_r_squared, 3))
        st.markdown("               ")
        st.markdown("#### RMSE: %s" % round(model_rmse_score, 3))
        st.markdown("               ")
        st.markdown("#### MAPE: %s %%" % round(model_mape_score*100, 1))

        st.subheader("Model Performance Figure")

        df_result=df_raw.copy()
        df_result["yhat"] = y_pred
        df_result["resid"] = df_y - y_pred
        fig_mod = tools.model_check_figure(df_result=df_result)
        st.plotly_chart(fig_mod, use_container_width=True)

        tools.reg_save(df_result, fig_mod, reg)

    if ana_type == "Classification":

        clf_type = st.selectbox("### Choose Classification Method", ["Support Vector", "Decision Tree", "K-Means", "Navie Bayes", "Gaussian Process"])

        # df_x = df_reg[factor]
        # df_y = df_reg[response]
        
        clf, y_pred = backend_clf(df_x, df_y, clf_type)
        # y_pred

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
                # st.subheader("**Under Construction**")

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
