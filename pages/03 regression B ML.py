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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# import graphviz
from sklearn.tree import export_graphviz

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    RANSACRegressor,
    TheilSenRegressor,
)

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold

from sklearn.metrics import make_scorer, r2_score
from sklearn.preprocessing import MinMaxScaler


import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

# import pickle
import tools





# color_sequence = ["#65BFA1", "#A4D6C1", "#D5EBE1", "#EBF5EC", "#00A0DF", "#81CDE4", "#BFD9E2"]
color_sequence = px.colors.qualitative.Pastel
template = "simple_white"


def compare_models(df_x, df_y, model_list, cv_splits=5):
    """
    Compare different regression models using StratifiedKFold and cross_val_score.

    Parameters:
        df_x (DataFrame): Features (X).
        df_y (Series): Target variable (y).
        models (dict): Dictionary of models to compare.
        cv_splits (int): Number of StratifiedKFold splits.

    Returns:
        DataFrame: Cross-validation scores for each model.
    """
    models = {}

    for i in model_list:
        if i == "Support Vector":
            models.update({"SVR-rbf": SVR(gamma="auto", kernel="rbf"),
                            "SVR-linear": SVR(gamma="auto", kernel="linear"),
                            "SVR-poly": SVR(gamma="auto", kernel="poly"),
                            "SVR-sigmoid": SVR(gamma="auto", kernel="sigmoid"),
                            })
        elif i == "Decision Tree":
            models.update({"Decision Tree-2": DecisionTreeRegressor(max_depth=2, random_state=42),
                            "Decision Tree-3": DecisionTreeRegressor(max_depth=3, random_state=42),
                            "Decision Tree-5": DecisionTreeRegressor(max_depth=5, random_state=42),
                            })
        elif i == "Random Forest":
            models.update({"Random Forest-20-2": RandomForestRegressor(n_estimators=20, max_depth=2, random_state=42),
                            "Random Forest-60-6": RandomForestRegressor(n_estimators=60, max_depth=6, random_state=42),
                            "Random Forest-100-2": RandomForestRegressor(n_estimators=100, max_depth=2, random_state=42),
                            "Random Forest-20-10": RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42),
                            "Random Forest-100-10": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                            })
        elif i == "K-Means":
            models.update({"K-Neighbors-3": KNeighborsRegressor(n_neighbors=3),
                            "K-Neighbors-5": KNeighborsRegressor(n_neighbors=5),
                            "K-Neighbors-10": KNeighborsRegressor(n_neighbors=10),
                            })
        elif i == "Linear Model":
            models.update({"Linear Regression": LinearRegression(),
                            "Huber": HuberRegressor(),
                            "RANSAC": RANSACRegressor(),
                            "TheilSen": TheilSenRegressor(),
                            })
        elif i == "Gaussian Process":
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
            kernel2 = RBF(length_scale=1.0)
            kernel3 = RBF(length_scale=1.0) + DotProduct(sigma_0=1.0)
            models.update({"Gaussian Process-rbf-white": GaussianProcessRegressor(kernel=kernel),
                           "Gaussian Process-rbf": GaussianProcessRegressor(kernel=kernel2),
                            "Gaussian Process-rbf-dot": GaussianProcessRegressor(kernel=kernel3),
                            })
    models
    
    results = {}
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    df_x_cv = tools.nom_checkbox(df_x, key="cv")[0]
    st_cv = st.button("Start Cross Validation", key="cv_button")
    if st_cv == True:
        for names, model in models.items():
            # Use cross_val_score with R² as the scoring metric
            names
            if "Gaussian Process" in names:
                # Use Gaussian Process with MinMaxScaler
                y_scaler = MinMaxScaler()
                y = y_scaler.fit_transform(df_y.values.reshape(-1, 1))
                df_y_cv = pd.DataFrame(y, columns=[df_y.name]).copy()
                # df_y = pd.DataFrame(y, columns=[df_y.name])
                # st.subheader("Normalize Y")
                # st.dataframe(y)
            else:
                df_y_cv = df_y.copy()
                df_x_cv = df_x.copy()
            # df_y_cv = df_y.copy()

            scores = cross_val_score(model, df_x_cv, df_y_cv, cv=kf, scoring=make_scorer(r2_score))
            results[names] = scores
            st.write(f"{names} R²: {scores.mean():.3f} ± {scores.std():.3f}")

    # Convert results to a DataFrame
    df_result = pd.DataFrame(results)

    return df_result

  
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
    st.markdown("---")
    x, nom_choice, df_nom = tools.nom_checkbox(df_x, key="nom_in_reg")
  

    st.markdown("---")
    # st.dataframe(pd.DataFrame(x).describe())
    # x
    gpr_std = None
    dict_yscarler = None
    if reg_type == "Support Vector":
        svr_ker = st.selectbox("Choose Kernal", ["linear", "rbf", "poly", "sigmoid"])
        reg = SVR(gamma='auto', kernel=svr_ker)

        
    elif reg_type == "Decision Tree":
        
        max_tree_layer = st.number_input("Setup Max Tree Layer", min_value=1, max_value=10, value=2)
        reg = DecisionTreeRegressor(max_depth=max_tree_layer, random_state=42)

    elif reg_type == "Random Forest":
        tree_num = st.number_input("Setup Max Tree Number", min_value=10, value=10)
        max_tree_layer = st.number_input("Setup Max Tree Layer", min_value=1, max_value=30, value=2)
        reg = RandomForestRegressor(n_estimators=tree_num, max_depth=max_tree_layer, random_state=42)

    elif reg_type == "K-Means":
        
        cluster_num = st.number_input("Setup Neighbor Q'ty", min_value=1, max_value=10, value=2)
        reg = KNeighborsRegressor(n_neighbors=cluster_num)
    # elif reg_type == "Navie Bayes":
    #     st.subheader("Under Construction")
    #     reg = GaussianNB()
    elif reg_type == "Gaussian Process":

        gpr_ker = st.selectbox("Choose Kernal", ["rbf+white_noise", "rbf", "rbf+dotproduct"])
        if gpr_ker == "rbf+white_noise":
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
        elif gpr_ker == "rbf":
            kernel = RBF(length_scale=1.0)
        elif gpr_ker == "rbf+dotproduct":
            kernel = RBF(length_scale=1.0) + DotProduct(sigma_0=1.0)
        
        dict_yscarler = {
            "Y_Max": df_y.max(),
            "Y_Min": df_y.min(),
            "Y_Delta": df_y.max() - df_y.min(),
            # "Y4": MinMaxScaler(feature_range=(0, 1)),
        }
        # dict_yscarler
        # y_name = df_y.name
        # y_name
        y_scaler = MinMaxScaler()
        y = y_scaler.fit_transform(df_y.values.reshape(-1, 1))

        # y
        df_y = pd.DataFrame(y, columns=[df_y.name])
        st.subheader("Normalize Y")
        st.dataframe(y)
        # kernel = RBF()
        # kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
        reg = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=10)

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
    if reg_type == "Gaussian Process":
        y_pred, gpr_std = reg.predict(x, return_std=True)
        st.write("GPR Std:")  
        st.dataframe(gpr_std)

    else:
        y_pred = reg.predict(x)
        
        # y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        # y_pred = pd.DataFrame(y_pred, columns=[df_y.name])
        # st.dataframe(y_pred)
    # y_pred = reg.predict(x)

    if reg_type == "Linear Model":
        if lin_type != "RANSAC":
            coef = reg.coef_
            st.markdown("Linear Model Coefficence")
            st.dataframe(coef)

        else:
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
    
    return reg, y_pred, nom_choice, df_nom, gpr_std, dict_yscarler


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
    x, nom_choice, df_nom = tools.nom_checkbox(df_x)

    if clf_type == "Support Vector":
        svc_ker = st.selectbox("Choose Kernal", ["linear", "rbf", "poly", "sigmoid"])
        clf = SVC(gamma='auto', kernel=svc_ker)
    elif clf_type == "Decision Tree":
        max_tree_layer = st.number_input("Setup Max Tree Layer", min_value=1, max_value=10, value=2)
        clf = DecisionTreeClassifier(max_depth=max_tree_layer)
    elif clf_type == "Random Forest":
        tree_num = st.number_input("Setup Max Tree Number", min_value=10, value=10)
        max_tree_layer = st.number_input("Setup Max Tree Layer", min_value=1, max_value=30, value=2)
        clf = RandomForestClassifier(n_estimators=tree_num, max_depth=max_tree_layer, random_state=42)
    
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

    
    return clf, y_pred, nom_choice, df_nom

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
        st.dataframe(df_reg.describe())
    
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
        cross_val_req = st.checkbox("Cross Validation", value=False, key="cross_val")
        if cross_val_req == True:
            st.subheader("Cross Validation")
            cross_val_mod = st.multiselect("Choose Cross Validation Compare Model", 
                                           ["Support Vector", "Decision Tree", "Random Forest", "K-Means", "Linear Model", "Gaussian Process"]
                                           )
            cross_val_num = st.number_input("Choose Cross Validation Number", value=5, min_value=2, max_value=10)
            df_cross_val = compare_models(df_x, df_y, model_list=cross_val_mod, cv_splits=cross_val_num)
            st.dataframe(df_cross_val)
            # cross_val_num = 5

            # reg.fit(x, df_y)
            # y_pred = reg.predict(x)

            # cv_score = cross_val_score(reg, df_x, df_y, cv=cross_val_num)
            # st.dataframe(cv_score)


        reg_type = st.selectbox("### Choose Regression Method", ["Support Vector", "Decision Tree", "Random Forest", "K-Means", "Gaussian Process", "Linear Model"])
        
        
        # reg_type = st.selectbox("### Choose Regression Method", ["Support Vector", "Decision Tree", "Random Forest", "K-Means", "Linear Model"])
        
        if reg_type == "Gaussian Process":
            reg, y_pred, nom_choice, df_nom, gpr_std, dict_yscarler = backend(df_x, df_y, reg_type)
            y_min = dict_yscarler["Y_Min"]
            # y_max = dict_yscarler["Y_Max"]    
            y_delta = dict_yscarler["Y_Delta"]
            df_y = (df_y - y_min)/y_delta
        else:
            reg, y_pred, nom_choice, df_nom, gpr_std, dict_yscarler = backend(df_x, df_y, reg_type)

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

        df_result=df_reg.copy()
        df_result["yhat"] = y_pred
        df_result["resid"] = df_y - y_pred
        fig_mod = tools.model_check_figure(df_result=df_result)
        st.plotly_chart(fig_mod, use_container_width=True)

        tools.reg_save(df_result, fig_mod, reg)


        predict_performance = st.checkbox("Predict New Data & Check Performance", key="reg")

        if predict_performance == True:
            st.subheader("**Predict Portion**")
        
            uploaded_df = st.file_uploader('#### 選擇您要上傳的CSV檔', type=["csv", "xlsx"], key="predict")

            if uploaded_df is None:
                
                url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTnqEkuIqYHm1eDDF-wHHyQ-Jm_cvmJuyBT4otEFt0ZE0A6FEQyg1tqpWTU0PXIFA_jYRX8O-G6SzU8/pub?gid=0&single=true&output=csv"

            else: 
                url = None
            df_test = tools.upload_file(uploaded_df, url)
            df_pred_x = df_test[factor]
            if nom_choice == True:
                    
                df_prd_nom = pd.DataFrame()
                for i in df_nom.columns:
                    # i
                    nom_mean = df_nom[i][0]
                    nom_std = df_nom[i][1]

                    df_prd_nom[i] = (df_pred_x[i] - nom_mean) / nom_std 
                df_pred_x = df_prd_nom.copy()
      
            y_hat = reg.predict(df_pred_x)
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

                uploaded_y = st.file_uploader('#### 選擇您要上傳的CSV檔', type=["csv", "xlsx"], key="up_y")

                if uploaded_y is None:
                    
                    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5skYRbfVPGE6RFYIM6Gg9QurH8u3h_RLfjt-CG0z5YgyxWEUTOdvoKmVkfWCLc2ECAuSEKaHVYPOA/pub?gid=0&single=true&output=csv"

                else: 
                    url = None

                df_y = tools.upload_file(uploaded_y, url)
                df_y
                select_list = list(df_raw.columns)
                y = st.selectbox("Please select real value", select_list)
                df_y = df_y[y]
                # df_y

                factor_number = st.number_input("Choose Factor Number", value=2, min_value=2)

                model_r2_score, adj_r_squared, model_rmse_score, model_mape_score, mape_series = tools.backend_pefmce(df_y, y_hat, factor_number)
                    
                st.markdown("#### $R^2$: %s" % round(model_r2_score, 3))
                st.markdown("               ")
                st.markdown("#### Adjusted $R^2$: %s" % round(adj_r_squared, 3))
                st.markdown("               ")
                st.markdown("#### RMSE: %s" % round(model_rmse_score, 3))
                st.markdown("               ")
                st.markdown("#### MAPE: %s %%" % round(model_mape_score*100, 1))

    if ana_type == "Classification":

        clf_type = st.selectbox("### Choose Classification Method", ["Support Vector", "Decision Tree", "Random Forest", "K-Means", "Navie Bayes", "Gaussian Process"])

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
