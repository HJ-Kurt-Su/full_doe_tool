import streamlit as st
import pandas as pd

import numpy as np


from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures



from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, export_text, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
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
    LogisticRegression,

)

from xgboost import XGBRegressor, XGBClassifier
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold

from sklearn.metrics import make_scorer, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


import plotly.express as px

import tools





# color_sequence = ["#65BFA1", "#A4D6C1", "#D5EBE1", "#EBF5EC", "#00A0DF", "#81CDE4", "#BFD9E2"]
color_sequence = px.colors.qualitative.Pastel
template = "simple_white"



def run_clustering(df_x, method, n_clusters=3, eps=0.5, min_samples=5):
    """
    執行分群演算法並回傳分群標籤
    """
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(df_x)
        sse = []
        max_k = 15
        for k in range(1, max_k):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_x)
            sse.append(kmeans.inertia_)

        # 轉換為 DataFrame 供 plotly 使用   
        df_elbow = pd.DataFrame({'K': range(1, max_k), 'SSE': sse})
        df_elbow["SSE_ratio"] = df_elbow["SSE"].pct_change()
        df_elbow["SSE_div_first"] = df_elbow["SSE"] / df_elbow["SSE"].iloc[0]
        df_elbow

        # 繪製手肘圖
        fig = px.line(df_elbow, x='K', y='SSE', markers=True,
                    title='Elbow Method for Optimal K',
                    labels={'SSE': 'Sum of Squared Errors'},
                    template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
    elif method == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(df_x)
    elif method == "Spectral Clustering":
        model = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans', random_state=42)
        labels = model.fit_predict(df_x)
    elif method == "Agglomerative Clustering":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(df_x)
    else:
        labels = np.zeros(df_x.shape[0])
    return labels

def plot_clusters(df_x, labels):
    """
    使用 Plotly 可視化分群結果（僅支援2D/3D）
    """
    df_vis = df_x.copy()
    df_vis["cluster"] = labels.astype(str)
    if df_x.shape[1] >= 3:
        fig = px.scatter_3d(df_vis, x=df_x.columns[0], y=df_x.columns[1], z=df_x.columns[2], color="cluster", symbol="cluster")
    elif df_x.shape[1] == 2:
        fig = px.scatter(df_vis, x=df_x.columns[0], y=df_x.columns[1], color="cluster", symbol="cluster")
    else:
        fig = px.scatter(df_vis, x=df_x.columns[0], y=[0]*len(df_x), color="cluster", symbol="cluster")
    return fig


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
                            "Degree-2": Pipeline([('poly_features', PolynomialFeatures(degree=2)),
                                                  ('linear_regression', LinearRegression())
                                                    ]),
                            })

        elif i == "Gaussian Process":
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
            kernel2 = RBF(length_scale=1.0)
            kernel3 = RBF(length_scale=1.0) + DotProduct(sigma_0=1.0)
            models.update({"Gaussian Process-rbf-white": GaussianProcessRegressor(kernel=kernel),
                           "Gaussian Process-rbf": GaussianProcessRegressor(kernel=kernel2),
                            "Gaussian Process-rbf-dot": GaussianProcessRegressor(kernel=kernel3),
                            })
        elif i== "Gradient Boosting":
            models.update({"Gradient Boosting-20-2": GradientBoostingRegressor(n_estimators=20, max_depth=2, learning_rate=0.1, random_state=42),
                            "Gradient Boosting-60-6": GradientBoostingRegressor(n_estimators=60, max_depth=6, learning_rate=0.5, random_state=42),
                            "Gradient Boosting-100-2": GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.9, random_state=42),
                            "Gradient Boosting-20-10": GradientBoostingRegressor(n_estimators=20, max_depth=10, learning_rate=0.5, random_state=42),
                            "Gradient Boosting-100-10": GradientBoostingRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42),
                            })
        elif i == "XGBoosting":
            models.update({"XGBoosting-20-2": XGBRegressor(objective="reg:squarederror", n_estimators=20, max_depth=2, learning_rate=0.1, random_state=42),
                            "XGBoosting-60-6": XGBRegressor(objective="reg:squarederror", n_estimators=60, max_depth=6, learning_rate=0.5, random_state=42),
                            "XGBoosting-100-2": XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=2, learning_rate=0.9, random_state=42),
                            "XGBoosting-20-10": XGBRegressor(objective="reg:squarederror", n_estimators=20, max_depth=10, learning_rate=0.5, random_state=42),
                            "XGBoosting-100-10": XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=10, learning_rate=0.1,  random_state=42),
                            })

    models
    
    results = {}
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    df_x_cv = tools.nom_checkbox(df_x, key="cv")[0]
    st_cv = st.button("Start Cross Validation", key="cv_button")
    if st_cv == True:
        for names, model in models.items():
            # Use cross_val_score with R² as the scoring metric
            # names
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
                # df_x_cv = df_x.copy()
            # df_y_cv = df_y.copy()

            scores = cross_val_score(model, df_x_cv, df_y_cv, cv=kf, scoring=make_scorer(r2_score),n_jobs=-1)
            results[names] = scores
            score_mean = scores.mean()
            score_std = scores.std()
            sn = 10*np.log(score_mean**2/score_std**2)
            # sn
            st.write(f"{names} R²: {score_mean:.3f} ± {score_std:.3f}, SNR: {sn:.3f}")

            # st.write(f"{names} R²: {scores.mean():.3f} ± {scores.std():.3f}")

    # Convert results to a DataFrame
    df_result = pd.DataFrame(results)

    return df_result



def compare_models_clf(df_x, df_y, model_list, cv_splits=5):
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
    score_method = st.selectbox("Choose Score Method", ["accuracy", "f1", "roc_auc", "recall"])

    for i in model_list:
        if i == "Support Vector":
            models.update({"SVR-rbf": SVC(gamma="auto", kernel="rbf"),
                            "SVR-linear": SVC(gamma="auto", kernel="linear"),
                            "SVR-poly": SVC(gamma="auto", kernel="poly"),
                            "SVR-sigmoid": SVC(gamma="auto", kernel="sigmoid"),
                            })
        elif i == "Decision Tree":
            models.update({"Decision Tree-2": DecisionTreeClassifier(max_depth=2, random_state=42),
                            "Decision Tree-3": DecisionTreeClassifier(max_depth=3, random_state=42),
                            "Decision Tree-5": DecisionTreeClassifier(max_depth=5, random_state=42),
                            })
        elif i == "Random Forest":
            models.update({"Random Forest-20-2": RandomForestClassifier(n_estimators=20, max_depth=2, random_state=42),
                            "Random Forest-60-6": RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42),
                            "Random Forest-100-2": RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42),
                            "Random Forest-20-10": RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42),
                            "Random Forest-100-10": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                            })
        elif i == "K-Means":
            models.update({"K-Neighbors-3": KNeighborsClassifier(n_neighbors=3),
                            "K-Neighbors-5": KNeighborsClassifier(n_neighbors=5),
                            "K-Neighbors-10": KNeighborsClassifier(n_neighbors=10),
                            })
        # elif i == "Linear Model":
        #     models.update({"Linear Regression": LinearRegression(),
        #                     "Huber": HuberRegressor(),
        #                     "RANSAC": RANSACRegressor(),
        #                     "TheilSen": TheilSenRegressor(),
        #                     })

        elif i == "Naive Bayes":
            models.update({"GaussianNB": GaussianNB(),
                            "BernoulliNB": BernoulliNB(),
                            "MultinomialNB": MultinomialNB(),
                            })
            
        elif i == "Gaussian Process":
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
            kernel2 = RBF(length_scale=1.0)
            kernel3 = RBF(length_scale=1.0) + DotProduct(sigma_0=1.0)
            models.update({"Gaussian Process-rbf-white": GaussianProcessClassifier(kernel=kernel),
                           "Gaussian Process-rbf": GaussianProcessClassifier(kernel=kernel2),
                            "Gaussian Process-rbf-dot": GaussianProcessClassifier(kernel=kernel3),
                            })
        elif i== "Gradient Boosting":
            models.update({"Gradient Boosting-20-2": GradientBoostingClassifier(n_estimators=20, max_depth=2, learning_rate=0.1, random_state=42),
                            "Gradient Boosting-60-6": GradientBoostingClassifier(n_estimators=60, max_depth=6, learning_rate=0.5, random_state=42),
                            "Gradient Boosting-100-2": GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.9, random_state=42),
                            "Gradient Boosting-20-10": GradientBoostingClassifier(n_estimators=20, max_depth=10, learning_rate=0.5, random_state=42),
                            "Gradient Boosting-100-10": GradientBoostingClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42),
                            })
            
        elif i == "XGBoosting":
            models.update({"XGBoosting-20-2": XGBClassifier(objective="binary:logistic", n_estimators=20, max_depth=2, learning_rate=0.1, random_state=42),
                            "XGBoosting-60-6": XGBClassifier(objective="binary:logistic", n_estimators=60, max_depth=6, learning_rate=0.5, random_state=42),
                            "XGBoosting-100-2": XGBClassifier(objective="binary:logistic", n_estimators=100, max_depth=2, learning_rate=0.9, random_state=42),
                            "XGBoosting-20-10": XGBClassifier(objective="binary:logistic", n_estimators=20, max_depth=10, learning_rate=0.5, random_state=42),
                            "XGBoosting-100-10": XGBClassifier(objective="binary:logistic", n_estimators=100, max_depth=10, learning_rate=0.1,  random_state=42),
                            })
    models
    
    results = {}
    # kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    df_x_cv = tools.nom_checkbox(df_x, key="cv")[0]
    st_cv = st.button("Start Cross Validation", key="cv_button")
    if st_cv == True:
        for names, model in models.items():
            # Use cross_val_score with R² as the scoring metric
            # names
            # if "Gaussian Process" in names:
            #     # Use Gaussian Process with MinMaxScaler
            #     y_scaler = MinMaxScaler()
            #     y = y_scaler.fit_transform(df_y.values.reshape(-1, 1))
            #     df_y_cv = pd.DataFrame(y, columns=[df_y.name]).copy()
            #     # df_y = pd.DataFrame(y, columns=[df_y.name])
            #     # st.subheader("Normalize Y")
            #     # st.dataframe(y)
            # else:
            df_y_cv = df_y.copy()
            # df_x_cv = df_x.copy()
            # df_y_cv = df_y.copy()

            scores = cross_val_score(model, df_x_cv, df_y_cv, cv=skf, scoring=score_method)
            results[names] = scores
            score_mean = scores.mean()
            score_std = scores.std()
            sn = 10*np.log(score_mean**2/score_std**2)
            # sn
            st.write(f"{names} {score_method}: {score_mean:.3f} ± {score_std:.3f}, SNR: {sn:.3f}")

    # Convert results to a DataFrame
    df_result = pd.DataFrame(results)

    return df_result


def optimize_model(df_x, df_y, reg_type, cv_time=5):
    """
    Optimize model parameters using GridSearchCV.

    Parameters:
        df_x (DataFrame): Features (X).
        df_y (Series): Target variable (y).
        reg_type (str): Type of regression model.

    Returns:
        DataFrame: Optimized parameters and scores.
    """

    # df_x_opt = tools.nom_checkbox(df_x, key="opt")[0]
    # st_opt = st.button("Start Optimize", key="opt_button")
    # if st_opt == True:
    st.subheader("Optimize Parameters")

    df_y_opt = df_y.copy()

    if reg_type == "Support Vector":
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'degree': [2, 3, 4],  # Only relevant for 'poly' kernel
            'epsilon': [0.1, 0.2, 0.5]  # Only relevant for regression tasks
        }
        reg = SVR()

    elif reg_type == "Decision Tree":
        param_grid = {
            'max_depth': [2, 4, 6, 8, 10],
            'min_samples_split': [2, 3, 5, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5]
        }
        reg = DecisionTreeRegressor(random_state=42)

    elif reg_type == "Random Forest":
        param_grid = {
            'n_estimators': [20, 50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 3, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8]
        }
        reg = RandomForestRegressor(random_state=42)

    elif reg_type == "K-Means":
        param_grid = {
            'n_neighbors': [3, 5, 7, 10, 15, 20, 25, 30]
        }
        reg = KNeighborsRegressor()

    elif reg_type == "Gaussian Process":
        param_grid = {
            'kernel': [
            RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1),
            RBF(length_scale=1.0) + WhiteKernel(noise_level=1),
            # RBF(length_scale=10.0) + WhiteKernel(noise_level=10),
            RBF(length_scale=0.1),
            RBF(length_scale=1.0),
            # RBF(length_scale=10.0),
            RBF(length_scale=1.0) + DotProduct(sigma_0=0.1),
            RBF(length_scale=1.0) + DotProduct(sigma_0=1.0),
            # RBF(length_scale=1.0) + DotProduct(sigma_0=10.0)
            ],
            'alpha': [1e-10, 1e-5, 1e-2, 0.1, 1.0],  # Regularization parameter
            'n_restarts_optimizer': [0, 5, 10, 20],  # Number of optimizer restarts
            # 'normalize_y': [True, False]  # Whether to normalize the target variable
        }
        reg = GaussianProcessRegressor(random_state=42, n_restarts_optimizer=10)

        y_scaler = MinMaxScaler()
        y = y_scaler.fit_transform(df_y.values.reshape(-1, 1))
        df_y_opt = pd.DataFrame(y, columns=[df_y.name]).copy()

        


    elif reg_type == "Linear Model":
        param_grid = {
            'fit_intercept': [True, False],
            # 'normalize': [True, False]
        }
        reg = LinearRegression()

    elif reg_type == "Gradient Boosting":
        param_grid = {
            'n_estimators': [20, 50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        reg = GradientBoostingRegressor(random_state=42)
    
    elif reg_type == "XGBoosting":
        param_grid = {
            'n_estimators': [20, 50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
        reg = XGBRegressor(random_state=42)


    ## Switch y to MinMaxScaler for Gaussian Process Only.
    # if reg_type == "Gaussian Process":
    #     # Use Gaussian Process with MinMaxScaler
    #     # y_scaler = MinMaxScaler()
    #     # y = y_scaler.fit_transform(df_y.values.reshape(-1, 1))
    #     df_y_opt = pd.DataFrame(y, columns=[df_y.name]).copy()
    # else:
    #     df_y_opt = df_y.copy()
            
    # grid_search = GridSearchCV(reg, param_grid, cv=5)
    grid_search = GridSearchCV(
    estimator=reg,
    param_grid=param_grid,
    cv=cv_time,  
    # scoring=make_scorer(r2_score),  # 使用 R² 作為評估指標
    scoring="r2",  # 使用 R² 作為評估指標
    verbose=1,
    n_jobs=-1  # 使用所有可用的 CPU 核心
    )
    grid_search.fit(df_x, df_y_opt)
    cv_results = grid_search.cv_results_
    df_cv_opt_rslt = pd.DataFrame(cv_results)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    imps = permutation_importance(best_model, df_x, df_y_opt)
    imps_mean = imps.importances_mean
    # imps_std = imps.importances_std
    # imps_mean
    # imps_std
    # df_x.columns
    # imps_mean
    df_imps = pd.DataFrame(imps_mean,
                           columns=["value"]
                           )
                #   index=list(df_x.columns))
                #   columns=("Imp_Mean", "Imp_Std"))
    # st.subheader("Feature Importance")
    # # imps_mean
    # st.dataframe(df_imps)

    return df_cv_opt_rslt, best_params, best_model, df_imps


def optimize_model_clf(df_x, df_y, clf_type, cv_time=5, sub_nb=None):
    """
    Optimize model parameters using GridSearchCV for classification.

    Parameters:
        df_x (DataFrame): Features (X).
        df_y (Series): Target variable (y).
        clf_type (str): Type of classification model.

    Returns:
        DataFrame: Optimized parameters and scores.
    """
    st.subheader("Optimize Parameters")
    df_y_opt = df_y.copy()

    if clf_type == "Support Vector":
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'degree': [2, 3, 4]  # Only relevant for 'poly' kernel
        }
        clf = SVC(probability=True)

    elif clf_type == "Decision Tree":
        param_grid = {
            'max_depth': [2, 4, 6, 8, 10],
            'min_samples_split': [2, 3, 5, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5]
        }
        clf = DecisionTreeClassifier(random_state=42)

    elif clf_type == "Random Forest":
        param_grid = {
            'n_estimators': [20, 50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 3, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8]
        }
        clf = RandomForestClassifier(random_state=42)

    elif clf_type == "K-Means":
        param_grid = {
            'n_neighbors': [3, 5, 7, 10, 15, 20, 25, 30]
        }
        clf = KNeighborsClassifier()

    elif clf_type == "Gaussian Process":
        param_grid = {
            'kernel': [
                RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1),
                RBF(length_scale=1.0) + WhiteKernel(noise_level=1),
                RBF(length_scale=0.1),
                RBF(length_scale=1.0),
                RBF(length_scale=1.0) + DotProduct(sigma_0=0.1),
                RBF(length_scale=1.0) + DotProduct(sigma_0=1.0)
            ],
            'max_iter_predict': [100, 200, 300],
            'n_restarts_optimizer': [0, 5, 10, 20]
        }
        clf = GaussianProcessClassifier(random_state=42)

    elif clf_type == "Logistic Regression":
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear', 'saga']
        }
        clf = LogisticRegression(max_iter=1000, random_state=42)

    elif clf_type == "Gradient Boosting":
        param_grid = {
            'n_estimators': [20, 50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        clf = GradientBoostingClassifier(random_state=42)

    elif clf_type == "XGBoosting":
        param_grid = {
            'n_estimators': [20, 50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        clf = XGBClassifier(random_state=42)

    elif clf_type == "Naive Bayes":
        # st.write(sub_nb)
        # sub_nb
        # sub_nb = st.selectbox("Choose Naive Bayes Type", ["GaussianNB", "BernoulliNB", "MultinomialNB"])
        if sub_nb == "Gaussian":
            param_grid = {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
            clf = GaussianNB()
            # st.write(clf)
        elif sub_nb == "Bernoulli":
            param_grid = {
            'alpha': [0.1, 0.5, 1.0, 2.0],
            'binarize': [0.0, 0.5, 1.0]
            }
            clf = BernoulliNB()
        elif sub_nb == "Multinomial":
            param_grid = {
            'alpha': [0.1, 0.5, 1.0, 2.0],
            'fit_prior': [True, False]
        }
            clf = MultinomialNB()
        # param_grid = {
        #     'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        # }
        # clf = GaussianNB()

    # Initialize GridSearchCV 
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=cv_time,
        scoring="accuracy",  # Use accuracy as the scoring metric
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(df_x, df_y)

    # Extract results
    cv_results = grid_search.cv_results_
    df_cv_opt_rslt = pd.DataFrame(cv_results)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    imps = permutation_importance(best_model, df_x, df_y_opt)
    imps_mean = imps.importances_mean
    # imps_std = imps.importances_std
    # imps_mean
    # imps_std
    # df_x.columns
    # imps_mean
    df_imps = pd.DataFrame(imps_mean,
                           columns=["value"]
                           )

    return df_cv_opt_rslt, best_params, best_model, df_imps


def backend(df_x, df_y, reg_type, factor):

    st.markdown("---")
    # x, nom_choice, df_nom = tools.nom_checkbox(df_x, key="nom_in_reg")
  

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
        
        max_tree_layer = st.number_input("Setup Max Tree Layer", min_value=1, max_value=30, value=2)
        min_leaf = st.number_input("Setup Min Sample Leaf", min_value=1, value=1)
        min_split = st.number_input("Setup Min Sample Split", min_value=2, value=2)
        tree_num = st.number_input("Setup Max Tree Number", min_value=10, value=10)
        reg = RandomForestRegressor(n_estimators=tree_num, max_depth=max_tree_layer, 
                                    min_samples_leaf=min_leaf, min_samples_split=min_split,
                                    random_state=42)

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
        lin_type = st.selectbox("Choose Model", ["OLS", "Polynomial","Huber", "RANSAC", "TheilSen"])
        if lin_type == "OLS":
            reg = LinearRegression()

        elif lin_type == "Polynomial":
            poly_degree = st.number_input("Setup Polynomial Degree", min_value=1, max_value=5, value=2)
            reg = Pipeline([('poly_features', PolynomialFeatures(degree=poly_degree)),
                            ('linear_regression', LinearRegression())
                            ])
  
        elif lin_type == "Huber":
            reg = HuberRegressor()

        elif lin_type == "RANSAC":
            reg = RANSACRegressor()

        elif lin_type == "TheilSen":
            reg = TheilSenRegressor()

    elif reg_type == "Gradient Boosting":
        tree_num = st.number_input("Setup Max Tree Number", min_value=10, value=10)
        max_tree_layer = st.number_input("Setup Max Tree Layer", min_value=1, max_value=30, value=2)
        lrn_rate = st.number_input("Setup Learning Rate", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
        reg = GradientBoostingRegressor(n_estimators=tree_num, max_depth=max_tree_layer, learning_rate=lrn_rate, random_state=42)

    elif reg_type == "XGBoosting":
        tree_num = st.number_input("Setup Max Tree Number", min_value=10, value=10)
        max_tree_layer = st.number_input("Setup Max Tree Layer", min_value=1, max_value=30, value=2)
        lrn_rate = st.number_input("Setup Learning Rate", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
        reg = XGBRegressor(objective="reg:squarederror", n_estimators=tree_num, max_depth=max_tree_layer, learning_rate=lrn_rate,  random_state=42)

    # clf = make_pipeline(StandardScaler(), DecisionTreeClassifier())

    reg.fit(df_x, df_y)
    if reg_type == "Gaussian Process":
        y_pred, gpr_std = reg.predict(df_x, return_std=True)
        st.write("GPR Std:")  
        st.dataframe(gpr_std)

    else:
        y_pred = reg.predict(df_x)
        
        # y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        # y_pred = pd.DataFrame(y_pred, columns=[df_y.name])
        # st.dataframe(y_pred)
    # y_pred = reg.predict(x)

    if reg_type == "Linear Model":

        if lin_type == "Polynomial":
            # poly_degree = st.number_input("Setup Polynomial Degree", min_value=1, max_value=5, value=2)
            # reg = Pipeline([('poly_features', PolynomialFeatures(degree=poly_degree)),
            #                 ('linear_regression', LinearRegression())
            #                 ])
            # poly = reg.named_steps['poly_features']
            # lin = reg.named_steps['linear_regression']
            # coef = lin.coef_
            # st.markdown("Polynomial Model Coefficence")
            # st.dataframe(coef)
            coef = None
        elif lin_type != "RANSAC":
            coef = reg.coef_
            st.markdown("Linear Model Coefficence")
            st.dataframe(coef)
        else:
            coef = reg.estimator_.coef_
            st.dataframe(coef)

    

    if reg_type == "Decision Tree":
        feature_importances = reg.feature_importances_

    imps = permutation_importance(reg, df_x, df_y)
    imps_mean = imps.importances_mean
    imps_std = imps.importances_std
    # imps_mean
    # imps_std
    # df_x.columns
    # imps_mean
    df_imps = pd.DataFrame(imps_mean, 
                           columns=["value"],
                           index=factor,
                  )
    df_imps.index.name = "factor"  # 設定索引名稱為 "factor"
                #   columns=("Imp_Mean", "Imp_Std"))
    # st.subheader("Feature Importance")
    # imps_mean
    # st.dataframe(df_imps)

    if reg_type == "Decision Tree":
            
        st.subheader("Tree Native Feature Importance")
        df_tree_imps = pd.Series(feature_importances,
                  index=factor)
        st.dataframe(df_tree_imps)
    
    return reg, y_pred, gpr_std, df_imps


def backend_clf(df_x, df_y, clf_type, factor):

    # gpr_std = None
    # dict_yscarler = None


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
    # x, nom_choice, df_nom = tools.nom_checkbox(df_x)

    if clf_type == "Support Vector":
        svc_ker = st.selectbox("Choose Kernal", ["linear", "rbf", "poly", "sigmoid"])
        clf = SVC(gamma='auto', kernel=svc_ker, probability=True)
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
    elif clf_type == "Naive Bayes":
        nb_method = st.selectbox("Choose Method", ["Gaussian", "Bernoulli", "Multinomial"])
        if nb_method == "Gaussian":
            clf = GaussianNB()
        elif nb_method == "Bernoulli":
            clf = BernoulliNB()
        elif nb_method == "Multinomial":
            clf = MultinomialNB()
    elif clf_type == "Gaussian Process":
        # kernel = 1.0 * RBF(1.0)
        gpr_ker = st.selectbox("Choose Kernal", ["rbf+white_noise", "rbf", "rbf+dotproduct"])
        if gpr_ker == "rbf+white_noise":
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
        elif gpr_ker == "rbf":
            kernel = RBF(length_scale=1.0)
        elif gpr_ker == "rbf+dotproduct":
            kernel = RBF(length_scale=1.0) + DotProduct(sigma_0=1.0)
        clf = GaussianProcessClassifier(kernel=kernel)

    elif clf_type == "Gradient Boosting":
        tree_num = st.number_input("Setup Max Tree Number", min_value=10, value=10)
        max_tree_layer = st.number_input("Setup Max Tree Layer", min_value=1, max_value=30, value=2)
        lrn_rate = st.number_input("Setup Learning Rate", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
        clf = GradientBoostingClassifier(n_estimators=tree_num, max_depth=max_tree_layer, learning_rate=lrn_rate, random_state=42)

    elif clf_type == "XGBoosting":
        tree_num = st.number_input("Setup Max Tree Number", min_value=10, value=10)
        max_tree_layer = st.number_input("Setup Max Tree Layer", min_value=1, max_value=30, value=2)
        lrn_rate = st.number_input("Setup Learning Rate", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
        clf = XGBClassifier(objective="binary:logistic", n_estimators=tree_num, max_depth=max_tree_layer, learning_rate=lrn_rate,  random_state=42)

    # clf = make_pipeline(StandardScaler(), DecisionTreeClassifier())
    clf.fit(df_x, df_y)
    # if clf_type == "Gaussian Process":
    #     y_pred, gpr_std = clf.predict(x, return_std=True)
    #     st.write("GPR Std:")
    #     st.dataframe(gpr_std)   
    # else:
    y_pred = clf.predict_proba(df_x)[:,1] # probability=True
    y_pred
    # y_pred = pd.DataFrame(y_pred)       
    # y_pred = clf.predict(x)

    
    # st.dataframe(proba)

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

    imps = permutation_importance(clf, df_x, df_y)
    imps_mean = imps.importances_mean
    # imps_std = imps.importances_std
    # imps_mean
    # imps_std
    # df_x.columns
    # imps_mean
    df_imps = pd.DataFrame(imps_mean, 
                           columns=["value"],
                           index=factor,
                  )
    df_imps.index.name = "factor"  # 設定索引名稱為 "factor"
    # df_imps = pd.DataFrame(imps_mean,
    #               index=list(df_x.columns))
                #   columns=("Imp_Mean", "Imp_Std"))
    
    # imps_mean
    # st.dataframe(df_imps)
    # print(imps.importances_mean)
    if clf_type == "Decision Tree":
            
        st.subheader("Tree Native Feature Importance")
        df_tree_imps = pd.Series(feature_importances,
                  index=list(df_x.columns))
        st.dataframe(df_tree_imps)
    
    # importances = clf_mod.feature_importances_ 
    # importances
    # clf.coef_()

    
    return clf, y_pred, df_imps

def main():

    fig_size = [1280, 960]

    st.title('ML Regression Tool')

    st.markdown("               ")
    st.markdown("               ")


    uploaded_raw = st.sidebar.file_uploader('#### 選擇您要上傳的CSV檔', type=["csv", "xlsx"])
    # uploaded_csv = st.file_uploader('#### 選擇您要上傳的CSV檔')

    ana_type = st.selectbox("Choose Analysis Mehthd", ["Regression Method", "Classification", "Clustering"])

    if ana_type == "Regression Method":
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTnqEkuIqYHm1eDDF-wHHyQ-Jm_cvmJuyBT4otEFt0ZE0A6FEQyg1tqpWTU0PXIFA_jYRX8O-G6SzU8/pub?gid=0&single=true&output=csv"

    elif ana_type == "Classification":
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSAL6t_HNdjVhKPyDzt-fVMHqT7ZZnWPrKLIY-QveQxF9vMbR-HbRwcDBM1MEyUjnHkC0JWKbL2TjP0/pub?gid=0&single=true&output=csv"

    elif ana_type == "Clustering":
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTnqEkuIqYHm1eDDF-wHHyQ-Jm_cvmJuyBT4otEFt0ZE0A6FEQyg1tqpWTU0PXIFA_jYRX8O-G6SzU8/pub?gid=0&single=true&output=csv"

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
    if ana_type == "Clustering":
        st.header("Clustering Analysis")
        st.markdown("請選擇分群演算法與參數，並可視化分群結果。")
        factor_list = select_list.copy()
        # factor_list.remove(response)
        factor = st.multiselect(
            "### Choose Factor(x)", factor_list)
        if not factor:
            st.error("Please select at least one factor.")

        if uploaded_raw is None:
            st.header('未上傳檔案，以下為 Demo：')
            response = "Y2"
            factor = ["R", "r", "t"]
        
        df_c = df_reg[factor]
        feature_name = factor

    else:

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
        feature_name = factor
        dict_yscarler = {
            "Y_Max": df_y.max(),
            "Y_Min": df_y.min(),
            "Y_Delta": df_y.max() - df_y.min(),
            # "Y4": MinMaxScaler(feature_range=(0, 1)),
        }    
    # df_factor = pd.DataFrame(factor)

    

    if ana_type == "Regression Method":

        # st.header("Under Construction")
        cross_val_req = st.checkbox("Cross Validation", value=False, key="cross_val")
        if cross_val_req == True:
            st.subheader("Cross Validation")
            cross_val_mod = st.multiselect("Choose Cross Validation Compare Model", 
                                           ["Support Vector", "Decision Tree", "Random Forest", 
                                            "K-Means", "Linear Model", "Gaussian Process", "Gradient Boosting",
                                            "XGBoosting",
                                            ]
                                           )
            cross_val_num = st.number_input("Choose Cross Validation Number", value=5, min_value=2, max_value=10)
            df_cross_val = compare_models(df_x, df_y, model_list=cross_val_mod, cv_splits=cross_val_num)
            st.dataframe(df_cross_val)
  


        reg_type = st.selectbox("### Choose Regression Method", 
                                ["Support Vector", "Decision Tree", "Random Forest", "K-Means",
                                  "Gaussian Process", "Linear Model", "Gradient Boosting",
                                    "XGBoosting",
                                  ])
        df_x_reg, df_nom = tools.nom_checkbox(df_x, key="nom_in_reg")
        ##### opt/pending first############

        if "optimize_triggered" not in st.session_state:
            st.session_state.optimize_triggered = False
        
        reg_select = st.select_slider("Choose Model Method", ["Manual Tune Model", "Use Optimize Model"], key="reg_select")
        if reg_select == "Use Optimize Model":

            df_x_opt = df_x_reg.copy()
            cv_time = st.number_input("Choose Optimize Trial Times", value=3, min_value=2, max_value=10)
            st_opt = st.button("Start Optimize", key="opt_button")
            # 如果按鈕被按下，更新 session_state
            if st_opt:
                st.session_state.optimize_triggered = True
            
             # 根據 session_state 的狀態執行操作
            if st.session_state.optimize_triggered:
            # if st_opt == True:
                df_cv_opt_rslt, best_params, best_model, df_imps_raw = optimize_model(df_x_opt, df_y, reg_type, cv_time=cv_time)
                df_imps = df_imps_raw.copy()
                # df_imps_nm["feature"] = factor
                df_imps.index = factor  # 將 factor 設定為索引
                df_imps.index.name = "factor"  # 設定索引名稱為 "factor"
                # df_imps.insert(0, "factor", factor)
                st.write("Best Parameters:")
                st.dataframe(best_params)
                st.markdown("-----------------")

                # st.write("Feature Importance:")
                st.dataframe(df_imps)
                # st.markdown("-----------------")

                st.write("Opt Overall Result:")
                st.dataframe(df_cv_opt_rslt)
                st.markdown("-----------------")
            else:
                st.warning("Please click the button to start optimize model.")
                st.stop() 
         ##### opt/pending first############
        # df_cv_opt_rslt
        # best_params
            best_model
            reg = best_model
            if reg_type == "Gaussian Process":
                y_pred, gpr_std = reg.predict(df_x_reg, return_std=True)
                st.write("GPR Std:")  
                st.dataframe(gpr_std)
            else:
                y_pred = reg.predict(df_x_reg)
            # st.write("Best Parameters:")
            # st.dataframe(best_params)
            # st.markdown("-----------------")

        elif reg_select == "Manual Tune Model":

            reg, y_pred, gpr_std, df_imps = backend(df_x_reg, df_y, reg_type, factor)
        
        if reg_type == "Gaussian Process":
            
            y_min = dict_yscarler["Y_Min"]
            y_delta = dict_yscarler["Y_Delta"]
            df_y = (df_y - y_min)/y_delta

        # else:
        #     reg, y_pred, nom_choice, df_nom, gpr_std, dict_yscarler, df_imps = backend(df_x, df_y, reg_type)

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


        st.subheader("Feature Importance")
        df_imps = df_imps.reset_index()  # 將 index 重置為普通列（如果 index 是 factor）
        # df_imps.columns = ["factor", "value"]
        df_imps = df_imps.sort_values(by="value", ascending=False)  # 根據 value 由大到小排序

        vline = df_imps["value"].quantile(0.25) # 第一四分位數 (25%)
        st.dataframe(df_imps)
        st.markdown("               ")
        st.markdown("#### $R^2$: %s" % round(model_r2_score, 3))
        st.markdown("               ")
        st.markdown("#### Adjusted $R^2$: %s" % round(adj_r_squared, 3))
        st.markdown("               ")
        st.markdown("#### RMSE: %s" % round(model_rmse_score, 3))
        st.markdown("               ")
        st.markdown("#### MAPE: %s %%" % round(model_mape_score*100, 1))

        dict_met = {
            "R2": model_r2_score,
            "Adj_R2": adj_r_squared,
            "RMSE": model_rmse_score,
            "MAPE": model_mape_score,
        }
        # dict_met
        df_perf = pd.DataFrame(dict_met, index=[0])





        st.subheader("Model Performance Figure")


        df_result=df_reg.copy()
        if reg_type == "Gaussian Process":
            df_result["yhat"] = y_pred * y_delta + y_min
        else:
            df_result["yhat"] = y_pred
        df_result["resid"] = df_reg[response] - df_result["yhat"]
        fig_mod = tools.model_check_figure(df_result=df_result, df_pareto=df_imps, t_val=vline)
        st.plotly_chart(fig_mod, use_container_width=True)
        
        # feature_name = factor
        reg_pkl = {
            "model": reg,
            "features": feature_name,
            "df_nom": df_nom,
            "df_result": df_result,
            "fig_perf": fig_mod,
            "df_perf": df_perf,
            "df_imps": df_imps,
        
        }

        if reg_type == "Gaussian Process":
            reg_pkl["df_yscale"] = dict_yscarler
            reg_pkl["df_gpr_std"] = gpr_std
            


        tools.reg_save(df_result, fig_mod, reg_pkl)


        predict_performance = st.checkbox("Predict New Data & Check Performance", key="reg")

        if predict_performance == True:
            st.subheader("**Predict Portion**")
        
            uploaded_df = st.file_uploader('#### 選擇您要上傳的資料檔', type=["csv", "xlsx"], key="predict")

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

        cross_val_clf = st.checkbox("Cross Validation", value=False, key="cross_val_clf")
        if cross_val_clf == True:
            st.subheader("Cross Validation")
            cross_val_mod = st.multiselect("Choose Cross Validation Compare Model", 
                                           ["Support Vector", "Decision Tree", "Random Forest", 
                                            "K-Means", "Gaussian Process",
                                            "Gradient Boosting", "XGBoosting",
                                            "Naive Bayes",
                                            ]
                                           )
            # cross_val_mod
            cross_val_num = st.number_input("Choose Cross Validation Number", value=5, min_value=2, max_value=10)
            df_cross_val = compare_models_clf(df_x, df_y, model_list=cross_val_mod, cv_splits=cross_val_num)
            st.dataframe(df_cross_val)

        clf_type = st.selectbox("### Choose Classification Method", ["Support Vector", "Decision Tree", 
                                                                     "Random Forest", "K-Means", "Naive Bayes", 
                                                                     "Gaussian Process","Gradient Boosting",
                                                                     "XGBoosting",
                                                                     ])

        df_x_clf, df_nom = tools.nom_checkbox(df_x, key="nom_in_reg")

        # 初始化按鈕狀態
        if "optimize_triggered" not in st.session_state:
            st.session_state.optimize_triggered = False

        clf_select = st.select_slider("Choose Model Method", ["Manual Tune Model", "Use Optimize Model"], key="reg_select")
        
        if clf_select == "Use Optimize Model":
        # df_x = df_reg[factor]
        # df_y = df_reg[response]
            df_x_opt = df_x_clf.copy()
            cv_time = st.number_input("Choose Optimize Trial Times", value=3, min_value=2, max_value=10)
            sub_nb = None
            if clf_type == "Naive Bayes":
                sub_nb = st.selectbox("Choose Navie Bayes Method", ["Gaussian", "Bernoulli", "Multinomial"])
            
            # st.write(sub_nb)
            # 按鈕觸發
            st_opt = st.button("Start Optimize", key="opt_button")

            # 如果按鈕被按下，更新 session_state
            if st_opt:
                st.session_state.optimize_triggered = True
            
             # 根據 session_state 的狀態執行操作
            if st.session_state.optimize_triggered:

                df_cv_opt_rslt, best_params, best_model, df_imps_raw = optimize_model_clf(df_x_opt, df_y, clf_type, cv_time=cv_time, sub_nb=sub_nb)
                df_imps = df_imps_raw.copy()
                # df_imps_nm["Feature"] = factor
                df_imps.insert(0, "Feature", factor)
                st.write("Best Parameters:")
                st.dataframe(best_params)
                st.markdown("-----------------")

                st.write("Feature Importance:")
                st.dataframe(df_imps)
                st.markdown("-----------------")

                st.write("Opt Overall Result:")
                st.dataframe(df_cv_opt_rslt)
                st.markdown("-----------------")
            else:
                st.warning("Please click the button to start optimize model.")
                st.stop() 
         ##### opt/pending first############
        # df_cv_opt_rslt
        # best_params
            best_model
            clf = best_model
            # if clf_type == "Gaussian Process":
            #     y_pred, gpr_std = reg.predict(df_x_reg, return_std=True)
            #     st.write("GPR Std:")  
            #     st.dataframe(gpr_std)
            # else:
            y_pred = clf.predict_proba(df_x_clf)[:,1] # probability=True
            # st.write("Best Parameters:")
            # st.dataframe(best_params)
            # st.markdown("-----------------")

        elif clf_select == "Manual Tune Model":
        
            clf, y_pred, df_imps = backend_clf(df_x_clf, df_y, clf_type, factor)
        # y_pred

        if clf_type == "Decision Tree":

            g = export_graphviz(clf,
                                feature_names=df_x.columns,
                                class_names=["0","1"],
                                filled=True)

            st.subheader("Decision Tree")
            st.graphviz_chart(g)

            # st.plotly_chart(g, use_container_width=True)

        st.subheader("Feature Importance")
        df_imps = df_imps.reset_index()  # 將 index 重置為普通列（如果 index 是 factor）
        # df_imps.columns = ["factor", "value"]
        df_imps = df_imps.sort_values(by="value", ascending=False)  # 根據 value 由大到小排序

        vline = df_imps["value"].quantile(0.25) # 第一四分位數 (25%)
        st.dataframe(df_imps)

        # df_roc_data, fig_roc = clf_score_sklearn(df_y, y_pred)
        gui_key_main = {"threshold": "classify_roc_main", "threshold_check": "key_in_main"}
        df_roc_data, fig_roc = tools.clf_score(df_y, y_pred, gui_key=gui_key_main)

        df_result=df_x.copy()
        df_result["yhat"] = y_pred


        clf_pkl = {
            "model": clf,
            "features": feature_name,
            "df_nom": df_nom,
            "df_result": df_result,
            "fig_perf": fig_roc,
            "df_perf": df_roc_data,
            "df_imps": df_imps,
        
        }


        tools.reg_save(df_result, fig_roc, clf_pkl)


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

    if ana_type == "Clustering":

        # select_list = list(df_raw.columns)
        # cluster_features = st.multiselect("### Choose Features for Clustering", select_list, default=select_list[:2])
        # if not cluster_features:
        #     st.warning("請至少選擇兩個特徵進行分群。")
        #     return
        # df_x_cluster = df_reg[cluster_features]

        cluster_method = st.selectbox("Choose Clustering Method", ["KMeans", "DBSCAN", "Spectral Clustering", "Agglomerative Clustering"])
        df_c_reg, df_nom = tools.nom_checkbox(df_c, key="nom_in_reg")
        if cluster_method in ["KMeans", "Spectral Clustering", "Agglomerative Clustering"]:
            n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=10, value=3)
        else:
            n_clusters = None
        if cluster_method == "DBSCAN":
            eps = st.number_input("DBSCAN eps", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
            min_samples = st.number_input("DBSCAN min_samples", min_value=1, max_value=20, value=5)
        else:
            eps = 0.5
            min_samples = 5

        if st.button("Run Clustering"):
            labels = run_clustering(df_c_reg, cluster_method, n_clusters=n_clusters if n_clusters else 3, eps=eps, min_samples=min_samples)
            st.write("Clustering Labels:")
            df_lb = pd.DataFrame({"index": df_c.index, "cluster": labels})
            # df_c = df_c_reg.copy()
            df_c["cluster"] = labels
            st.dataframe(df_lb)
            st.dataframe(df_c)
            fig = plot_clusters(df_c, labels)
            st.plotly_chart(fig, use_container_width=True)

            tools.download_file(name_label="Input Clustering File Name",
                            button_label='Download clustering result as CSV',
                            file=df_c,
                            file_type="csv",
                            gui_key="clustering_data"
                            )

#%% Web App 頁面
if __name__ == '__main__':
    main()

# %%
