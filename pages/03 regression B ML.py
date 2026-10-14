import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    RANSACRegressor,
    TheilSenRegressor,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from xgboost import XGBClassifier, XGBRegressor

import tools

DEMO_COLS = ("Y2", ["R", "r", "t"])
KERNELS = ["linear", "rbf", "poly", "sigmoid"]
NAIVE_BAYES = {"Gaussian": GaussianNB, "Bernoulli": BernoulliNB, "Multinomial": MultinomialNB}
REG_MODELS = ["Support Vector", "Decision Tree", "Random Forest", "K-Means", "Gaussian Process",
              "Linear Model", "Gradient Boosting", "XGBoosting"]
CLF_MODELS = ["Support Vector", "Decision Tree", "Random Forest", "K-Means", "Naive Bayes",
              "Gaussian Process", "Gradient Boosting", "XGBoosting"]
# (regressor, classifier) for every model family; "K-Means" is the k-neighbours model.
CLASSES = {
    "Support Vector": (SVR, SVC),
    "Decision Tree": (DecisionTreeRegressor, DecisionTreeClassifier),
    "Random Forest": (RandomForestRegressor, RandomForestClassifier),
    "K-Means": (KNeighborsRegressor, KNeighborsClassifier),
    "Gradient Boosting": (GradientBoostingRegressor, GradientBoostingClassifier),
    "XGBoosting": (XGBRegressor, XGBClassifier),
    "Gaussian Process": (GaussianProcessRegressor, GaussianProcessClassifier),
}
TREE_PRESETS = [(20, 2, 0.1), (60, 6, 0.5), (100, 2, 0.9), (20, 10, 0.5), (100, 10, 0.1)]  # trees, depth, lr


def model_class(name, task):
    return CLASSES[name][task == "clf"]


def gp_kernels():
    return {
        "rbf+white_noise": RBF(1.0) + WhiteKernel(1.0),
        "rbf": RBF(1.0),
        "rbf+dotproduct": RBF(1.0) + DotProduct(1.0),
    }


# ---------------------------------------------------------------- model catalogues
def cv_candidates(task, names):
    """Preset estimators compared by cross validation."""
    out = {}
    for name in names:
        if name in CLASSES:
            cls = model_class(name, task)
        if name == "Support Vector":
            out.update({f"SVR-{k}": cls(gamma="auto", kernel=k) for k in KERNELS})
        elif name == "Decision Tree":
            out.update({f"Decision Tree-{d}": cls(max_depth=d, random_state=42) for d in (2, 3, 5)})
        elif name == "Random Forest":
            out.update({f"Random Forest-{t}-{d}": cls(n_estimators=t, max_depth=d, random_state=42)
                        for t, d, _ in TREE_PRESETS})
        elif name == "K-Means":
            out.update({f"K-Neighbors-{k}": cls(n_neighbors=k) for k in (3, 5, 10)})
        elif name == "Gradient Boosting":
            out.update({f"Gradient Boosting-{t}-{d}": cls(n_estimators=t, max_depth=d, learning_rate=lr, random_state=42)
                        for t, d, lr in TREE_PRESETS})
        elif name == "XGBoosting":
            out.update({f"XGBoosting-{t}-{d}": cls(n_estimators=t, max_depth=d, learning_rate=lr, random_state=42)
                        for t, d, lr in TREE_PRESETS})
        elif name == "Gaussian Process":
            kernels = gp_kernels()
            out.update({f"Gaussian Process-{label}": cls(kernel=kernels[k])
                        for label, k in (("rbf-white", "rbf+white_noise"), ("rbf", "rbf"), ("rbf-dot", "rbf+dotproduct"))})
        elif name == "Linear Model":
            out.update({
                "Linear Regression": LinearRegression(),
                "Huber": HuberRegressor(),
                "RANSAC": RANSACRegressor(),
                "TheilSen": TheilSenRegressor(),
                "Degree-2": Pipeline([("poly_features", PolynomialFeatures(degree=2)),
                                      ("linear_regression", LinearRegression())]),
            })
        elif name == "Naive Bayes":
            out.update({f"{k}NB": c() for k, c in NAIVE_BAYES.items()})
    return out


_TREE_GRID = {"max_depth": [3, 5, 7, 10], "n_estimators": [20, 50, 100, 200, 300]}


def grid_spec(task, name, sub_nb=None):
    """(estimator, parameter grid) used by the optimiser."""
    reg = task == "reg"
    boost = {**_TREE_GRID, "learning_rate": [0.01, 0.05, 0.1, 0.2]}
    if name == "Support Vector":
        grid = {"C": [0.1, 1, 10, 100], "kernel": KERNELS, "gamma": ["scale", "auto", 0.01, 0.1, 1], "degree": [2, 3, 4]}
        if reg:
            return SVR(), {**grid, "epsilon": [0.1, 0.2, 0.5]}
        return SVC(probability=True), grid
    if name == "Decision Tree":
        return model_class(name, task)(random_state=42), {
            "max_depth": [2, 4, 6, 8, 10], "min_samples_split": [2, 3, 5, 10], "min_samples_leaf": [1, 2, 3, 4, 5]}
    if name == "Random Forest":
        return model_class(name, task)(random_state=42), {
            **_TREE_GRID, "min_samples_split": [2, 3, 5, 10], "min_samples_leaf": [1, 2, 4, 8]}
    if name == "K-Means":
        return model_class(name, task)(), {"n_neighbors": [3, 5, 7, 10, 15, 20, 25, 30]}
    if name == "Gaussian Process":
        kernels = [RBF(0.1) + WhiteKernel(0.1), RBF(1.0) + WhiteKernel(1.0), RBF(0.1), RBF(1.0),
                   RBF(1.0) + DotProduct(0.1), RBF(1.0) + DotProduct(1.0)]
        extra = {"alpha": [1e-10, 1e-5, 1e-2, 0.1, 1.0]} if reg else {"max_iter_predict": [100, 200, 300]}
        return model_class(name, task)(random_state=42), {
            "kernel": kernels, **extra, "n_restarts_optimizer": [0, 5, 10, 20]}
    if name == "Linear Model":
        return LinearRegression(), {"fit_intercept": [True, False]}
    if name == "Gradient Boosting":
        return model_class(name, task)(random_state=42), {
            **boost, "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
    if name == "XGBoosting":
        return model_class(name, task)(random_state=42), {
            **boost, "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]}
    # Naive Bayes
    grids = {
        "Gaussian": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
        "Bernoulli": {"alpha": [0.1, 0.5, 1.0, 2.0], "binarize": [0.0, 0.5, 1.0]},
        "Multinomial": {"alpha": [0.1, 0.5, 1.0, 2.0], "fit_prior": [True, False]},
    }
    return NAIVE_BAYES[sub_nb](), grids[sub_nb]


def manual_estimator(task, name):
    """Render the hyper-parameter widgets of `name` and return the configured estimator."""
    clf = task == "clf"
    num = st.number_input

    def boosting():
        return dict(n_estimators=num("Setup Max Tree Number", min_value=10, value=10),
                    max_depth=num("Setup Max Tree Layer", min_value=1, max_value=30, value=2),
                    learning_rate=num("Setup Learning Rate", value=0.1, min_value=0.01, max_value=1.0, step=0.01))

    if name == "Support Vector":
        return model_class(name, task)(gamma="auto", kernel=st.selectbox("Choose Kernal", KERNELS),
                                       **({"probability": True} if clf else {}))
    if name == "Decision Tree":
        return model_class(name, task)(max_depth=num("Setup Max Tree Layer", min_value=1, max_value=10, value=2),
                                       random_state=42)
    if name == "Random Forest":
        depth = num("Setup Max Tree Layer", min_value=1, max_value=30, value=2)
        leaf = num("Setup Min Sample Leaf", min_value=1, value=1)
        split = num("Setup Min Sample Split", min_value=2, value=2)
        trees = num("Setup Max Tree Number", min_value=10, value=10)
        return model_class(name, task)(n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
                                       min_samples_split=split, random_state=42)
    if name == "K-Means":
        return model_class(name, task)(n_neighbors=num("Setup Neighbor Q'ty", min_value=1, max_value=10, value=2))
    if name in ("Gradient Boosting", "XGBoosting"):
        return model_class(name, task)(random_state=42, **boosting())
    if name == "Gaussian Process":
        kernel = gp_kernels()[st.selectbox("Choose Kernal", list(gp_kernels()))]
        if clf:
            return GaussianProcessClassifier(kernel=kernel)
        return GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=10)
    if name == "Naive Bayes":
        return NAIVE_BAYES[st.selectbox("Choose Method", list(NAIVE_BAYES))]()
    # Linear Model
    kind = st.selectbox("Choose Model", ["OLS", "Polynomial", "Huber", "RANSAC", "TheilSen"])
    if kind == "Polynomial":
        degree = num("Setup Polynomial Degree", min_value=1, max_value=5, value=2)
        return Pipeline([("poly_features", PolynomialFeatures(degree=degree)), ("linear_regression", LinearRegression())])
    return {"OLS": LinearRegression, "Huber": HuberRegressor, "RANSAC": RANSACRegressor, "TheilSen": TheilSenRegressor}[kind]()


# ---------------------------------------------------------------- cached heavy work
@st.cache_data(show_spinner="Running cross validation...")
def cross_validate_models(task, names, X, y, splits, scoring):
    cv_cls = KFold if task == "reg" else StratifiedKFold
    cv = cv_cls(n_splits=splits, shuffle=True, random_state=42)
    y_scaled = (y - y.min()) / (y.max() - y.min())  # Gaussian process is fitted on scaled y
    models = cv_candidates(task, names)

    def score(label, est):
        target = y_scaled if task == "reg" and "Gaussian Process" in label else y
        return cross_val_score(est, X, target, cv=cv, scoring=scoring)

    scores = Parallel(n_jobs=-1)(delayed(score)(label, est) for label, est in models.items())
    return pd.DataFrame(dict(zip(models, scores)))


@st.cache_data(show_spinner="Optimizing parameters...")
def run_grid(task, name, sub_nb, cv, X, y):
    est, grid = grid_spec(task, name, sub_nb)
    search = GridSearchCV(est, grid, cv=cv, scoring="r2" if task == "reg" else "accuracy", n_jobs=-1)
    search.fit(X, y)
    imps = permutation_importance(search.best_estimator_, X, y, random_state=42).importances_mean
    return pd.DataFrame(search.cv_results_), search.best_params_, search.best_estimator_, imps


@st.cache_data(show_spinner="Fitting model...")
def fit_model(sig, _est, X, y):
    """`sig` (the estimator repr) is the cache key; `_est` is excluded from hashing."""
    _est.fit(X, y)
    imps = permutation_importance(_est, X, y, random_state=42).importances_mean
    return _est, imps


@st.cache_data(show_spinner="Computing elbow curve...")
def kmeans_elbow(X, max_k=15):
    ks = range(1, min(max_k, len(X) + 1))
    df = pd.DataFrame({"K": list(ks), "SSE": [KMeans(n_clusters=k, random_state=42).fit(X).inertia_ for k in ks]})
    df["SSE_ratio"] = df["SSE"].pct_change()
    df["SSE_div_first"] = df["SSE"] / df["SSE"].iloc[0]
    return df


# ---------------------------------------------------------------- UI building blocks
def cross_validation_ui(task, df_x, df_y):
    names = REG_MODELS if task == "reg" else CLF_MODELS
    st.subheader("Cross Validation")
    chosen = st.multiselect("Choose Cross Validation Compare Model", names)
    splits = st.number_input("Choose Cross Validation Number", value=5, min_value=2, max_value=10)
    scoring = "r2" if task == "reg" else st.selectbox("Choose Score Method", ["accuracy", "f1", "roc_auc", "recall"])
    X, _ = tools.nom_checkbox(df_x, key="cv")

    if not chosen:
        return
    sig = (task, tuple(chosen), int(splits), scoring, tuple(df_x.columns), len(df_x), X.to_numpy().sum())
    if not tools.run_gate("Start Cross Validation", "cv_button", sig):
        return
    df_cv = cross_validate_models(task, tuple(chosen), X, df_y, int(splits), scoring)
    mean, std = df_cv.mean(), df_cv.std(ddof=0)
    summary = pd.DataFrame({f"{scoring} mean": mean, f"{scoring} std": std,
                            "SNR": 10 * np.log10(mean**2 / std**2)})
    st.dataframe(summary)
    st.dataframe(df_cv)


def show_model_details(task, name, model, factor):
    """Coefficients / native importance / tree drawing for the models that have them."""
    if name == "Linear Model" and not isinstance(model, Pipeline):
        base = model.estimator_ if isinstance(model, RANSACRegressor) else model
        st.markdown("Linear Model Coefficence")
        st.dataframe(base.coef_)
    elif name == "Support Vector" and getattr(model, "kernel", None) == "linear" and task == "clf":
        st.subheader("SVC Model Coefficence")
        st.write(model.coef_)
    elif name == "Decision Tree":
        st.subheader("Tree Native Feature Importance")
        st.dataframe(pd.Series(model.feature_importances_, index=factor))
        st.subheader("Decision Tree")
        st.graphviz_chart(export_graphviz(model, feature_names=factor, filled=True,
                                          class_names=["0", "1"] if task == "clf" else None))


def fit_or_optimize(task, name, X, y_fit, factor):
    """Manual-tune or grid-search flow. Returns (model, feature importances)."""
    mode = st.select_slider("Choose Model Method", ["Manual Tune Model", "Use Optimize Model"], key="reg_select")
    if mode == "Manual Tune Model":
        est = manual_estimator(task, name)
        if task == "reg" and name == "Gaussian Process":
            st.subheader("Normalize Y")
            st.dataframe(y_fit)
        try:
            return fit_model(repr(est), est, X, y_fit)
        except Exception as e:
            st.error(f"Model fitting failed: {e}")
            st.stop()

    cv = int(st.number_input("Choose Optimize Trial Times", value=3, min_value=2, max_value=10))
    sub_nb = st.selectbox("Choose Navie Bayes Method", list(NAIVE_BAYES)) if name == "Naive Bayes" else None
    sig = (task, name, cv, sub_nb, tuple(factor), len(X), float(np.sum(X)))
    if not tools.run_gate("Start Optimize", "opt_button", sig):
        st.warning("Please click the button to start optimize model.")
        st.stop()

    st.subheader("Optimize Parameters")
    df_cv, best_params, model, imps = run_grid(task, name, sub_nb, cv, X, y_fit)
    st.write("Best Parameters:")
    st.dataframe(pd.Series({k: str(v) for k, v in best_params.items()}, name="value"))
    st.write("Opt Overall Result:")
    st.dataframe(df_cv)
    st.divider()
    return model, imps


def model_page(task, df_reg, factor, response):
    reg = task == "reg"
    df_x, df_y = df_reg[factor], df_reg[response]

    if st.checkbox("Cross Validation", value=False, key=f"cross_val_{task}"):
        cross_validation_ui(task, df_x, df_y)

    name = st.selectbox("### Choose Regression Method" if reg else "### Choose Classification Method",
                        REG_MODELS if reg else CLF_MODELS)
    X, df_nom = tools.nom_checkbox(df_x, key="nom_in_reg")

    scaled_y = reg and name == "Gaussian Process"  # Gaussian process regression is fitted on min-max scaled y
    y_min, y_delta = df_y.min(), df_y.max() - df_y.min()
    y_fit = (df_y - y_min) / y_delta if scaled_y else df_y

    model, imps = fit_or_optimize(task, name, X, y_fit, factor)
    show_model_details(task, name, model, factor)

    gpr_std = None
    if scaled_y:
        pred, gpr_std = model.predict(X, return_std=True)
        st.write("GPR Std:")
        st.dataframe(gpr_std)
    else:
        pred = model.predict(X) if reg else model.predict_proba(X)[:, 1]
    y_hat = pred * y_delta + y_min if scaled_y else pred

    df_imps = (pd.DataFrame({"factor": factor, "value": imps})
               .sort_values("value", ascending=False).reset_index(drop=True))
    ref_line = df_imps["value"].quantile(0.25)
    st.subheader("Feature Importance")
    st.dataframe(df_imps)

    def apply_model_nom(df_test):
        data = df_test[factor]
        return tools.apply_nom(data, df_nom) if len(df_nom) else data

    pkl = {"model": model, "features": factor, "df_nom": df_nom, "df_imps": df_imps}
    if reg:
        r2, adj, rmse, mape = tools.show_perf(df_y, y_hat, p=len(factor))
        st.subheader("Model Performance Figure")
        df_result = df_reg.copy()
        df_result["yhat"] = y_hat
        df_result["resid"] = df_y - y_hat
        fig = tools.model_check_figure(df_result, df_imps, ref_line)
        st.plotly_chart(fig, width="stretch")
        pkl["df_perf"] = pd.DataFrame({"R2": r2, "Adj_R2": adj, "RMSE": rmse, "MAPE": mape}, index=[0])
        if scaled_y:
            pkl["df_yscale"] = {"Y_Max": df_y.max(), "Y_Min": y_min, "Y_Delta": y_delta}
            pkl["df_gpr_std"] = gpr_std

        def predict(df_test):
            out = model.predict(apply_model_nom(df_test))
            return out * y_delta + y_min if scaled_y else out

        def score(y, y_pred):
            tools.show_perf(y, y_pred, key="reg")
    else:
        df_roc, fig = tools.clf_score(df_y, y_hat, key="main")
        df_result = df_x.copy()
        df_result["yhat"] = y_hat
        pkl["df_perf"] = df_roc

        def predict(df_test):
            return model.predict(apply_model_nom(df_test))

        def score(y, y_pred):
            tools.clf_score(y, y_pred, key="pred", threshold_ui=False)

    pkl.update(df_result=df_result, fig_perf=fig)
    tools.reg_save(df_result, fig, pkl)
    tools.predict_section(predict, score, key=task)


# ---------------------------------------------------------------- clustering
def run_clustering(X, method, n_clusters, eps, min_samples):
    if method == "KMeans":
        return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)
    if method == "DBSCAN":
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    if method == "Spectral Clustering":
        return SpectralClustering(n_clusters=n_clusters, assign_labels="kmeans", random_state=42).fit_predict(X)
    return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)


def plot_clusters(df):
    """2-D / 3-D scatter of the first features coloured by cluster."""
    cols = [c for c in df.columns if c != "cluster"]
    df = df.assign(cluster=df["cluster"].astype(str))
    if len(cols) >= 3:
        return px.scatter_3d(df, x=cols[0], y=cols[1], z=cols[2], color="cluster", symbol="cluster")
    if len(cols) == 2:
        return px.scatter(df, x=cols[0], y=cols[1], color="cluster", symbol="cluster")
    return px.scatter(df.assign(zero=0), x=cols[0], y="zero", color="cluster", symbol="cluster")


def clustering_page(df_reg, factor):
    df_c = df_reg[factor].copy()
    method = st.selectbox("Choose Clustering Method", ["KMeans", "DBSCAN", "Spectral Clustering", "Agglomerative Clustering"])
    X, df_nom = tools.nom_checkbox(df_c, key="nom_in_reg")

    n_clusters, eps, min_samples = 3, 0.5, 5
    if method == "DBSCAN":
        eps = st.number_input("DBSCAN eps", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
        min_samples = st.number_input("DBSCAN min_samples", min_value=1, max_value=20, value=5)
    else:
        n_clusters = int(st.number_input("Number of Clusters", min_value=2, max_value=10, value=3))

    sig = (method, n_clusters, eps, min_samples, tuple(factor), len(df_nom), len(X))
    if not tools.run_gate("Run Clustering", "cluster_btn", sig):
        return
    if method == "KMeans":
        df_elbow = kmeans_elbow(X)
        st.dataframe(df_elbow)
        st.plotly_chart(px.line(df_elbow, x="K", y="SSE", markers=True, title="Elbow Method for Optimal K",
                                labels={"SSE": "Sum of Squared Errors"}, template="plotly_white"), width="stretch")

    df_c["cluster"] = run_clustering(X, method, n_clusters, eps, min_samples)
    st.write("Clustering Labels:")
    st.dataframe(df_c)
    st.plotly_chart(plot_clusters(df_c), width="stretch")
    tools.download_file("Input Clustering File Name", "Download clustering result as CSV", df_c, "csv", "clustering_data")


def main():
    st.title("ML Regression Tool")

    uploaded = st.sidebar.file_uploader(tools.UPLOAD_LABEL, type=tools.TABLE_TYPES)
    ana_type = st.selectbox("Choose Analysis Mehthd", ["Regression Method", "Classification", "Clustering"])

    df_raw = tools.load_table(uploaded, tools.DEMO_CLF if ana_type == "Classification" else tools.DEMO_REG)
    st.dataframe(df_raw)
    df_reg = tools.filter_data(df_raw)
    st.dataframe(df_reg.describe())

    cols = list(df_reg.columns)
    demo_y, demo_x = DEMO_COLS if uploaded is None else (None, None)
    response = None
    if ana_type == "Clustering":
        st.header("Clustering Analysis")
        st.markdown("請選擇分群演算法與參數，並可視化分群結果。")
        factor = tools.pick("### Choose Factor(x)", cols, demo_x, multi=True)
    else:
        response = tools.pick("### Choose Response(y)", cols, demo_y)
        factor = tools.pick("### Choose Factor(x)", [c for c in cols if c != response], demo_x, multi=True)
    if not factor:
        st.error("Please select at least one factor.")
        st.stop()

    if ana_type == "Clustering":
        clustering_page(df_reg, factor)
    else:
        model_page("reg" if ana_type == "Regression Method" else "clf", df_reg, factor, response)


main()
