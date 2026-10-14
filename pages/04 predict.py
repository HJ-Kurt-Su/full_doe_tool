import io
import pickle

import requests
import streamlit as st

import tools


@st.cache_resource(show_spinner=False)
def load_package(data):
    package = pickle.load(io.BytesIO(data))
    if not isinstance(package, dict):  # a bare fitted model
        names = getattr(getattr(package, "model", None), "exog_names", [])
        package = {"model": package, "features": list(names)}
    return package


@st.cache_resource(ttl=3600, show_spinner="Downloading demo model...")
def demo_package():
    return load_package(requests.get(tools.DEMO_MODEL, timeout=30).content)


def predict(package, df, proba):
    """Prediction column for `df` using a model package saved by the regression pages."""
    model = package["model"]
    if getattr(getattr(model, "model", None), "formula", None):  # statsmodels formula result
        X = df
    else:
        X = df[list(package["features"])]
        df_nom = package.get("df_nom")
        if df_nom is not None and len(df_nom) == 2:
            X = tools.apply_nom(X, df_nom)

    if proba:
        return model.predict_proba(X)[:, 1]
    y_hat = model.predict(X)
    if "GaussianProcessRegressor" in type(model).__name__ and "df_yscale" in package:
        scale = package["df_yscale"]
        y_hat = y_hat * scale["Y_Delta"] + scale["Y_Min"]
    return y_hat


def main():
    st.title("Predict")

    uploaded_model = st.sidebar.file_uploader("請上傳您的 Model 檔案", type=["pickle"])
    uploaded_raw = st.sidebar.file_uploader("請上傳您的 CSV 檔案", type=tools.TABLE_TYPES)

    df_raw = tools.load_table(uploaded_raw, tools.DEMO_PREDICT)
    st.dataframe(df_raw)
    df_prd = tools.filter_data(df_raw)

    if uploaded_model is not None:
        st.header("您所上傳的 Model 檔內容：")
        st.write(uploaded_model.name)
        package = load_package(uploaded_model.getvalue())
    else:
        st.header("未上傳 Model，以下為 Demo：")
        package = demo_package()
    st.divider()

    st.markdown("#### Model Parameter")
    st.dataframe(package["features"])

    model = package["model"]
    proba = False
    if hasattr(model, "predict_proba"):
        proba = st.radio("#### Choose Result Type", ("Result Only", "Probability")) == "Probability"
        if proba:
            st.markdown("#### Probability Result")

    df_predict = df_prd.copy()
    df_predict["predict"] = predict(package, df_prd, proba)

    st.header("預測結果：")
    st.dataframe(df_predict)
    st.divider()
    tools.download_file("Input Result File Name", "Download predict result as CSV", df_predict, "csv", "result_data")
    st.divider()

    if not st.checkbox("Check Predict Performance"):
        return
    source = st.radio("**Real Y From:**", ["another upload", "original data"])
    if source == "another upload":
        up = st.file_uploader(tools.UPLOAD_LABEL, type=tools.TABLE_TYPES)
        df_real = tools.load_table(up, tools.DEMO_REAL_Y)
        st.dataframe(df_real)
    else:
        df_real = df_prd
    y = st.selectbox("Please select real value", list(df_real.columns))
    tools.show_perf(df_real[y], df_predict["predict"], key="predict")


main()
