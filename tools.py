"""Shared helpers for the Streamlit pages: data loading, downloads, filters,
normalisation, model metrics and common figures."""

import datetime
import io
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_percentage_error,
    precision_recall_curve,
    r2_score,
    roc_curve,
    root_mean_squared_error,
)

_SHEET = "https://docs.google.com/spreadsheets/d/e/{}/pub?gid=0&single=true&output=csv"
DEMO_DOE = _SHEET.format("2PACX-1vRt_gecFPJZUw3MmI16zI042Z_RaeWj7w-Fs07wLaj-qKymK_K_XwRx-G09IcrqHmHhUQqP8-Qe0cQQ")
DEMO_REG = _SHEET.format("2PACX-1vTnqEkuIqYHm1eDDF-wHHyQ-Jm_cvmJuyBT4otEFt0ZE0A6FEQyg1tqpWTU0PXIFA_jYRX8O-G6SzU8")
DEMO_TAGUCHI = _SHEET.format("2PACX-1vR2pTVvSJJEWf1woRdODBYzBvJsHvgIgcJWAly2EDoNGm610xiMISNdaf8yq1f8h732zqel7v-vNon7")
DEMO_CLF = _SHEET.format("2PACX-1vSAL6t_HNdjVhKPyDzt-fVMHqT7ZZnWPrKLIY-QveQxF9vMbR-HbRwcDBM1MEyUjnHkC0JWKbL2TjP0")
DEMO_REAL_Y = _SHEET.format("2PACX-1vQ5skYRbfVPGE6RFYIM6Gg9QurH8u3h_RLfjt-CG0z5YgyxWEUTOdvoKmVkfWCLc2ECAuSEKaHVYPOA")
DEMO_PREDICT = _SHEET.format("2PACX-1vQjqLJWGJ46N6GkXkkMCcmthgALF9J28Bm1SGnwYUdmTpTn4Kq8tCzQ-G5FMddTZFF5YANEtPYese-g")
DEMO_MODEL = "https://raw.githubusercontent.com/HJ-Kurt-Su/full_doe_tool/main/2023-07-01_model.pickle"

UPLOAD_LABEL = "#### 選擇您要上傳的CSV檔"
TABLE_TYPES = ["csv", "xlsx"]
COLOR_SEQ = px.colors.qualitative.Pastel
POINT_COLOR = "rgba(19, 166, 255, 0.6)"


## Data loading
@st.cache_data(ttl=3600, show_spinner=False)
def read_url(url):
    return pd.read_csv(url, encoding="utf-8")


@st.cache_data(show_spinner=False)
def read_upload(name, data):
    buf = io.BytesIO(data)
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    return pd.read_csv(buf, encoding="utf-8")


def load_table(uploaded, demo_url):
    """Return the uploaded table, or the demo table when nothing is uploaded."""
    if uploaded is None:
        st.header("未上傳檔案，以下為 Demo：")
        return read_url(demo_url)
    st.header("您所上傳的CSV檔內容：")
    return read_upload(uploaded.name, uploaded.getvalue())


def filter_data(df, key="filter"):
    """Optional 'filter rows by column value' widget; returns the (filtered) frame."""
    if not st.checkbox("Filter Data", key=f"{key}_req"):
        return df
    st.divider()
    st.markdown("#### Filter Parameter")
    col = st.selectbox("### Choose filter column", list(df.columns), key=f"{key}_col")
    items = st.multiselect("### Choose item", df[col].unique(), key=f"{key}_items")
    st.divider()
    out = df[df[col].isin(items)].copy()
    st.markdown("#### Filter DataFrame")
    st.dataframe(out)
    st.divider()
    if out.empty:
        st.warning("Please choose at least one item to keep.")
        st.stop()
    return out


def pick(label, options, demo, multi=False):
    """selectbox/multiselect that preselects the demo columns when `demo` is given."""
    if multi:
        return st.multiselect(label, options, default=[c for c in demo or [] if c in options])
    return st.selectbox(label, options, index=options.index(demo) if demo in options else 0)


def fig_size_inputs():
    c1, c2 = st.columns(2)
    with c1:
        width = st.number_input("Figure Width", min_value=640, value=1280, max_value=5120, step=320)
    with c2:
        height = st.number_input("Figure Height", min_value=480, value=960, max_value=3840, step=240)
    return width, height


## Downloads
def _csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def _html_bytes(fig):
    return fig.to_html(include_plotlyjs="cdn").encode()


_SERIALIZERS = {
    "csv": (_csv_bytes, "text/csv"),
    "html": (_html_bytes, "text/html"),
    "pickle": (pickle.dumps, "application/octet-stream"),
}


def download_file(name_label, button_label, file, file_type, gui_key):
    """File-name box + download button. The payload is only built when clicked."""
    serialize, mime = _SERIALIZERS[file_type]
    stem = st.text_input(label=name_label, value=f"{datetime.date.today()}_result", key=gui_key)
    st.download_button(
        label=button_label,
        data=lambda: serialize(file),
        file_name=f"{stem}.{file_type}",
        mime=mime,
        key=f"{gui_key}dl",
        on_click="ignore",
    )


def reg_save(df_result, fig, model):
    for label, btn, obj, ftype, key in [
        ("Input Model File Name", "Download model as PICKLE", model, "pickle", "model"),
        ("Input Figure File Name", "Download figure as HTML", fig, "html", "figure"),
        ("Input Result File Name", "Download statistics result as CSV", df_result, "csv", "result_data"),
    ]:
        st.divider()
        download_file(label, btn, obj, ftype, key)
    st.divider()


def run_gate(label, key, sig):
    """Button whose result survives reruns until `sig` (the inputs) changes."""
    if st.button(label, key=key):
        st.session_state[f"{key}_sig"] = sig
    return st.session_state.get(f"{key}_sig") == sig


## Normalisation (row 0 = mean, row 1 = std, the layout stored in model pickles)
def apply_nom(df, df_nom):
    cols = list(df_nom.columns)
    mean = df_nom.iloc[0].astype(float)
    std = df_nom.iloc[1].astype(float)
    return (df[cols].astype(float) - mean) / std


def nom_checkbox(df_x, key=None):
    """Optional standardisation. Returns (X, df_nom); df_nom has no rows when unused."""
    if not st.checkbox("Normalize Factor", value=False, key=key):
        return df_x, pd.DataFrame(columns=df_x.columns)

    st.subheader("Normalize Factor Mean & Standard Deviation")
    mean = df_x.mean()
    std = df_x.std(ddof=0).replace(0, 1.0)
    df_nom = pd.DataFrame([mean, std]).reset_index(drop=True)
    x = apply_nom(df_x, df_nom)
    st.subheader("Normalize X")
    st.dataframe(x)
    st.dataframe(df_nom)
    download_file("Normalize Factor Mean & Std", "Download Normalize Factor", df_nom, "csv", f"{key}_nom_df")
    return x, df_nom


## Metrics
def perf_metrics(y, prediction, p):
    """R2, adjusted R2, RMSE, MAPE for `p` factors."""
    r2 = r2_score(y, prediction)
    n = len(y)
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else float("nan")
    return r2, adj, root_mean_squared_error(y, prediction), mean_absolute_percentage_error(y, prediction)


def show_perf(y, prediction, p=None, key="perf", default_p=2):
    """Render regression metrics; asks for the factor number when `p` is None."""
    if len(y) != len(prediction):
        st.error(f"Real value ({len(y)} rows) and prediction ({len(prediction)} rows) differ in length.")
        st.stop()
    if p is None:
        p = st.number_input("Choose Factor Number", value=default_p, min_value=default_p, key=f"{key}_p")
    r2, adj, rmse, mape = perf_metrics(y, prediction, p)
    for text in (
        f"$R^2$: {r2:.3f}",
        f"Adjusted $R^2$: {adj:.3f}",
        f"RMSE: {rmse:.3f}",
        f"MAPE: {mape * 100:.1f} %",
    ):
        st.markdown(f"#### {text}")
    return r2, adj, rmse, mape


def predict_section(predict_fn, score_fn, key):
    """'Predict new data' + 'check performance' block shared by the regression pages.

    predict_fn(df_test) -> predictions; score_fn(y_true, y_hat) renders the metrics.
    """
    if not st.checkbox("Predict New Data & Check Performance", key=f"{key}_pp"):
        return
    st.subheader("**Predict Portion**")
    up = st.file_uploader("#### 選擇您要上傳的資料檔", type=TABLE_TYPES, key=f"{key}_predict")
    df_test = load_table(up, DEMO_REG)
    y_hat = predict_fn(df_test)
    df_test["predict"] = y_hat
    st.markdown("Predict Result:")
    st.dataframe(df_test)
    download_file("Input Predict File Name", "Download predict result as CSV", df_test, "csv", f"{key}_predict_data")
    st.divider()

    if not st.checkbox("Check Model Performance", key=f"{key}_chk"):
        return
    st.subheader("**Model Perfomance Portion**")
    up_y = st.file_uploader(UPLOAD_LABEL, type=TABLE_TYPES, key=f"{key}_up_y")
    df_y = load_table(up_y, DEMO_REAL_Y)
    st.dataframe(df_y)
    y = st.selectbox("Please select real value", list(df_y.columns), key=f"{key}_y")
    score_fn(df_y[y], y_hat)


def clf_score(y, y_pred, key, threshold_ui=True):
    """ROC / confusion-matrix report for a binary classifier.

    y_pred holds probabilities (threshold_ui=True) or already-binary labels.
    Returns (ROC data, ROC figure).
    """
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    st.markdown(f"### AUC is: {auc(fpr, tpr):.4f}")

    df_roc = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thresholds})
    st.subheader("ROC Data")
    st.dataframe(df_roc)
    fig_roc = px.line(df_roc, x="FPR", y="TPR", range_x=[0, 1], range_y=[0, 1.1], width=640, height=640)
    st.subheader("ROC Figure")
    st.plotly_chart(fig_roc, width="stretch")

    if threshold_ui:
        st.subheader("Accuracy Judge")
        if st.checkbox("Key-in threshold", key=f"{key}_ti"):
            cut = st.number_input("Key-in threshold value", min_value=0.0001, max_value=0.9999, value=0.5, key=f"{key}_tv")
        else:
            cut = st.selectbox("Choose Threshold", thresholds, key=f"{key}_ts")
        st.write(f"Threshold value is: {cut}")
        y_code = (y_pred >= cut).astype(int)
    else:
        y_code = y_pred

    cm = confusion_matrix(y, y_code, labels=[0, 1])
    df_cm = pd.DataFrame(cm, index=["Real Pass", "Real Fail"], columns=["Predict Pass", "Predict Fail"])
    st.markdown(f"### Accuracy is: {accuracy_score(y, y_code):.4f}")
    st.markdown(f"### Risk is: {cm[1, 0] / y_code.size:.4f}")
    st.markdown("### Confusion Matrix:")
    st.dataframe(df_cm)
    st.divider()

    if st.checkbox("Use Precision & Recall Method", key=f"{key}_pr"):
        precision, recall, _ = precision_recall_curve(y, y_code)
        st.markdown(f"### Average Precision is: {average_precision_score(y, y_code):.4f}")
        st.markdown(f"### Precision-Recall AUC is: {auc(recall, precision):.4f}")
        st.markdown(f"### F1 Socre is: {f1_score(y, y_code):.4f}")
        df_pr = pd.DataFrame({"Precision": precision, "Recall": recall})
        st.dataframe(df_pr)
        fig_pr = px.line(df_pr, x="Recall", y="Precision", range_x=[0, 1], range_y=[0, 1.1], width=640, height=640)
        st.subheader("Precision Recall Figure")
        st.plotly_chart(fig_pr, width="stretch")

    return df_roc, fig_roc


## Figures
def model_check_figure(df_result, df_bar, ref, bar_title="feature-importance (red line as ref. criteria)"):
    """2x2 residual diagnostics. df_bar needs 'factor' & 'value'; ref is the red reference line."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "yhat-residual-plot (random better)",
            "residual-histogram-plot (normal distribution better)",
            "redidual-sequence-plot (random better)",
            bar_title,
        ),
    )
    marker = dict(color=POINT_COLOR)
    df_bar = df_bar.sort_values("value")
    fig.add_trace(go.Scatter(x=df_result["yhat"], y=df_result["resid"], mode="markers", marker=marker), row=1, col=1)
    fig.add_trace(go.Histogram(x=df_result["resid"], marker=marker), row=1, col=2)
    fig.add_trace(go.Scatter(y=df_result["resid"], mode="lines+markers", marker=marker), row=2, col=1)
    fig.add_trace(go.Bar(x=df_bar["value"], y=df_bar["factor"], orientation="h", width=0.8, marker=marker), row=2, col=2)
    fig.add_vline(x=ref, line_width=2, line_dash="dash", line_color="red", row=2, col=2)

    for (r, c), (xt, yt) in {
        (1, 1): ("Y-hat", "Residual"),
        (1, 2): ("Residual", "Count"),
        (2, 1): ("Sequence", "Residual"),
        (2, 2): ("Factor Importance", "Factor"),
    }.items():
        fig.update_xaxes(title_text=xt, row=r, col=c)
        fig.update_yaxes(title_text=yt, row=r, col=c)

    fig.update_layout(height=960, width=1280, title_text="Model Check Figure", showlegend=False)
    return fig
