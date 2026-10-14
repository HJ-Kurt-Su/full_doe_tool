import numpy as np
import pandas as pd
import plotly.express as px
import pyDOE3
import streamlit as st

import tools

TAGUCHI_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQMX31nAb2tY_U33hGK-Y3Up-ta7sl55grC591QmS--kqz9EhQyCvxYMe9_fG3YnPIoZuiPlkMQ-zg_/pubhtml"
TAGUCHI_ARRAYS = {"L4_3F2L": 0, "L8_7F2L": 1, "L9_4F3L": 2, "L12_11F2L": 3, "L16_15F2L": 4, "L16b_5F4L": 5, "L18_1F2L-7F3L": 6}
GSD_URL = "https://doi.org/10.1021/acs.analchem.7b00506"


@st.cache_data(ttl=3600, show_spinner="Loading Taguchi array...")
def taguchi_table(array):
    return pd.read_html(TAGUCHI_URL)[TAGUCHI_ARRAYS[array]].iloc[1:, 2:].apply(pd.to_numeric, errors="coerce")


@st.cache_data(show_spinner=False)
def doe_code_table(doe_type, fac_n, para):
    """Coded DoE matrix. Cached so random designs (latin-hypercube) stay stable across reruns."""
    para = dict(para)
    if doe_type == "response surface":
        return pd.DataFrame(pyDOE3.ccdesign(fac_n))
    if doe_type == "2 lv full":
        return pd.DataFrame(pyDOE3.ff2n(fac_n))
    if doe_type == "taguchi":
        return taguchi_table(para["array"]).iloc[:, :fac_n]
    if doe_type == "latin-hypercube":
        return pd.DataFrame(pyDOE3.lhs(fac_n, samples=int(para["samples"]), criterion=para["criteria"]))
    # gsd
    arr = pyDOE3.gsd(list(para["levels"]), para["reduction"], n=para["comp"])
    return pd.DataFrame(arr if para["comp"] == 1 else arr[para["fold"]])


def to_real_levels(df_code, df_fac):
    """Linearly map each coded column onto the factor's [low, high] levels."""
    out = df_code.copy()
    out.columns = df_fac.columns
    for col in out.columns:
        lo, hi = out[col].min(), out[col].max()
        out[col] = np.interp(out[col], [lo, hi], [df_fac[col].iloc[0], df_fac[col].iloc[1]])
    return out


def doe_controls(doe_type, df_fac):
    """Render the parameters of the chosen DoE type; return them as a hashable tuple."""
    if doe_type == "taguchi":
        st.markdown(f"#### Taguchi Array [Link]({TAGUCHI_URL})")
        return (("array", st.selectbox("Choose Taguchi Array:", list(TAGUCHI_ARRAYS))),)

    if doe_type == "latin-hypercube":
        criteria = st.selectbox("Choose Sample Method:", (None, "center", "maximin", "centermaximin", "correlation"))
        samples = st.number_input("Insert a number", min_value=1, value=10)
        return (("criteria", criteria), ("samples", int(samples)))

    if doe_type == "gsd":
        st.markdown(f"#### GSD Introduction [Link]({GSD_URL})")
        st.markdown("###### Define Level for Each Factor:")
        levels = tuple(
            int(st.number_input(col, min_value=2, value=3, step=1, key=f"gsd_lv_{col}")) for col in df_fac.columns
        )
        st.markdown("--------------------------")
        st.markdown("###### Define GSD Rest Parameter:")
        reduction = st.number_input("Reduction", min_value=2, value=4, max_value=8, step=1)
        comp = st.number_input("Complementary", min_value=1, max_value=4, step=1)
        fold = st.number_input("Select fold number", min_value=0, max_value=comp - 1, step=1)
        return (("levels", levels), ("reduction", int(reduction)), ("comp", int(comp)), ("fold", int(fold)))

    return ()


def main():
    st.title("DoE (Design of Experiment) Tool")

    uploaded = st.sidebar.file_uploader(tools.UPLOAD_LABEL, type=tools.TABLE_TYPES)
    df_fac = tools.load_table(uploaded, tools.DEMO_DOE)
    st.dataframe(df_fac)

    doe_type = st.selectbox("Choose DoE Type:", ("2 lv full", "response surface", "taguchi", "gsd", "latin-hypercube"))
    para = doe_controls(doe_type, df_fac)
    st.write("You selected:", doe_type)

    width, height = tools.fig_size_inputs()

    fac_n = df_fac.shape[1]
    try:
        df_code = doe_code_table(doe_type, fac_n, para)
    except Exception as e:
        st.error(f"Cannot build the DoE table: {e}")
        st.stop()
    if df_code.shape[1] < fac_n:
        st.error("Factor Q'ty is not enough, please select another array!!")
        st.stop()

    df_resp = to_real_levels(df_code, df_fac)
    st.dataframe(df_resp)

    fig = px.scatter_matrix(df_resp, dimensions=df_resp.columns, color_discrete_sequence=tools.COLOR_SEQ,
                            width=width, height=height)
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    st.plotly_chart(fig, width="stretch")

    st.divider()
    tools.download_file("Input Result File Name", "Download DOE result as CSV", df_resp, "csv", "result_data")
    st.divider()
    tools.download_file("Input Figure File Name", "Download figure as HTML", fig, "html", "figure")
    st.divider()


main()
