import itertools

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde

import tools

FIG_TYPES = ["box", "violin", "histogram", "distribution", "pair plot", "interaction(DOE)", "scatter", "line", "bubble", "bubble animate"]
TEMPLATE = "simple_white"


def controls(fig_type, cols):
    """Render the variable pickers needed by `fig_type`; return them as a dict."""
    para = {}
    if fig_type == "pair plot":
        para["focus"] = st.multiselect("### Choose focus factor", cols)
    elif fig_type == "histogram":
        para["x"] = st.selectbox("## Choose x variable", cols)
        para["bins"] = st.number_input("Choose bins", min_value=5, value=10, max_value=100, step=5)
        para["barmode"] = st.selectbox("## Choose Overlay/Stack Bar", ["overlay", "stack"])
    elif fig_type == "distribution":
        para["x"] = st.selectbox("## Choose x variable", cols)
        para["bin_size"] = st.number_input("Choose bin size", min_value=0.00, value=1.00, step=0.01)
    else:
        para["y"] = st.selectbox("## Choose y variable", cols)
        x_list = [c for c in cols if c != para["y"]]
        if fig_type == "interaction(DOE)":
            para["focus"] = st.multiselect("### Choose focus factor", x_list)
        else:
            para["x"] = st.selectbox("### Choose x variable", x_list)
            if fig_type == "line":
                para["markers"] = st.checkbox("Show markers", value=True)
                para["sort_x"] = st.checkbox("Sort by x", value=True)
            if fig_type in ("bubble", "bubble animate"):
                para["size"] = st.selectbox("### Choose bubble size", x_list)
            if fig_type == "bubble animate":
                para["animate"] = st.selectbox("### Choose animate", x_list)
    return para


def interaction_figure(df, focus, y, width, height):
    pairs = list(itertools.combinations(focus, 2))
    if not pairs:
        st.info("Please choose at least two focus factors.")
        st.stop()
    f1, f2 = st.selectbox("### Select inter-action factor", pairs)
    df_sel = df[[f1, f2, y]]
    df_mean = df_sel.groupby([f1, f2], as_index=False).mean()
    scatter = px.scatter(df_sel, x=f1, y=y, color=f2)
    line = px.line(df_mean, x=f1, y=y, color=f2, markers=True)
    fig = go.Figure(data=scatter.data + line.data)
    fig.update_layout(autosize=False, width=width, height=height, xaxis_title=f1, yaxis_title=y,
                      title=f2, legend_title=f2)
    return fig


def distribution_figure(groups, bin_size, width, height):
    """Density histogram + KDE curve + rug for each (label, values) group."""
    fig = go.Figure()
    for i, (label, values) in enumerate(groups):
        color = tools.COLOR_SEQ[i % len(tools.COLOR_SEQ)]
        common = dict(name=label, legendgroup=label, marker_color=color)
        fig.add_histogram(x=values, histnorm="probability density", xbins=dict(size=bin_size), opacity=0.6, **common)
        if len(values) > 1 and values.std() > 0:
            xs = np.linspace(values.min(), values.max(), 200)
            fig.add_scatter(x=xs, y=gaussian_kde(values)(xs), mode="lines", showlegend=False,
                            line_color=color, legendgroup=label)
        fig.add_scatter(x=values, y=np.zeros(len(values)), mode="markers", showlegend=False, legendgroup=label,
                        marker=dict(color=color, symbol="line-ns-open", size=10))
    fig.update_layout(barmode="overlay", template=TEMPLATE, width=width, height=height)
    return fig


def build_figure(fig_type, df, category, para, width, height):
    common = dict(color=category, color_discrete_sequence=tools.COLOR_SEQ, template=TEMPLATE,
                  width=width, height=height)
    hover = dict(hover_data=df.columns)
    x, y = para.get("x"), para.get("y")

    if fig_type == "box":
        fig = px.box(df, x=x, y=y, points="all", **hover, **common)
    elif fig_type == "violin":
        fig = px.violin(df, x=x, y=y, points="all", box=False, **hover, **common)
    elif fig_type == "histogram":
        fig = px.histogram(df, x=x, nbins=para["bins"], marginal="rug", **hover, **common)
        fig.update_layout(barmode=para["barmode"])
        if para["barmode"] == "overlay":
            fig.update_traces(opacity=0.75)
    elif fig_type == "distribution":
        groups = [(x, df[x])] if category is None else [(str(k), g[x]) for k, g in df.groupby(category)]
        fig = distribution_figure([(k, g.to_numpy(dtype=float)) for k, g in groups], para["bin_size"], width, height)
    elif fig_type == "line":
        df_line = df.sort_values(x) if para["sort_x"] else df
        fig = px.line(df_line, x=x, y=y, markers=para["markers"], **hover, **common)
    elif fig_type == "pair plot":
        if not para["focus"]:
            st.info("Please choose at least one focus factor.")
            st.stop()
        fig = px.scatter_matrix(df, dimensions=para["focus"], color=category,
                                color_discrete_sequence=tools.COLOR_SEQ, template="plotly_white",
                                width=width, height=height)
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
    elif fig_type == "interaction(DOE)":
        fig = interaction_figure(df, para["focus"], y, width, height)
    else:  # scatter / bubble / bubble animate
        fig = px.scatter(df, x=x, y=y, size=para.get("size"), size_max=45,
                         animation_frame=para.get("animate"), **common)

    if fig_type in ("box", "violin", "scatter", "line", "bubble", "bubble animate"):
        fig.update_layout(xaxis_title=x, yaxis_title=y, font=dict(family="Courier New, monospace", size=18),
                          yaxis=dict(tickfont=dict(size=25)), xaxis=dict(tickfont=dict(size=15)))
    return fig


def main():
    st.title("EDA (Exploratory Data Analysis) Tool")

    uploaded = st.sidebar.file_uploader(tools.UPLOAD_LABEL, type=tools.TABLE_TYPES)
    df_raw = tools.load_table(uploaded, tools.DEMO_REG)
    st.dataframe(df_raw)
    st.markdown("**Data Description**")
    st.dataframe(df_raw.describe())

    fig_type = st.selectbox("### Choose figure type", FIG_TYPES)
    cols = list(df_raw.columns)

    category = st.selectbox("### Choose category", cols) if st.checkbox("Category Required") else None
    df_plot = tools.filter_data(df_raw)

    para = controls(fig_type, cols)
    st.divider()
    width, height = tools.fig_size_inputs()
    st.divider()

    fig = build_figure(fig_type, df_plot, category, para, width, height)
    st.plotly_chart(fig, width="stretch")

    st.divider()
    tools.download_file("Input Figure File Name", "Download figure as HTML", fig, "html", "figure")
    st.divider()


main()
