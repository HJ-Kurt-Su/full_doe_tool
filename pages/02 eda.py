import streamlit as st
import pandas as pd
import itertools

import datetime
import numpy as np
import io
import plotly.express as px
# from plotly.subplots import make_subplots
import plotly.graph_objects as go

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def backend():
    st.markdown("Under Construction")

def main():
    st.title('EDA (Exploratory Data Analysis) Tool')

    st.markdown("#### Author & License:")

    st.markdown("**Kurt Su** (phononobserver@gmail.com)")

    st.markdown("**This tool release under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license**")

    st.markdown("               ")
    st.markdown("               ")

    uploaded_csv = st.sidebar.file_uploader("請上傳您的 CSV 檔案", type=["csv"])

    if uploaded_csv is not None:
        df_raw = pd.read_csv(uploaded_csv, encoding="utf-8")
        st.header('您所上傳的CSV檔內容：')

    else:
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTnqEkuIqYHm1eDDF-wHHyQ-Jm_cvmJuyBT4otEFt0ZE0A6FEQyg1tqpWTU0PXIFA_jYRX8O-G6SzU8/pub?gid=0&single=true&output=csv"
        st.header('未上傳檔案，以下為 Demo：')
        df_raw = pd.read_csv(url, encoding="utf-8")
    
    st.dataframe(df_raw)
    st.markdown("**Data Description**")
    st.dataframe(df_raw.describe())

    select_list = list(df_raw.columns)
    # select_list
    y_var = st.selectbox("### Choose y variable", select_list)
    if not y_var:
        st.error("Please select one y variable.")

    # response
    x_list = select_list.copy()
    x_list.remove(y_var)
    x_var = st.selectbox(
        "### Choose x variable", x_list)
    if not x_var:
        st.error("Please select one x variable.")

    st.markdown("----------------")  

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        # st.markdown("#### **Choose figure type**")
        fig_type = st.selectbox(
            "### Choose figure type", 
            ["box", "violin", "histogram", "pair plot", "interaction(DOE)", "scatter", "bubble", "bubble animate", ],
        )
        
    with fig_col2:
        # if uploaded_csv is not None:
        category = st.selectbox(
            "### Choose category", x_list)
        
    size_col1, size_col2 = st.columns(2)
    with size_col1:

        fig_width = st.number_input('Figure Width', min_value=640, value=1280, max_value=5120, step=320) 
        
    with size_col2:
        fig_height = st.number_input('Figure Height', min_value=480, value=960, max_value=3840, step=240) 

    st.markdown("----------------")  

    if fig_type == "histogram":
        # st.markdown("##### Choose ")
        bins = st.number_input('Choose bins', min_value=5, value=10, max_value=100, step=5) 

    elif fig_type == "pair plot":
        focus_factor = st.multiselect(
            "### Choose focus factor", df_raw.columns,
        )

    elif fig_type == "interaction(DOE)":
        focus_factor = st.multiselect(
            "### Choose focus factor", x_list,
        )

    elif fig_type in ["bubble", "bubble animate"]:
        size_var = st.selectbox(
                "### Choose bubble size", x_list)
        
        if fig_type == "bubble animate":
            animate = st.selectbox(
                "### Choose animate", x_list)


    # if st.checkbox('Plot'):
    df_plot = df_raw.copy()

    color_sequence = ["#65BFA1", "#A4D6C1", "#D5EBE1", "#EBF5EC", "#00A0DF", "#81CDE4", "#BFD9E2"]
    color_sequence = px.colors.qualitative.Pastel
    template = "simple_white"


    if fig_type == "box":
        fig = px.box(df_plot, x = x_var, y=y_var, color=category, points="all", 
                        color_discrete_sequence=color_sequence, template=template, 
                        # range_y=y_range, 
                        width=fig_width, height=fig_height,
                        hover_data=df_plot.columns,
                    )

    elif fig_type == "violin":
        fig = px.violin(df_plot, x=x_var, y=y_var, color=category, points="all",
                        box=False, color_discrete_sequence=color_sequence, 
                        template=template, #range_y=y_range, 
                        width=fig_width, height=fig_height,
                        hover_data=df_plot.columns,
                        )
        
    elif fig_type == "histogram":
        fig = px.histogram(df_plot, x=y_var, nbins=bins, color=category,
                    color_discrete_sequence=color_sequence, template=template,
                    # range_x=x_range, range_y=y_range,        
                    width=fig_width, height=fig_height
                    )
    
    elif fig_type == "pair plot":
        fig = px.scatter_matrix(df_plot, dimensions=focus_factor, color=category, 
                                color_discrete_sequence=color_sequence, template="plotly_white",
                            width=fig_width, height=fig_height)
        fig.update_traces(diagonal_visible=False, showupperhalf=False)


    elif fig_type in ["scatter", "bubble", "bubble animate"]:
        if fig_type == "scatter":
            size_var = None
            animate = None
        
        if fig_type == "bubble":
            animate = None
            
        fig = px.scatter(df_plot, x=x_var, y=y_var, size=size_var, size_max=45,
                animation_frame=animate, #animation_group=animate_g, 
                color=category, color_discrete_sequence=color_sequence, template=template, 
                # range_x=x_range, range_y=y_range, 
                # log_x=xlog_scale, log_y=ylog_scale,
                width=fig_width, height=fig_height
                )

    elif fig_type == "interaction(DOE)":
        iter_list =  list(itertools.combinations(focus_factor, 2))

        # j=1
                
        for inter_factor in iter_list:
            inter_factor
            factor_1 = inter_factor[0]
            factor_2 = inter_factor[1]
            # print(list(i).append(y))
            # print(factor_1)
            # print(factor_2)
            select_list = list(inter_factor)
            select_list.append(y_var)
            # print(select_list)



            df_interact_plot = df_plot[select_list]
            df_interact_group = df_interact_plot.groupby(list(inter_factor), as_index=False).mean()


            fig_scatter = px.scatter(df_interact_plot, x=factor_1, y=y_var, color=factor_2, 
                                    width=fig_width, height=fig_height,
                                    )
            fig_line = px.line(df_interact_group, x=factor_1, y=y_var, color=factor_2, width=fig_width, height=fig_height, markers=True)
            fig = go.Figure(data=fig_scatter.data + fig_line.data)
            fig.update_layout(
                autosize=False,
                width=fig_width,
                height=fig_height,
                xaxis_title=factor_1,
                yaxis_title=y_var,
                title=factor_2,
                legend_title=factor_2,
                legend_orientation="h",
                # color_discrete_sequence=color_sequence, template=template

                # margin=dict(
                #     l=50,
                #     r=50,
                #     b=100,
                #     t=100,
                #     pad=4
                # ),
                # paper_bgcolor="LightSteelBlue",
            )
            st.plotly_chart(fig, use_container_width=True)



    if fig_type != "pair plot" or fig_type != "interaction(DOE)":

        fig.update_layout(
            xaxis_title=x_var,
            yaxis_title=y_var,
            # legend_title="Legend Title",
            font=dict(
                family="Courier New, monospace",
                size=18
                # color="RebeccaPurple"
            ),
            yaxis = dict(tickfont = dict(size=25)),
            xaxis = dict(tickfont = dict(size=15))
        )

    if fig_type != "interaction(DOE)":

        st.plotly_chart(fig, use_container_width=True)

    date = str(datetime.datetime.now()).split(" ")[0]
    mybuff = io.StringIO()
    fig_file_name = date + "_" + fig_type + ".html"
    # fig_html = fig_pair.write_html(fig_file_name)
    fig.write_html(mybuff, include_plotlyjs='cdn')
    html_bytes = mybuff.getvalue().encode()

    st.download_button(label="Download figure",
                        data=html_bytes,
                        file_name=fig_file_name,
                        mime='text/html'
                        )

#%% Web App 頁面
if __name__ == '__main__':
    main()

# %%