import streamlit as st
import pandas as pd
import itertools

import datetime
import numpy as np
import io
import plotly.express as px
# from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


def backend(fig_type, df_plot, para):
    # para
    cate_req = para["cate_req"]
    if cate_req == True:
        category = para["cate"]
    else:
        category = None


    fig_width = para["width"]
    fig_height = para["height"]

    color_sequence = ["#65BFA1", "#A4D6C1", "#D5EBE1", "#EBF5EC", "#00A0DF", "#81CDE4", "#BFD9E2"]
    color_sequence = px.colors.qualitative.Pastel
    template = "simple_white"

    if fig_type not in ["pair plot", "histogram", "distribution"]:
        fig_type
        y_var = para["y"]
    
    if fig_type not in ["pair plot", "interaction(DOE)"]:
        x_var = para["x"]


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
        bins = para["bins"]
        barmode = para["barmode"]
        fig = px.histogram(df_plot, x=x_var, nbins=bins, color=category,
                    color_discrete_sequence=color_sequence, template=template,
                    marginal="rug", hover_data=df_plot.columns,
                    # range_x=x_range, range_y=y_range,        
                    width=fig_width, height=fig_height
                    )       
        fig.update_layout(barmode=barmode)            
        if barmode == "overlay":
            # Reduce opacity to see both histograms 
            fig.update_traces(opacity=0.75)
    
    elif fig_type == "distribution":
        bin_size = para["bin_size"]
        if cate_req == False:
            his_data=[df_plot[x_var].to_numpy()]
            group_labels = [x_var]
        else:
            unique_cate = df_plot[category].unique()
            group_labels = []
            his_data = []
            for i in unique_cate:
                group_labels.append(str(i))
                df_his_temp = df_plot[df_plot[category] == i]
                his_temp = df_his_temp[x_var].to_numpy()
                his_data.append(his_temp)    
        # group_labels
        # his_data
        fig = ff.create_distplot(his_data, group_labels, bin_size=bin_size)

    
    elif fig_type == "pair plot":
        focus_factor = para["focus"]
        fig = px.scatter_matrix(df_plot, dimensions=focus_factor, color=category, 
                                color_discrete_sequence=color_sequence, template="plotly_white",
                            width=fig_width, height=fig_height)
        fig.update_traces(diagonal_visible=False, showupperhalf=False)


    elif fig_type in ["scatter", "bubble", "bubble animate"]:
        if fig_type == "scatter":
            size_var = None
            animate = None
        
        elif fig_type == "bubble":
            size_var = para["size"]
            animate = None
        
        elif fig_type == "bubble animate":
            size_var = para["size"]
            animate = para["animate"]

            
        fig = px.scatter(df_plot, x=x_var, y=y_var, size=size_var, size_max=45,
                animation_frame=animate, #animation_group=animate_g, 
                color=category, color_discrete_sequence=color_sequence, template=template, 
                # range_x=x_range, range_y=y_range, 
                # log_x=xlog_scale, log_y=ylog_scale,
                width=fig_width, height=fig_height
                )

    elif fig_type == "interaction(DOE)":
        focus_factor = para["focus"]
        iter_list =  list(itertools.combinations(focus_factor, 2))
        inter_factor = st.selectbox(
        "### Select inter-action factor", 
        iter_list,
        )
                
        # for inter_factor in iter_list:
            
        # inter_factor
        factor_1 = inter_factor[0]
        factor_2 = inter_factor[1]

        select_list = list(inter_factor)
        select_list.append(y_var)

        df_interact_plot = df_plot[select_list]
        df_interact_group = df_interact_plot.groupby(list(inter_factor), as_index=False).mean()
        # color=factor_2, 
        fig_scatter = px.scatter(df_interact_plot, x=factor_1, y=y_var, 
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

        )



    if fig_type not in  ["pair plot", "interaction(DOE)", "histogram", "distribution"]:

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

    return fig



def main():
    st.title('EDA (Exploratory Data Analysis) Tool')

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

    fig_type = st.selectbox(
        "### Choose figure type", 
        ["box", "violin", "histogram", "distribution", "pair plot", "interaction(DOE)", "scatter", "bubble", "bubble animate", ],
    )

    select_list = list(df_raw.columns)

    para = {}
    cate_req = st.checkbox('Category Required')
    para["cate_req"] = cate_req
    if cate_req == True:
        category = st.selectbox(
            "### Choose category", select_list)
        para["cate"] = category
    

 
  
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
        # filter_item        
    
    if fig_type == "pair plot":
        
        focus_factor = st.multiselect(
            "### Choose focus factor", select_list,
        )
        para["focus"] = focus_factor

    else:
        if fig_type == "histogram":
            x_var = st.selectbox("## Choose x variable", select_list)
            para["x"] = x_var
            bins = st.number_input('Choose bins', min_value=5, value=10, max_value=100, step=5) 
            para["bins"] = bins
            barmode = st.selectbox("## Choose Overlay/Stack Bar", ["overlay", "stack"])
            para["barmode"] = barmode

        elif fig_type == "distribution":
            x_var = st.selectbox("## Choose x variable", select_list)
            para["x"] = x_var
            bin_size = st.number_input('Choose bin size', min_value=0.00, value=1.00, step=0.01) 
            para["bin_size"] = bin_size
            # barmode = st.selectbox("## Choose Overlay/Stack Bar", ["overlay", "stack"])
            # para["barmode"] = barmode

        else:
            y_var = st.selectbox("## Choose y variable", select_list)
            para["y"] = y_var
            if not y_var:
                st.error("Please select one y variable.")

            x_list = select_list.copy()
            x_list.remove(y_var)

            if fig_type == "interaction(DOE)":
                focus_factor = st.multiselect(
                    "### Choose focus factor", x_list,
                )
                para["focus"] = focus_factor

            else: 
                x_var = st.selectbox(
                "### Choose x variable", x_list)
                para["x"] = x_var

                if not x_var:
                    st.error("Please select one x variable.")
            
                if fig_type in ["bubble", "bubble animate"]:
                    size_var = st.selectbox(
                            "### Choose bubble size", x_list)
                    
                    para["size"] = size_var
                    if fig_type == "bubble animate":
                        animate = st.selectbox(
                            "### Choose animate", x_list)
                        para["animate"] = animate

    st.markdown("----------------")  

        
    size_col1, size_col2 = st.columns(2)
    with size_col1:
        fig_width = st.number_input('Figure Width', min_value=640, value=1280, max_value=5120, step=320) 
        para["width"] = fig_width
    with size_col2:
        fig_height = st.number_input('Figure Height', min_value=480, value=960, max_value=3840, step=240) 
        para["height"] = fig_height

    st.markdown("----------------")  

    if filter_req == True:
        df_plot = df_raw[df_raw[filter_para].isin(filter_item)].copy()
        st.markdown("----------------") 
        st.markdown("#### Filter DataFrame")
        df_plot
        st.markdown("----------------")  
    else:
        df_plot = df_raw.copy()

    fig = backend(fig_type, df_plot, para)
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