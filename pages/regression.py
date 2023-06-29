import streamlit as st
import pandas as pd
import itertools

import datetime
import numpy as np
import io
import statsmodels.formula.api as smf
from scipy.stats import shapiro
from scipy import stats
# from statsmodels.graphics.gofplots import qqplot


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pickle


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

# @st.cache_data
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv().encode('utf-8')

def ols_reg(formula, df):

  model = smf.ols(formula, df)
  res = model.fit()
  df_result = df.copy()
  df_result['yhat'] = res.fittedvalues
  df_result['resid'] = res.resid

#   print(df_result.head())

  return res, df_result, model

# def acquire_qq_data(df_result_resid):
#   qqplot_data = qqplot(df_result_resid, line='s').gca().lines

#   df_qq = pd.DataFrame()
#   df_qq["x_point"] = qqplot_data[0].get_xdata()
#   df_qq["y_point"] = qqplot_data[0].get_ydata()

#   df_qq["x_line"] = qqplot_data[1].get_xdata()
#   df_qq["y_line"] = qqplot_data[1].get_ydata()

#   return df_qq



fig_size = [1280, 960]

st.title('Linear Regression Tool')

st.markdown("#### Author & License:")

st.markdown("**Kurt Su** (phononobserver@gmail.com)")

st.markdown("**This tool release under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license**")

st.markdown("               ")
st.markdown("               ")

# Provide dataframe example & relative url
reg_ex_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTnqEkuIqYHm1eDDF-wHHyQ-Jm_cvmJuyBT4otEFt0ZE0A6FEQyg1tqpWTU0PXIFA_jYRX8O-G6SzU8/pub?gid=0&single=true&output=csv"
# st.write("Factor Format Example File [link](%s)" % factor_ex_url)
st.markdown("### **Regression Data Format Example File [Demo File](%s)**" % reg_ex_url)


taguchi_ex_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR2pTVvSJJEWf1woRdODBYzBvJsHvgIgcJWAly2EDoNGm610xiMISNdaf8yq1f8h732zqel7v-vNon7/pub?gid=0&single=true&output=csv"
st.markdown("### **Taguchi Data Format Example File [Demo File](%s)**" % taguchi_ex_url)

uploaded_csv = st.file_uploader('#### 選擇您要上傳的CSV檔')

if uploaded_csv is not None:
    df_raw = pd.read_csv(uploaded_csv, encoding="utf-8")
    st.header('您所上傳的CSV檔內容：')
    st.dataframe(df_raw)
    # fac_n = df_fac.shape[1]

    ana_type = st.selectbox("Choose Analysis Mehthd", ["Regression Method", "Taguchi Method"])

    if ana_type == "Regression Method":

        select_list = list(df_raw.columns)
        # select_list
        response = st.selectbox("### Choose Response(y)", select_list)
        # response
        factor_list = select_list.copy()
        factor_list.remove(response)
        factor = st.multiselect(
            "### Choose Factor(x)", factor_list)
        if not factor:
            st.error("Please select at least one factor.")

        # factor_2nd order
        factor_2od_list = list()
        for j in factor:
            factor_2od_list.append("I("+j+" ** 2)")

        factor_2od = st.multiselect(
            "### Choose Factor 2nd Order(x^2)", factor_2od_list)
        if not factor_2od:
            st.error("Please select at 2nd order factor.")

        # factor_2od_rev = 

        # factor_interaction
        factor_inter_tmp = list(itertools.combinations(factor, 2))
        factor_inter_list =  list()
        for i in factor_inter_tmp:
            tmp = "*".join(i)
            factor_inter_list.append(tmp)

        factor_inter = st.multiselect(
            "### Choose Factor Interaction(x1 * x2)", factor_inter_list)
        if not factor_inter:
            st.error("Please select interaction factor.")
        # st.write(factor)
        # st.write(factor_inter_list)
        # factor
        # factor_final = factor + factor_inter  
        factor_final = factor + factor_2od + factor_inter




        if st.checkbox('Perform Analysis'):
            x_formula = "+".join(factor_final)
            formula = response + "~" + x_formula
            formula
            df_reg = df_raw.copy()
            # x_formula
            # formula
            # filter_list = factor + list(response)
            # filter_list
            # df_x = df_raw[factor]
            # df_x

            result, df_result, model = ols_reg(formula, df_reg)
            st.write(result.summary())

            alpha = 0.05
            f_num = len(result.tvalues)-1
            # dof = round(f_num/3, 0)
            dof = result.df_resid
            t_val = stats.t.ppf(1-alpha/2, dof)

            df_pareto = result.tvalues[1:].abs()
            df_pareto = df_pareto.sort_values(ascending=True)
            df_pareto = pd.DataFrame(df_pareto).reset_index(level=0)
            df_pareto.columns = ["factor", "t-value"]




            SW, sw_p_val = shapiro(df_result["resid"])
            # df_qq = acquire_qq_data(df_result["resid"])

            st.markdown("#### Normality Test P Value:%s " % round(sw_p_val,4))
            if sw_p_val >= 0.05:
                st.markdown("##### Residual is NOT normal distribution!!")
            else:
                st.markdown("##### Residual is normal distribution")    

            color_sequence = ["#65BFA1", "#A4D6C1", "#D5EBE1", "#EBF5EC", "#00A0DF", "#81CDE4", "#BFD9E2"]
            color_sequence = px.colors.qualitative.Pastel
            template = "simple_white"

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("yhat-residual-plot (random better)", "residual-histogram-plot (normal distribution better)", 
                                "redidual-sequence-plot (random better)", "pareto-plot (red line as criteria)"))

            fig.add_trace(go.Scatter(x=df_result["yhat"], y=df_result["resid"], mode="markers", 
                                    marker=dict(color='rgba(19, 166, 255, 0.6)')),
                        row=1, col=1)

            fig.add_trace(go.Histogram(x=df_result["resid"],
                                    marker=dict(color='rgba(19, 166, 255, 0.6)')),
                        row=1, col=2)

            fig.add_trace(go.Scatter(y=df_result["resid"], mode="lines+markers",
                                    marker=dict(color='rgba(19, 166, 255, 0.6)')),
                        row=2, col=1)

            fig.add_trace(go.Bar(x=df_pareto["t-value"], y=df_pareto["factor"], orientation='h', width=0.8,
                                marker=dict(color='rgba(19, 166, 255, 0.6)')
                                ),
                        row=2, col=2)
            fig.add_vline(x=t_val, line_width=2, line_dash='dash', line_color='red',
                        row=2, col=2)

            # fig.add_trace(go.Scatter(x=df_qq["x_line"], y=df_qq["y_line"], mode="lines"),
            #               row=2, col=2)

            fig.update_xaxes(title_text="Y-hat", row=1, col=1)
            fig.update_yaxes(title_text="Residual", row=1, col=1)

            fig.update_xaxes(title_text="Residual", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=2)

            fig.update_xaxes(title_text="Sequence", row=2, col=1)
            fig.update_yaxes(title_text="Residual", row=2, col=1)

            fig.update_xaxes(title_text="Factor Importance", row=2, col=2)
            fig.update_yaxes(title_text="Factor", row=2, col=2)

            fig.update_layout(height=fig_size[1], width=fig_size[0],
                            title_text="Model Check Figure",
                            showlegend=False)
            

            date = str(datetime.datetime.now()).split(" ")[0]

            mybuff = io.StringIO()
            fig_file_name = date + "_reg-judge.html"
            # fig_html = fig_pair.write_html(fig_file_name)
            fig.write_html(mybuff, include_plotlyjs='cdn')
            html_bytes = mybuff.getvalue().encode()

            st.plotly_chart(fig, use_container_width=True)

            csv = convert_df(df_result)
        
            # table_filename = doe_type + "_table_" + date
            result_file = date + "_lin-reg.csv"
            model_file_name = date + "_model.pickle"

            st.download_button(label='Download statistics result as CSV',  
                            data=csv, 
                            file_name=result_file,
                            mime='text/csv')
                
            st.download_button(label="Download figure",
                                data=html_bytes,
                                file_name=fig_file_name,
                                mime='text/html'
                                )
            
            st.download_button(label="Download Model",
                                data=pickle.dumps(result),
                                file_name=model_file_name,
                                # mime='application/octet-stream'
                                )

    if ana_type == "Taguchi Method":
        st.markdown("##### Under Construction!!")


