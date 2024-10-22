import streamlit as st
import pandas as pd
import itertools

import numpy as np

import statsmodels.formula.api as smf
import statsmodels.api as sm 
from scipy.stats import shapiro
from scipy import stats


# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix

# from sklearn.metrics import roc_curve, auc


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import tools




def ols_reg(formula, df):

  model = smf.ols(formula, df)
  res = model.fit()
  df_result = df.copy()
  df_result['yhat'] = res.fittedvalues
  df_result['resid'] = res.resid

#   print(df_result.head())

  return res, df_result, model




# def clf_score_log(y, y_predict, gui_key):
#         thresh_key = gui_key["threshold"]
#         thresh_check_key = gui_key["threshold_check"]
#         fpr, tpr, threshold = roc_curve(y, y_predict)
#         roc_auc = auc(fpr,tpr)
#         # df_y
#         # y_pred

#         st.markdown("### AUC is: %s" % round(roc_auc,4))

#         dict_sum = {"FPR": fpr, "TPR": tpr, "Threshold": threshold}
#         df_roc_data = pd.DataFrame.from_dict(dict_sum)
#         st.subheader("ROC Data")
#         df_roc_data
        
#         fig_size=640

#         fig_roc = px.line(df_roc_data, x="FPR", y="TPR", markers=False, labels=roc_auc, 
#                           range_x=[0,1], range_y=[0,1.1], width=fig_size, height=fig_size)
        
#         st.subheader("ROC Figure")
#         st.plotly_chart(fig_roc, use_container_width=True)

#         st.subheader("Accuracy Judge")
#         key_in = st.checkbox("Key-in threshold", key=thresh_check_key)
#         if key_in == True:
#             threshold_cut = st.number_input("Key-in thershold value", min_value=0.0001, max_value=0.9999, value=0.5)
#         else:
#             threshold_cut = st.selectbox("Choose Threshold", threshold, key=thresh_key)

#         y_pred_code = y_predict.map(lambda x: 1 if x >= threshold_cut else 0)
#         # y_pred
#         # y_pred_code
#         acc = accuracy_score(y, y_pred_code)
#         cof_mx = confusion_matrix(y, y_pred_code)
#         df_cof_mx  = pd.DataFrame(cof_mx)
#         df_cof_mx = df_cof_mx.rename(columns={0: "Predict Pass", 1: "Predict Fail"}, index={0: "Real Pass", 1: "Real Fail"})
#         risk_qty = cof_mx[1, 0]
#         risk =  risk_qty/y_pred_code.size
  
#         st.markdown("### Accuracy is: %s" % round(acc,4))
#         st.markdown("### Risk is: %s" % round(risk,4))
#         st.markdown("### Confusion Matrix:")
#         st.dataframe(df_cof_mx)

#         st.markdown("---")
#         return df_roc_data, fig_roc



# def acquire_qq_data(df_result_resid):
#   qqplot_data = qqplot(df_result_resid, line='s').gca().lines

#   df_qq = pd.DataFrame()
#   df_qq["x_point"] = qqplot_data[0].get_xdata()
#   df_qq["y_point"] = qqplot_data[0].get_ydata()

#   df_qq["x_line"] = qqplot_data[1].get_xdata()
#   df_qq["y_line"] = qqplot_data[1].get_ydata()

#   return df_qq
# color_sequence = ["#65BFA1", "#A4D6C1", "#D5EBE1", "#EBF5EC", "#00A0DF", "#81CDE4", "#BFD9E2"]
# color_sequence = px.colors.qualitative.Pastel
# template = "simple_white"
  
def backend(df_reg, formula, fig_size):
    # df_reg = df_raw.copy()

    result, df_result, model = ols_reg(formula, df_reg)

    alpha = 0.05
    # f_num = len(result.tvalues)-1
    # dof = round(f_num/3, 0)
    dof = result.df_resid
    t_val = stats.t.ppf(1-alpha/2, dof)

    df_pareto = result.tvalues[1:].abs()
    df_pareto = df_pareto.sort_values(ascending=True)
    df_pareto = pd.DataFrame(df_pareto).reset_index(level=0)
    df_pareto.columns = ["factor", "t-value"]


    SW, sw_p_val = shapiro(df_result["resid"])
    # df_qq = acquire_qq_data(df_result["resid"])



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
 
    return result, df_result, fig, sw_p_val


def backend_tgch(df_raw, factors, responses, target_type, fig_size):

    df_tgch = df_raw.melt(id_vars=factors, value_vars=responses, var_name="Response", value_name="Y")
    df_tgch
    fil_list = factors.copy()
    # factors
    fil_list.append("Y") 
    # fil_list
    df_mean = df_tgch[fil_list].groupby(factors, as_index=False).mean()
    df_mean.rename(columns = {"Y":'Mean'}, inplace = True)

    # df_mean
    df_tgch_use = df_tgch.drop(columns=['Response'])
    if target_type=="nominal best":
        df_std = df_tgch_use.groupby(factors, as_index=False).std()
        df_std.rename(columns = {"Y":'Std'}, inplace = True)
        df_taguchi_summary = pd.merge(df_mean, df_std, on=factors)
        df_taguchi_summary["SN"] = -10 * np.log10(df_taguchi_summary["Std"].pow(2)/df_taguchi_summary["Mean"].pow(2))

    elif target_type=="smaller best":
        df_tgch_use["Sum_Square"] = df_tgch_use["Y"].pow(2)
        df_sum_square = df_tgch_use.groupby(factors, as_index=False).sum()
        df_taguchi_summary = pd.merge(df_mean, df_sum_square, on=factors)
        # df_taguchi_summary
        df_taguchi_summary["SN"] = -10 * np.log10(df_taguchi_summary["Sum_Square"]/len(responses))
        

    elif target_type=="larger best":
        df_tgch_use["Sum_Square_Inv"] = 1/df_tgch_use["Y"].pow(2)
        df_sum_square = df_tgch_use.groupby(factors, as_index=False).sum()
        df_taguchi_summary = pd.merge(df_mean, df_sum_square, on=factors)
        df_taguchi_summary["SN"] = -10 * np.log10(df_taguchi_summary["Sum_Square_Inv"]/len(responses))
        # df_taguchi_summary
    # df_taguchi_summary

    df_factor_summary = pd.DataFrame()

    for i in factors:
        # i
        filter_list = [i, "Mean", "SN"]
        # filter_list
        df_mean_tmp = df_taguchi_summary.groupby([i], as_index=False).mean()

        df_mean_tmp = df_mean_tmp[filter_list]
        # df_mean_tmp
        df_factor_summary = pd.concat([df_factor_summary,df_mean_tmp], ignore_index=True)
        # df_factor_summary
    df_factor_summary = df_factor_summary[factors+["Mean", "SN"]]
    # df_factor_summary


    # color_sequence = ["#65BFA1", "#A4D6C1", "#D5EBE1", "#EBF5EC", "#00A0DF", "#81CDE4", "#BFD9E2"]
    # color_sequence = px.colors.qualitative.Pastel
    # template = "simple_white"


    fig_mean = make_subplots(rows=1, cols=len(factors), subplot_titles=factors)
    fig_sn = make_subplots(rows=1, cols=len(factors), subplot_titles=factors)

    col_location = 1
    for i in factors:

        fig_mean.add_trace(
            go.Scatter(x=df_factor_summary[i], y=df_factor_summary["Mean"]),
            row=1, col=col_location
        )

        fig_sn.add_trace(
            go.Scatter(x=df_factor_summary[i], y=df_factor_summary["SN"]),
            row=1, col=col_location
        ) 
        col_location+=1
        # print(col_location)

    fig_mean.update_layout(height=fig_size[1], width=fig_size[0], title_text="Mean")
    fig_sn.update_layout(height=fig_size[1], width=fig_size[0], title_text="SN")

    return df_factor_summary, df_taguchi_summary, fig_mean, fig_sn


def backend_tgch_ana(df_fac_summary, factor_list, sn_average, mean_average):
    df_sn_max=pd.DataFrame()
    factor_index_list = list()

    for i in factor_list:
        # factor_list

        df_factor = df_fac_summary[df_fac_summary[i].notna()]
        df_sn_tmp = df_factor.loc[df_factor.loc[:,"SN"].idxmax(),:]
        factor_index_list.append(list(df_factor.index))
        # print(pd.DataFrame(sn_tmp).T)
        df_sn_max = pd.concat([df_sn_max,pd.DataFrame(df_sn_tmp).T], ignore_index=True)

    max_sn = df_sn_max["SN"].sum() - (len(factor_list)-1) * sn_average
    max_sn_mean = df_sn_max["Mean"].sum() - (len(factor_list)-1) * mean_average
    combination_list = list(itertools.product(*factor_index_list))
    

    df_result_all = pd.DataFrame()
    for j in combination_list:
        comb_tmp = df_fac_summary.loc[list(j)]
        factor_lv_tmp = comb_tmp[factor_list]
        factor_lv_tmp2 = np.array(factor_lv_tmp)
        # print(aac.diagonal())
        factor_lv=list(factor_lv_tmp2.diagonal())
        each_sn = comb_tmp["SN"].sum() - (len(factor_list)-1) * sn_average
        each_mean = comb_tmp["Mean"].sum() - (len(factor_list)-1) * mean_average

        factor_lv.append(each_mean)
        factor_lv.append(each_sn)
        
        df_result_tmp = pd.DataFrame(factor_lv)
        df_result_all = pd.concat([df_result_all,df_result_tmp.T], ignore_index=True)

    df_result_all.columns=factor_list+["Mean", "SN"]

    return df_sn_max, max_sn, max_sn_mean, df_result_all


def main():

    fig_size = [1280, 960]

    st.title('Linear Regression Tool')

    st.markdown("               ")
    st.markdown("               ")


    uploaded_raw = st.sidebar.file_uploader('#### 選擇您要上傳的CSV檔', type=["csv", "xlsx"])
    # uploaded_csv = st.file_uploader('#### 選擇您要上傳的CSV檔')

    ana_type = st.selectbox("Choose Analysis Mehthd", ["Regression Method", "Taguchi Method", "2 LV Classification"])

    if ana_type == "Regression Method":
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTnqEkuIqYHm1eDDF-wHHyQ-Jm_cvmJuyBT4otEFt0ZE0A6FEQyg1tqpWTU0PXIFA_jYRX8O-G6SzU8/pub?gid=0&single=true&output=csv"

    elif ana_type == "Taguchi Method":
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR2pTVvSJJEWf1woRdODBYzBvJsHvgIgcJWAly2EDoNGm610xiMISNdaf8yq1f8h732zqel7v-vNon7/pub?gid=0&single=true&output=csv"

    elif ana_type == "2 LV Classification":
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
    

    if ana_type == "Regression Method":

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

        if uploaded_raw is None:
            st.header('未上傳檔案，以下為 Demo：')
            response = "Y1"
            factor = ["R", "r", "t"]

        factor_final = factor + factor_2od + factor_inter

        # scaler = StandardScaler()
        # df_ttmp = df_reg[factor]
        # df_sc = scaler.fit_transform(df_ttmp)
        # df_sc

        # if st.checkbox('Perform Analysis'):
        x_formula = "+".join(factor_final)
        formula = response + "~" + x_formula
        st.subheader(formula)
        result, df_result, fig, sw_p_val = backend(df_reg, formula, fig_size)
        st.write(result.summary())
        
        st.markdown("#### Normality Test P Value:%s " % round(sw_p_val,4))
        if sw_p_val >= 0.05:
            st.markdown("##### Residual is NOT normal distribution!!")
        else:
            st.markdown("##### Residual is normal distribution")    

        st.plotly_chart(fig, use_container_width=True)

        tools.reg_save(df_result, fig, result)

        predict_performance = st.checkbox("Predict New Data & Check Performance", key="reg")

        if predict_performance == True:
            st.subheader("**Predict Portion**")
        
            uploaded_df = st.file_uploader('#### 選擇您要上傳的CSV檔', type=["csv", "xlsx"], key="predict")

            if uploaded_df is None:
                
                url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTnqEkuIqYHm1eDDF-wHHyQ-Jm_cvmJuyBT4otEFt0ZE0A6FEQyg1tqpWTU0PXIFA_jYRX8O-G6SzU8/pub?gid=0&single=true&output=csv"

            else: 
                url = None
            df_test = tools.upload_file(uploaded_df, url)

            y_hat = result.predict(df_test)
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
                



    if ana_type == "Taguchi Method":

        responses = st.multiselect(
            "### Choose Response(Yn)", select_list)
        if not responses:
            st.error("Please select at least one factor.")
        

        factor_list = select_list.copy()
        
        for i in responses:
            # i
            factor_list.remove(i)

        factors = st.multiselect(
            "### Choose Factors(Xn)", factor_list)
        if not factors:
            st.error("Please select at least one factor.")

        target_type = st.selectbox("### Choose Traget", ["larger best", "nominal best", "smaller best"])

        if uploaded_raw is None:
            st.header('未上傳檔案，以下為 Demo：')
            responses = ["Z1", "Z2", "Z3"]
            factors = ["R", "r", "t"]
            factor_list = ["R", "r", "t"]

        df_fac_summary,  df_tgch_summary, fig_mean, fig_sn = backend_tgch(df_reg, factors, responses, target_type, fig_size)
        mean_average = df_tgch_summary["Mean"].mean()
        sn_average = df_tgch_summary["SN"].mean()

        st.subheader("Taguchi Factor Lv vs. Mean & SN")
        st.dataframe(df_fac_summary)

        tools.download_file(name_label="Input Result File Name",
                      button_label='Download Taguchi result as CSV',
                      file=df_fac_summary,
                      file_type="csv",
                      gui_key="tgch_factor_data"
                      )

        st.markdown("---")

        st.subheader("Taguchi Result Summary")
        st.dataframe(df_tgch_summary)

        tools.download_file(name_label="Input Result File Name",
                      button_label='Download Taguchi result as CSV',
                      file=df_tgch_summary,
                      file_type="csv",
                      gui_key="tgch_summary_data"
                      )

        st.markdown("---")

        st.subheader("Taguchi Factors vs. Mean")
        st.plotly_chart(fig_mean, use_container_width=True)

        tools.download_file(name_label="Input Figure File Name",
                      button_label='Download Taguchi Mean figure as HTML',
                      file=fig_mean,
                      file_type="html",
                      gui_key="tgch_mean_figure"
                      )
        
        st.markdown("---")

        st.subheader("Taguchi Factors vs. SN")
        st.plotly_chart(fig_sn, use_container_width=True)

        tools.download_file(name_label="Input Figure File Name",
                      button_label='Download Taguchi Mean figure as HTML',
                      file=fig_sn,
                      file_type="html",
                      gui_key="tgch_sn_figure"
                      )
        
        st.markdown("---")





        df_sn_max, max_sn, max_sn_mean, df_result_all = backend_tgch_ana(df_fac_summary, factor_list, sn_average, mean_average)

        st.write("Max SN is: ", str(round(max_sn, 2)))
        st.write("Relative Mean is: ", str(round(max_sn_mean,2)))
        # max_sn_mean
        st.dataframe(df_sn_max)


        upper_col, lower_col = st.columns(2)
        with upper_col:

            upper_mean = st.number_input('Uppder Mean Value', value=0.33) 
            
        with lower_col:
            lower_mean = st.number_input('Lower Mean Value', value=0.3) 


        df_result_filter = df_result_all[(df_result_all["Mean"]>lower_mean) & (df_result_all["Mean"]<upper_mean)]
        # print(df_result_all)
        df_result_filter = df_result_filter.sort_values(by=["SN"], ascending=False)
 
        st.dataframe(df_result_filter)

        # st.markdown("---")

        tools.download_file(name_label="Input Result File Name",
                      button_label='Download Taguchi filter result as CSV',
                      file=df_result_filter,
                      file_type="csv",
                      gui_key="filter_result_data"
                      )

        st.markdown("---")




    if ana_type == "2 LV Classification":

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

        # st.header("Under Construction")
        df_x = df_reg[factor]
        df_y = df_reg[response]


        log_model = sm.Logit(df_y, df_x).fit()
            

        log_reg_sum = log_model.summary()

        st.subheader("Model Summary")
        log_reg_sum

        y_pred = log_model.predict(df_x)

        gui_key_main = {"threshold": "classify_roc_main", "threshold_check": "key_in_main"}

        df_roc_data, fig_roc = tools.clf_score(df_y, y_pred, gui_key= gui_key_main)

        tools.reg_save(df_roc_data, fig_roc, log_model)


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

            y_hat = log_model.predict(df_pred_x)
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

                gui_key_predict = {"threshold": "classify_roc_pred", "threshold_check": "key_in_pred"}

                # clf_score_log(df_y, y_hat, gui_key= gui_key_predict)
                tools.clf_score(df_y, y_pred, gui_key=gui_key_predict)




#%% Web App 頁面
if __name__ == '__main__':
    main()

# %%
