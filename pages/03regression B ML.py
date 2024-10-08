import streamlit as st
import pandas as pd
# import itertools

# import datetime
import numpy as np
# import io
# import statsmodels.formula.api as smf
# import statsmodels.api as sm 
# from scipy.stats import shapiro
# from scipy import stats
# from statsmodels.graphics.gofplots import qqplot


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


# from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
import graphviz
from sklearn.tree import export_graphviz

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# import pickle
import tools





# color_sequence = ["#65BFA1", "#A4D6C1", "#D5EBE1", "#EBF5EC", "#00A0DF", "#81CDE4", "#BFD9E2"]
color_sequence = px.colors.qualitative.Pastel
template = "simple_white"
  
def backend(df_reg, formula, fig_size):
    # df_reg = df_raw.copy()
    st.markdown("Under Construction")

    # result, df_result, model = ols_reg(formula, df_reg)

    # alpha = 0.05
    # # f_num = len(result.tvalues)-1
    # # dof = round(f_num/3, 0)
    # dof = result.df_resid
    # t_val = stats.t.ppf(1-alpha/2, dof)

    # df_pareto = result.tvalues[1:].abs()
    # df_pareto = df_pareto.sort_values(ascending=True)
    # df_pareto = pd.DataFrame(df_pareto).reset_index(level=0)
    # df_pareto.columns = ["factor", "t-value"]


    # SW, sw_p_val = shapiro(df_result["resid"])
    # df_qq = acquire_qq_data(df_result["resid"])



    # fig = make_subplots(
    #     rows=2, cols=2,
    #     subplot_titles=("yhat-residual-plot (random better)", "residual-histogram-plot (normal distribution better)", 
    #                     "redidual-sequence-plot (random better)", "pareto-plot (red line as criteria)"))

    # fig.add_trace(go.Scatter(x=df_result["yhat"], y=df_result["resid"], mode="markers", 
    #                         marker=dict(color='rgba(19, 166, 255, 0.6)')),
    #             row=1, col=1)

    # fig.add_trace(go.Histogram(x=df_result["resid"],
    #                         marker=dict(color='rgba(19, 166, 255, 0.6)')),
    #             row=1, col=2)

    # fig.add_trace(go.Scatter(y=df_result["resid"], mode="lines+markers",
    #                         marker=dict(color='rgba(19, 166, 255, 0.6)')),
    #             row=2, col=1)

    # fig.add_trace(go.Bar(x=df_pareto["t-value"], y=df_pareto["factor"], orientation='h', width=0.8,
    #                     marker=dict(color='rgba(19, 166, 255, 0.6)')
    #                     ),
    #             row=2, col=2)
    # fig.add_vline(x=t_val, line_width=2, line_dash='dash', line_color='red',
    #             row=2, col=2)

    # # fig.add_trace(go.Scatter(x=df_qq["x_line"], y=df_qq["y_line"], mode="lines"),
    # #               row=2, col=2)

    # fig.update_xaxes(title_text="Y-hat", row=1, col=1)
    # fig.update_yaxes(title_text="Residual", row=1, col=1)

    # fig.update_xaxes(title_text="Residual", row=1, col=2)
    # fig.update_yaxes(title_text="Count", row=1, col=2)

    # fig.update_xaxes(title_text="Sequence", row=2, col=1)
    # fig.update_yaxes(title_text="Residual", row=2, col=1)

    # fig.update_xaxes(title_text="Factor Importance", row=2, col=2)
    # fig.update_yaxes(title_text="Factor", row=2, col=2)

    # fig.update_layout(height=fig_size[1], width=fig_size[0],
    #                 title_text="Model Check Figure",
    #                 showlegend=False)
 
    # return result, df_result, fig, sw_p_val

def backend_clf(df_x, df_y, clf_type):


    if clf_type == "Support Vector" or "Decision Tree":
        # clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        nom_choice = st.checkbox("Normalize", value=False)
        if nom_choice == True:
            scaler = StandardScaler()
            x = scaler.fit_transform(df_x)
            st.subheader("Normalize X")
            st.dataframe(x)
        else:
            x = df_x
        
        if clf_type == "Support Vector":
            clf = SVC(gamma='auto', kernel="linear")
        elif clf_type == "Decision Tree":
            max_tree_layer = st.number_input("Setup Max Tree Layer", min_value=1, max_value=10, value=2)
            clf = DecisionTreeClassifier(max_depth=max_tree_layer)
        # clf = make_pipeline(StandardScaler(), DecisionTreeClassifier())
        clf.fit(x, df_y)
        y_pred = clf.predict(x)
        # clf.fit(df_x, df_y)
        # y_pred = clf.predict(df_x)
        # feature_importances = clf.feature_importances_
        if clf_type == "Support Vector":
            st.subheader("Only For Reference")
            feature_importances = clf.coef_
        elif clf_type == "Decision Tree":
            feature_importances = clf.feature_importances_
            # tree_rules = export_text(clf, feature_names=df_x.columns)
            # tree_rules
        st.subheader("Feature Importance")
        feature_importances
        
        # importances = clf_mod.feature_importances_ 
        # importances
        # clf.coef_()

    fpr, tpr, threshold = roc_curve(df_y, y_pred)
    dict_sum = {"FPR": fpr, "TPR": tpr, "Threshold": threshold}
    df_roc_data = pd.DataFrame.from_dict(dict_sum)
    roc_auc = auc(fpr,tpr)
    
    return df_roc_data, roc_auc, clf, y_pred

def main():

    fig_size = [1280, 960]

    st.title('ML Regression Tool')

    st.markdown("               ")
    st.markdown("               ")


    uploaded_raw = st.sidebar.file_uploader('#### 選擇您要上傳的CSV檔', type=["csv", "xlsx"])
    # uploaded_csv = st.file_uploader('#### 選擇您要上傳的CSV檔')

    ana_type = st.selectbox("Choose Analysis Mehthd", ["Regression Method", "2 LV Classification"])

    if ana_type == "Regression Method":
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTnqEkuIqYHm1eDDF-wHHyQ-Jm_cvmJuyBT4otEFt0ZE0A6FEQyg1tqpWTU0PXIFA_jYRX8O-G6SzU8/pub?gid=0&single=true&output=csv"

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

    if ana_type == "Regression Method":

        st.header("Under Construction")

        # # select_list
        # response = st.selectbox("### Choose Response(y)", select_list)
        # # response
        # factor_list = select_list.copy()
        # factor_list.remove(response)
        # factor = st.multiselect(
        #     "### Choose Factor(x)", factor_list)
        # if not factor:
        #     st.error("Please select at least one factor.")


        # # factor_2nd order
        # factor_2od_list = list()
        # for j in factor:
        #     factor_2od_list.append("I("+j+" ** 2)")

        # factor_2od = st.multiselect(
        #     "### Choose Factor 2nd Order(x^2)", factor_2od_list)
        # if not factor_2od:
        #     st.error("Please select at 2nd order factor.")

        # # factor_interaction
        # factor_inter_tmp = list(itertools.combinations(factor, 2))
        # factor_inter_list =  list()
        # for i in factor_inter_tmp:
        #     tmp = "*".join(i)
        #     factor_inter_list.append(tmp)

        # factor_inter = st.multiselect(
        #     "### Choose Factor Interaction(x1 * x2)", factor_inter_list)
        # if not factor_inter:
        #     st.error("Please select interaction factor.")
        # st.write(factor)
        # st.write(factor_inter_list)
        # factor
        # factor_final = factor + factor_inter  

        # if uploaded_raw is None:
        #     st.header('未上傳檔案，以下為 Demo：')
        #     response = "Y1"
        #     factor = ["R", "r", "t"]

        # factor_final = factor + factor_2od + factor_inter

        # scaler = StandardScaler()
        # df_ttmp = df_reg[factor]
        # df_sc = scaler.fit_transform(df_ttmp)
        # df_sc

        # if st.checkbox('Perform Analysis'):
        # x_formula = "+".join(factor_final)
        # formula = response + "~" + x_formula
        # st.subheader(formula)
        # result, df_result, fig, sw_p_val = backend(df_reg, formula, fig_size)
        # st.write(result.summary())
        
        # st.markdown("#### Normality Test P Value:%s " % round(sw_p_val,4))
        # if sw_p_val >= 0.05:
        #     st.markdown("##### Residual is NOT normal distribution!!")
        # else:
        #     st.markdown("##### Residual is normal distribution")    

        # st.plotly_chart(fig, use_container_width=True)


        # st.markdown("---")

        # tools.download_file(name_label="Input Result File Name",
        #               button_label='Download statistics result as CSV',
        #               file=df_result,
        #               file_type="csv",
        #               gui_key="result_data"
        #               )

        # st.markdown("---")

        # tools.download_file(name_label="Input Figure File Name",
        #               button_label='Download figure as HTML',
        #               file=fig,
        #               file_type="html",
        #               gui_key="figure" 
        #               )
        
        # st.markdown("---")

        # tools.download_file(name_label="Input Model File Name",
        #               button_label='Download model as PICKLE',
        #               file=result,
        #               file_type="pickle",
        #               gui_key="model"
        #               )
        
        # st.markdown("---")



    if ana_type == "2 LV Classification":

        clf_type = st.selectbox("### Choose Classification Method", ["Support Vector", "Decision Tree"])

        df_x = df_reg[factor]
        df_y = df_reg[response]
        
        df_roc_data, roc_auc, clf, y_pred = backend_clf(df_x, df_y, clf_type)

        st.markdown("### AUC is: %s" % round(roc_auc,4))


        st.subheader("ROC Data")
        df_roc_data
        
        fig_size=640

        fig_roc = px.line(df_roc_data, x="FPR", y="TPR", markers=False, labels=roc_auc, 
                          range_x=[0,1], range_y=[0,1.1], width=fig_size, height=fig_size)
        
        st.subheader("ROC Figure")
        st.plotly_chart(fig_roc, use_container_width=True)


        if clf_type == "Decision Tree":

            g = export_graphviz(clf,
                                feature_names=df_x.columns,
                                class_names=["0","1"],
                                filled=True)

            st.subheader("Decision Tree")
            st.graphviz_chart(g)

            # st.plotly_chart(g, use_container_width=True)

        if clf_type == "Support Vector" or "Decision Tree":
            y_pred = pd.Series(y_pred)
            threshold_cut = 1


        y_pred_code = y_pred.map(lambda x: 1 if x >= threshold_cut else 0)

        acc = accuracy_score(df_y, y_pred_code)
        cof_mx = confusion_matrix(df_y, y_pred_code)
        risk_qty = cof_mx[1, 0]
        risk =  risk_qty/y_pred_code.size
  
        st.markdown("### Accuracy is: %s" % round(acc,4))
        st.markdown("### Risk is: %s" % round(risk,4))
        st.markdown("### Confusion Matrix:")
        st.dataframe(cof_mx)

        st.markdown("---")

        tools.download_file(name_label="Input Result File Name",
                      button_label='Download regression result as CSV',
                      file=df_roc_data,
                      file_type="csv",
                      gui_key="result_data"
                      )

        st.markdown("---")

        tools.download_file(name_label="Input Figure File Name",
                      button_label='Download figure as HTML',
                      file=fig_roc,
                      file_type="html",
                      gui_key="figure"
                      )
        



#%% Web App 頁面
if __name__ == '__main__':
    main()

# %%
