import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

import tools

DEMO = {
    "Regression Method": (tools.DEMO_REG, "Y1", ["R", "r", "t"]),
    "Taguchi Method": (tools.DEMO_TAGUCHI, ["Z1", "Z2", "Z3"], ["R", "r", "t"]),
    "2 LV Classification": (tools.DEMO_CLF, "Y2", ["R", "r", "t"]),
}


# ---------------------------------------------------------------- OLS regression
def ols_model_figure(df_reg, formula):
    res = smf.ols(formula, df_reg).fit()
    df_result = df_reg.copy()
    df_result["yhat"] = res.fittedvalues
    df_result["resid"] = res.resid

    t_crit = stats.t.ppf(1 - 0.05 / 2, res.df_resid)
    t_vals = res.tvalues.drop("Intercept", errors="ignore").abs()
    df_pareto = pd.DataFrame({"factor": t_vals.index, "value": t_vals.to_numpy()})
    fig = tools.model_check_figure(df_result, df_pareto, t_crit, bar_title="pareto-plot (red line as criteria)")
    sw_p = stats.shapiro(df_result["resid"]).pvalue
    return res, df_result, fig, sw_p


def regression_page(df_reg, demo_cols):
    cols = list(df_reg.columns)
    response = tools.pick("### Choose Response(y)", cols, demo_cols[1] if demo_cols else None)
    others = [c for c in cols if c != response]
    factor = tools.pick("### Choose Factor(x)", others, demo_cols[2] if demo_cols else None, multi=True)
    if not factor:
        st.error("Please select at least one factor.")
        st.stop()

    factor_2od = st.multiselect("### Choose Factor 2nd Order(x^2)", [f"I({f} ** 2)" for f in factor])
    factor_inter = st.multiselect("### Choose Factor Interaction(x1 * x2)",
                                  ["*".join(p) for p in itertools.combinations(factor, 2)])

    features = factor + factor_2od + factor_inter
    formula = f"{response}~{'+'.join(features)}"
    st.subheader(formula)

    result, df_result, fig, sw_p = ols_model_figure(df_reg, formula)
    st.write(result.summary())
    st.markdown(f"#### Normality Test P Value:{sw_p:.4f} ")
    st.markdown("##### Residual is normal distribution" if sw_p >= 0.05 else "##### Residual is NOT normal distribution!!")
    st.plotly_chart(fig, width="stretch")

    tools.reg_save(df_result, fig, {
        "model": result,
        "features": features,
        "df_nom": pd.DataFrame(columns=factor),
        "df_result": df_result,
        "reg_summary": result.summary(),
    })
    tools.predict_section(result.predict, lambda y, y_hat: tools.show_perf(y, y_hat, key="reg"), key="reg")


# ---------------------------------------------------------------- Taguchi
def taguchi_summary(df, factors, responses, target):
    long = df.melt(id_vars=factors, value_vars=responses, var_name="Response", value_name="Y")
    n_resp = len(responses)
    if target == "nominal best":
        stat, agg = "Std", ("Y", "std")
    elif target == "smaller best":
        stat, agg = "Sum_Square", ("Sq", "sum")
        long["Sq"] = long["Y"].pow(2)
    else:
        stat, agg = "Sum_Square_Inv", ("Sq", "sum")
        long["Sq"] = 1 / long["Y"].pow(2)

    summary = long.groupby(factors, as_index=False).agg(Mean=("Y", "mean"), **{stat: agg})
    if target == "nominal best":
        ratio = summary[stat].pow(2) / summary["Mean"].pow(2)
    else:
        ratio = summary[stat] / n_resp
    summary["SN"] = -10 * np.log10(ratio)

    per_factor = pd.concat(
        [summary.groupby(f, as_index=False)[["Mean", "SN"]].mean() for f in factors], ignore_index=True
    )[factors + ["Mean", "SN"]]
    return per_factor, summary


def taguchi_figures(df_fac, factors):
    figs = {}
    for metric in ("Mean", "SN"):
        fig = make_subplots(rows=1, cols=len(factors), subplot_titles=factors)
        for i, f in enumerate(factors, start=1):
            d = df_fac[df_fac[f].notna()]
            fig.add_trace(go.Scatter(x=d[f], y=d[metric]), row=1, col=i)
        fig.update_layout(height=960, width=1280, title_text=metric)
        figs[metric] = fig
    return figs


def taguchi_best(df_fac, factors, sn_avg, mean_avg):
    """Best level per factor, plus the additive prediction of every level combination."""
    k = len(factors)
    tables = [df_fac[df_fac[f].notna()].reset_index(drop=True) for f in factors]
    df_best = pd.DataFrame([t.loc[t["SN"].idxmax()] for t in tables]).reset_index(drop=True)
    max_sn = df_best["SN"].sum() - (k - 1) * sn_avg
    max_mean = df_best["Mean"].sum() - (k - 1) * mean_avg

    idx = [g.ravel() for g in np.meshgrid(*[np.arange(len(t)) for t in tables], indexing="ij")]
    df_all = pd.DataFrame({f: t[f].to_numpy()[i] for f, t, i in zip(factors, tables, idx)})
    df_all["Mean"] = sum(t["Mean"].to_numpy()[i] for t, i in zip(tables, idx)) - (k - 1) * mean_avg
    df_all["SN"] = sum(t["SN"].to_numpy()[i] for t, i in zip(tables, idx)) - (k - 1) * sn_avg
    return df_best, max_sn, max_mean, df_all


def taguchi_page(df_reg, demo_cols):
    cols = list(df_reg.columns)
    responses = tools.pick("### Choose Response(Yn)", cols, demo_cols[1] if demo_cols else None, multi=True)
    factors = tools.pick("### Choose Factors(Xn)", [c for c in cols if c not in responses],
                   demo_cols[2] if demo_cols else None, multi=True)
    target = st.selectbox("### Choose Traget", ["larger best", "nominal best", "smaller best"])
    if not responses or not factors:
        st.error("Please select at least one response and one factor.")
        st.stop()

    df_fac, df_summary = taguchi_summary(df_reg, factors, responses, target)
    figs = taguchi_figures(df_fac, factors)

    st.subheader("Taguchi Factor Lv vs. Mean & SN")
    st.dataframe(df_fac)
    tools.download_file("Input Result File Name", "Download Taguchi result as CSV", df_fac, "csv", "tgch_factor_data")
    st.divider()

    st.subheader("Taguchi Result Summary")
    st.dataframe(df_summary)
    tools.download_file("Input Result File Name", "Download Taguchi result as CSV", df_summary, "csv", "tgch_summary_data")
    st.divider()

    for metric, title in (("Mean", "Taguchi Factors vs. Mean"), ("SN", "Taguchi Factors vs. SN")):
        st.subheader(title)
        st.plotly_chart(figs[metric], width="stretch")
        tools.download_file("Input Figure File Name", "Download Taguchi Mean figure as HTML", figs[metric], "html",
                            f"tgch_{metric.lower()}_figure")
        st.divider()

    df_best, max_sn, max_mean, df_all = taguchi_best(df_fac, factors, df_summary["SN"].mean(), df_summary["Mean"].mean())
    st.write("Max SN is: ", str(round(max_sn, 2)))
    st.write("Relative Mean is: ", str(round(max_mean, 2)))
    st.dataframe(df_best)

    lower_col, upper_col = st.columns(2)
    with upper_col:
        upper = st.number_input("Uppder Mean Value", value=0.33)
    with lower_col:
        lower = st.number_input("Lower Mean Value", value=0.3)

    df_filter = df_all[(df_all["Mean"] > lower) & (df_all["Mean"] < upper)].sort_values("SN", ascending=False)
    st.dataframe(df_filter)
    tools.download_file("Input Result File Name", "Download Taguchi filter result as CSV", df_filter, "csv",
                        "filter_result_data")
    st.divider()


# ---------------------------------------------------------------- Logistic
def logit_page(df_reg, demo_cols):
    cols = list(df_reg.columns)
    response = tools.pick("### Choose Response(y)", cols, demo_cols[1] if demo_cols else None)
    factor = tools.pick("### Choose Factor(x)", [c for c in cols if c != response],
                  demo_cols[2] if demo_cols else None, multi=True)
    if not factor:
        st.error("Please select at least one factor.")
        st.stop()

    df_x, df_y = df_reg[factor], df_reg[response]
    model = sm.Logit(df_y, df_x).fit(disp=False)
    st.subheader("Model Summary")
    st.write(model.summary())

    y_pred = model.predict(df_x)
    df_roc, fig_roc = tools.clf_score(df_y, y_pred, key="main")

    df_result = df_reg.copy()
    df_result["yhat"] = y_pred
    tools.reg_save(df_roc, fig_roc, {
        "model": model,
        "features": factor,
        "df_nom": pd.DataFrame(columns=factor),
        "df_result": df_result,
    })
    tools.predict_section(lambda d: model.predict(d[factor]),
                          lambda y, y_hat: tools.clf_score(y, y_hat, key="pred"), key="clf")


def main():
    st.title("Linear Regression Tool")

    uploaded = st.sidebar.file_uploader(tools.UPLOAD_LABEL, type=tools.TABLE_TYPES)
    ana_type = st.selectbox("Choose Analysis Mehthd", list(DEMO))
    demo_url = DEMO[ana_type][0]

    df_raw = tools.load_table(uploaded, demo_url)
    st.dataframe(df_raw)
    df_reg = tools.filter_data(df_raw)

    demo_cols = DEMO[ana_type] if uploaded is None else None
    {"Regression Method": regression_page, "Taguchi Method": taguchi_page, "2 LV Classification": logit_page}[
        ana_type
    ](df_reg, demo_cols)


main()
