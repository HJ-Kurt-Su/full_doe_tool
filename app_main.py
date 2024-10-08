import streamlit as st 

def main():
    st.title("Author & License:")
    st.markdown("**Kurt Su** (phononobserver@gmail.com)")
    st.markdown("**This tool release under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license**")

    st.markdown("               ")
    st.markdown("               ")

    st.header("Page Purpose & Description")
    st.markdown("**doe**: Design Of Experiment tool with different method")
    st.markdown("**eda**: Exploratory Data Analysis tool with different figure type")
    st.markdown("**regression A Trad**: Traditional statistics regression tool with linear, logistic & taguchi method")
    st.markdown("**regression B ML**: Machine Learning regression/classification tool")
    st.markdown("**predict**: Predict result with load trained model")
    st.markdown("**predict performance**: Predict result accracy index with real result")

if __name__ == '__main__':
    main()
