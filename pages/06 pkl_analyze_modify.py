import pickle

import pandas as pd
import streamlit as st

import tools


def show_item(package):
    """Display one entry of a pickled dict, specialised by its key prefix (model_/df_/fig_/img_)."""
    key = st.selectbox("請選擇要顯示的項目", list(package))
    item = package[key]
    kind = key.split("_")[0]
    if kind == "fig":
        st.plotly_chart(item, width="stretch")
        tools.download_file("Input Result File Name", "Download result as HTML", item, "html", "figure")
    elif kind == "img":
        st.image(item)
    else:
        if kind not in ("model", "df"):
            st.markdown("所選項目尚未建立特化項目！有可能無法正確展示")
        st.write(item)
        if kind == "df":
            tools.download_file("Input Result File Name", "Download result as CSV", pd.DataFrame(item), "csv", "df_data")


def main():
    st.title("Pickle Analyze & Modify")

    uploaded = st.sidebar.file_uploader("請上傳您的 Model 檔案", type=["pickle", "pkl", "zip"], key="model_file")
    if uploaded is None:
        st.warning("請上傳一個有效的 pickle 檔案！")
        return

    st.header("您所上傳的 Model 檔內容：")
    st.write(uploaded.name)

    # keep the (possibly modified) package across reruns until another file is uploaded
    file_id = (uploaded.name, uploaded.size)
    if st.session_state.get("pkl_src") != file_id:
        st.session_state.pkl_src = file_id
        st.session_state.pkl_package = pickle.load(uploaded)
    package = st.session_state.pkl_package

    try:
        if isinstance(package, dict):
            show_item(package)
        else:
            st.info(f"Pickle 檔案內容型態為: {type(package)}")
            st.write(package)
    except Exception as e:
        st.error(f"Error displaying content: {e}")

    if not isinstance(package, dict):
        return

    st.divider()
    st.header("新增 Pickle 檔內容物：")
    new_file = st.file_uploader("Upload a file to add (e.g., .jpg)", type=["jpg", "png", "jpeg", "txt", "csv", "xlsx"],
                                key="new_file")
    new_key = st.text_input("Enter a key name for the new content", value="new_content")
    if st.button("Add Content to Pickle") and new_file and new_key:
        package[new_key] = new_file.getvalue()
        st.success(f"Content '{new_key}' added successfully!")

    tools.download_file("Input New Model File Name", "Download new pickle file", package, "pickle", "new_package")


main()
