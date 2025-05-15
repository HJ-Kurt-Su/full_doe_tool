import pandas as pd 
import streamlit as st
import requests
from statsmodels.iolib.smpickle import load_pickle
import pickle
# import datetime
from io import BytesIO
import tools

# def add_content_to_pickle(package, new_file, new_key):
#     # 讀取現有的 pickle 檔案
#     # package = pickle.load(uploaded_model)

#     # 將新內容新增到 pickle
#     package[new_key] = new_file.getvalue()

#     # 儲存更新後的 pickle 檔案
#     updated_pickle = BytesIO()
#     pickle.dump(package, updated_pickle)
#     updated_pickle.seek(0)

#     return updated_pickle

def backend(uploaded_model):
    package = load_pickle(uploaded_model)
    pkl_key = package.keys()
    show_item = st.selectbox("請選擇要顯示的項目", pkl_key)
    file_type = show_item.split("_")[0]
    if file_type == "model":
        st.write(package[show_item])
    elif file_type == "df":
        st.write(package[show_item])
        tools.download_file(
            name_label="Input Result File Name",
            button_label='Download result as CSV',
            file=pd.DataFrame(package[show_item]),
            file_type="csv",
            gui_key="df_data"
            )
    elif file_type == "fig":
        st.plotly_chart(package[show_item], use_container_width=True)
        tools.download_file(
            name_label="Input Result File Name",
            button_label='Download result as HTML',
            file=package[show_item],
            file_type="html",
            gui_key="figure"
            )
    elif file_type == "img":
        st.image(package[show_item])

    else:
    
        st.markdown("所選項目尚未建立特化項目！有可能無法正確展示")
        st.write(package[show_item])
        # st.image(package[show_item])
    
    return package
    # st.write(package["model"])


 
def bknd_add(package, new_file, new_key, st_add):
    # package = load_pickle(uploaded_model)
    if "add_triggered" not in st.session_state:
        st.session_state.add_triggered = False
    if new_file and new_key:
        
        if st_add:
            st.session_state.add_triggered = True
        
        if st.session_state.add_triggered:
            package[new_key] = new_file.getvalue()
            # updated_pickle = add_content_to_pickle(uploaded_model, new_file, new_key)
            st.success(f"Content '{new_key}' added successfully!")

            # 提供下載更新後的 pickle 檔案
            updated_pickle = BytesIO()
            pickle.dump(package, updated_pickle)
            updated_pickle.seek(0)
            # new_package = pickle.dumps(package)
            # 提供下載更新後的 pickle 檔案
            # 重置按鈕狀態
            st.session_state.add_triggered = False
    return package


def main():
    st.title('Predict')

    # st.markdown("#### Author & License:")

    # st.markdown("**Kurt Su** (phononobserver@gmail.com)")

    # st.markdown("**This tool release under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license**")

    st.markdown("               ")
    st.markdown("               ")


    uploaded_model = st.sidebar.file_uploader("請上傳您的 Model 檔案", type=["pickle"])
    # uploaded_model
    # uploaded_raw = st.sidebar.file_uploader("請上傳您的 CSV 檔案", type=["csv", "xlsx"])

 

    if uploaded_model is not None:
        st.header('您所上傳的 Model 檔內容：')
        uploaded_model.name

        package = backend(uploaded_model)

        st.markdown("---")
        st.header('新增 Pickle 檔內容物：')

        new_file = st.file_uploader("Upload a file to add (e.g., .jpg)", type=["jpg", "png", "jpeg", "txt", "csv", "xlsx"], key="new_file")
        new_key = st.text_input("Enter a key name for the new content", value="new_content")
        st_add = st.button("Add Content to Pickle")

        package_new = bknd_add(package, new_file, new_key, st_add)

        tools.download_file(name_label="Input New Model File Name",
                    button_label='Download new pickle file',
                    file=package_new,
                    file_type="pickle",
                    gui_key="new_package"
                    )

    else:
        st.warning("請上傳一個有效的 pickle 檔案！")
        # url_model = "https://raw.githubusercontent.com/HJ-Kurt-Su/full_doe_tool/main/2023-07-01_model.pickle"
        # st.header('未上傳 Model，以下為 Demo：')
        # uploaded_model = BytesIO(requests.get(url_model).content)
        # # r

    st.markdown("---")
 
    # st.checkbox("Add Content", value=False, key="add_cont")
    # if st.session_state.add_cont:
    #     st.markdown("#### Please add content name:")
    #     st.text_input("Add Content Name", value="")
    #     st.file_uploader("請上傳您的檔案", key="add_cont_file")
    #     st.button("Add Content", key="add_cont_btn")
    #     if st.session_state.add_cont_btn:
    #         st.session_state.add_cont = False
    #         st.session_state.add_cont_btn = False
    #         st.session_state.add_cont_file = None
    #         st.session_state.add_cont_name = None
    #         st.success("Add Content Success")




    st.markdown("---")





if __name__ == '__main__':
    main()

