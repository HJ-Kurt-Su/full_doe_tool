import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import pyDOE2
import numpy as np
import io



@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

# @st.cache_data
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv().encode('utf-8')


def lv_doe_table(df, df_fac):
    # Target: change doe code table to factor real upper & lower level 
    # Input: doe code table with dataframe type
    # 
    df.columns = df_fac.columns
    df_resp = df.copy()
    # df_resp.columns = df_raw.columns

    for i in df.columns:

      resp_sfac_max = df[i].max()
      resp_sfac_min = df[i].min()

      low_lv = df_fac[i][0]
      hi_lv = df_fac[i][1]

      doe_code = [resp_sfac_min, resp_sfac_max]
      true_range = df_fac[i]

      x_range = df[i].sort_values(ascending=True)
      x_range = x_range.unique()[1:-1]
      for j in x_range:

        tmp_lv = np.interp(j, doe_code, true_range)
        df_resp[i].replace({j:tmp_lv}, inplace=True)
        
      df_resp[i].replace({resp_sfac_min:low_lv, resp_sfac_max:hi_lv}, inplace=True)

    return df_resp


def backend(doe_type, fac_n, df_fac, para):

    # Generate DOE table
        
    if doe_type == "response surface":

      doe_array = pyDOE2.ccdesign(fac_n)
      df_code = pd.DataFrame(doe_array)


    elif doe_type == "2 lv full":

      doe_array = pyDOE2.ff2n(fac_n)
      df_code = pd.DataFrame(doe_array)


    elif doe_type == "taguchi":
      taguchi_dict = para["dict"]
      taguchi_array = para["array"]
      taguchi_url = para["url"]

      table_location = taguchi_dict[taguchi_array]
      start_col = 2
      df_tmp = pd.read_html(taguchi_url)[table_location]
      df_code = df_tmp.iloc[1:,start_col:]
      # print(df)
      if fac_n > df_code.shape[1]:

        st.markdown("### Factor Q'ty is not enough, please select another array!!")

      elif fac_n <= df_code.shape[1]:
        df_code = df_code.iloc[:,:fac_n]

    elif doe_type == "latin-hypercube":
      lhc_criteria = para["criteria"]
      design_samples = para["samples"]


      doe_lhs_array = pyDOE2.lhs(fac_n, samples=int(design_samples), criterion=lhc_criteria)
      df_code = pd.DataFrame(doe_lhs_array)

    elif doe_type == "gsd":

      levels = para["levels"]
      reduction = para["reduction"]
      complementary = para["comp"]
      select_fold_no = para["fold"]

      doe_gsd_arry = pyDOE2.gsd(levels, reduction, n=complementary)
    # print(doe_gsd_arry)
      if complementary == 1:
        df_code = pd.DataFrame(doe_gsd_arry)
      else:
        df_code = pd.DataFrame(doe_gsd_arry[select_fold_no])


    # Turn DOE code table to mapping real factor upper & lower limit
    df_resp = lv_doe_table(df_code, df_fac)
    
    # Plot figure
    fig_width = para["width"]
    fig_height = para["height"]

    color_sequence = ["#65BFA1", "#A4D6C1", "#D5EBE1", "#EBF5EC", "#00A0DF", "#81CDE4", "#BFD9E2"]
    color_sequence = px.colors.qualitative.Pastel
    template = "simple_white"
    # df = pd.read_csv("idc_nb_tidy.csv", encoding="utf-8")  
    fig_pair = px.scatter_matrix(df_resp, dimensions=df_resp.columns, 
                                color_discrete_sequence=color_sequence,
                            width=fig_width, height=fig_height)
    
    fig_pair.update_traces(diagonal_visible=False, showupperhalf=False,)

    return df_resp, fig_pair


def main():
    st.title('DoE (Design of Experiment) Tool')

    st.markdown("               ")
    st.markdown("               ")


    uploaded_csv = st.sidebar.file_uploader('#### 選擇您要上傳的CSV檔')

    if uploaded_csv is not None:
        df_fac = pd.read_csv(uploaded_csv, encoding="utf-8")
        st.header('您所上傳的CSV檔內容：')

    else:
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRt_gecFPJZUw3MmI16zI042Z_RaeWj7w-Fs07wLaj-qKymK_K_XwRx-G09IcrqHmHhUQqP8-Qe0cQQ/pub?gid=0&single=true&output=csv"
        st.header('未上傳檔案，以下為 Demo：')
        df_fac = pd.read_csv(url, encoding="utf-8")
    
    st.dataframe(df_fac)

    doe_type = st.selectbox(
    'Choose DoE Type:',
    ("2 lv full", "response surface", "taguchi", "gsd", "latin-hypercube"))
    para = {}

  # Define doe type & show relative parameter based on DOE type
    if doe_type == "taguchi":
      taguchi_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQMX31nAb2tY_U33hGK-Y3Up-ta7sl55grC591QmS--kqz9EhQyCvxYMe9_fG3YnPIoZuiPlkMQ-zg_/pubhtml"
      taguchi_dict = {"L4_3F2L":0, "L8_7F2L": 1, "L9_4F3L": 2, "L12_11F2L": 3, "L16_15F2L": 4, "L16b_5F4L": 5, "L18_1F2L-7F3L": 6}
      st.markdown("#### Taguchi Array [Link](%s)" % taguchi_url)
      
      taguchi_array = st.selectbox(
        'Choose Taguchi Array:',
        ("L4_3F2L", "L8_7F2L", "L9_4F3L", "L12_11F2L", "L16_15F2L", "L16b_5F4L", "L18_1F2L-7F3L"))
      
      para = {"url": taguchi_url, "dict": taguchi_dict, "array": taguchi_array}

    elif doe_type == "latin-hypercube":
      lhc_criteria = st.selectbox(
        'Choose Sample Method:',
        (None, "center", "maximin", "centermaximin", "correlation"))

      design_samples = st.number_input('Insert a number', min_value=1, value=10)

      para = {"criteria": lhc_criteria, "samples": design_samples}

    elif doe_type == "gsd":
      gsd_url = "https://doi.org/10.1021/acs.analchem.7b00506"
      st.markdown("#### GSD Introduction [Link](%s)" % gsd_url)

      #@markdown Number of factor levels per factor in design. Must match factor q'ty  

      #@markdown [3, 4] means 2 factor. 1st factor is 3 levels, 2nd factor is 4 levels.

      # levels =  #@param {type:"raw"}

      st.markdown("###### Define Level for Each Factor:")
      levels=list()
      a=0
      for i in df_fac.columns:
        tmp_lv = st.number_input(i, min_value=2, value=3, step=1, key=a)
        levels.append(tmp_lv)
        a+=1
      # levels = [3, 4, 2, 3]


      st.markdown("--------------------------")
      st.markdown("###### Define GSD Rest Parameter:")
      #@markdown Reduce the number of experiments to approximately half (Default 2).
      reduction = st.number_input('Reduction', min_value=2, value=4, max_value=8, step=1)   
      #@markdown Fold like to divide doe array (Default 1)
      complementary = st.number_input('Complementary', min_value=1, max_value=4, step=1)
      # complementary = 1 #@param {type:"slider", min:1, max:4, step:1}
      #@markdown Choose which fold matrix (must < **complementary**)
      select_fold_no = st.number_input('Select fold number', min_value=0, max_value=complementary-1, step=1)
      
      para = {"levels": levels, "reduction": reduction, "comp": complementary, "fold": select_fold_no}

    st.write('You selected:', doe_type)

    size_col1, size_col2 = st.columns(2)
    with size_col1:

        fig_width = st.number_input('Figure Width', min_value=640, value=1280, max_value=5120, step=320) 
        
    with size_col2:
        fig_height = st.number_input('Figure Height', min_value=480, value=960, max_value=3840, step=240) 

    para["width"] = fig_width
    para["height"] = fig_height

    fac_n = df_fac.shape[1]
    df_resp, fig_pair = backend(doe_type, fac_n, df_fac, para)
    st.dataframe(df_resp)
    
    st.plotly_chart(fig_pair, use_container_width=True)

    mybuff = io.StringIO()
    fig_file_name = doe_type + "_pair-plot-test.html"
    # fig_html = fig_pair.write_html(fig_file_name)
    fig_pair.write_html(mybuff, include_plotlyjs='cdn')
    html_bytes = mybuff.getvalue().encode()
    

    csv = convert_df(df_resp)
    date = str(datetime.datetime.now()).split(" ")[0]
    # table_filename = doe_type + "_table_" + date
    doe_table = doe_type + "_" + date + "_doe-table.csv"
    st.download_button(label='Download DOE table as CSV', 
                      data=csv, 
                      file_name=doe_table,
                      mime='text/csv')

    st.download_button(label="Download figure",
                              data=html_bytes,
                              file_name=fig_file_name,
                              mime='text/html'
                              )
    
#%% Web App 頁面
if __name__ == '__main__':
    main()

# %%