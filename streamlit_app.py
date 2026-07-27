import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="CSV展示事例",
    layout="wide"
)

st.title("示例一")
st.markdown("读取项目下的data/目录下的CSV，用表格和简单筛选展示")

st.sidebar.header("数据源")
csv_choice = st.sidebar.selectbox(
    "选择CSV文件",
    options=["users.csv", "videos.csv", "interactions_cleaned.csv"],
    index=0
)

data_path = Path("data") / csv_choice

@st.cache_data #保存读取的缓存结果，提升效率，比如vsc就不会重复去读取了
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_csv(str(data_path))

#创建多列layout
c1, c2, c3 = st.columns(3)

c1.metric("行数：", f"{len(df)}")
c2.metric("列数：", f"{df.shape[1]}")
c3.metric("文件：", csv_choice)


st.subheader("数据预览")
st.slider(
    "最多显示多少行",
    min_value=5,
    max_value=200,
    value=20,
    step=5
)

st.dataframe(df.head(max_rows))

st.subheader("数据可视化示例")
col_img1, col_img2 = st.columns(2)
with col_img1:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(df.select_dtypes(include=["number"]).iloc[:,0] if len(df.select_dtypes(includes=["number"]).columns)>0 else [1,2,3])
    ax.set_title("数据分布")
    st.pyplot(fig)

with col_img2:
    image_file = Path("data") / "example.png"
    st.image(str(image_file))

    