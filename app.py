import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="F1 Pit Wall", layout="wide")

st.title("Hello Pit Wall")

# Demo data
df = pd.DataFrame(
    {
        "driver": ["NOR", "VER", "LEC", "HAM", "ALO"],
        "lap": [1, 2, 3, 4, 5],
        "lap_time_s": [92.3, 92.5, 92.7, 92.6, 92.8],
    }
)

st.subheader("Lap Times Table")
st.dataframe(df, use_container_width=True)

st.subheader("Lap Time Bar Chart")
fig = px.bar(df, x="driver", y="lap_time_s", text="lap_time_s")
fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
st.plotly_chart(fig, use_container_width=True)
