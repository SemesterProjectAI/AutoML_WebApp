import streamlit as st
import pandas as pd
import os

# Importing Profiling stuff
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

# Importing AutoML stuff
from pycaret.classification import setup, compare_models, pull, save_model


# Creating SideBar
with st.sidebar:
    st.title("AutoML Project")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])


# Getting the source file
if os.path.exists("Datasets/SourceData.csv"):
    df = pd.read_csv("Datasets/SourceData.csv")


# On Upload Selection
if choice == "Upload":
    st.title("Upload your Data for modeling")
    file = st.file_uploader("Upload your dataset", type=["csv"])
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("Datasets/SourceData.csv", index=False)
        st.dataframe(df)

# On Profiling Selection
if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

# On ML Selection
if choice == "ML":
    st.title("Auto ML Precessing")
    target = st.selectbox("Select your target", df.columns)

    if st.button("Train models"):
        setup(df, target=target)

        st.info("This is the ML Experiment Settings")
        setup_df = pull()
        st.dataframe(setup_df)

        best_model = compare_models()

        st.info("This is the ML models performance")
        compare_df = pull()
        st.dataframe(compare_df)

        st.info("Best Model settings")
        best_model
        save_model(best_model,"Models/best_model")


# On Download Selection
if choice == "Download":
    st.title("Best model trained, Download it")
    with open("Models/best_model.pkl", "rb") as f:
        st.download_button('Download Model', f, 'Trained_model.pkl')
