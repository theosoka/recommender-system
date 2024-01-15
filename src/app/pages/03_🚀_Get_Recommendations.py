import streamlit as st

models = ("dtc", "gnb", "mnb", "pn")

option = st.selectbox(
    "Which model would you like to use for computing recommendations?",
    models,
    index=None,
    placeholder="Select a model..",
)
