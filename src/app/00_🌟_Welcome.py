import streamlit as st

st.set_page_config(
    page_title="Welcome",
)

st.write("# Welcome to my thesis app! 👋")


st.markdown(
    """
    Navigate through menu on the left to explore more:
    - **Data Exploration** - see on which data the model was trained.
    - **Explore Models** - learn more about used models and how they work. 
    - **Get Recommendations** - and finally, try out the models! Get some music recommendations for yourself. 
"""
)
