import streamlit as st

st.set_page_config(
    page_title="Welcome",
)

st.write("# :green[Unveiling Music Recommender Systems Approaches]")
st.divider()


st.markdown(
    """
    Welcome to the web app dedicated to my bachelor thesis!
    My goal was to explore *Music Recommender Systems*, approaches to them, challenges, 
    mathematical aspects and the most important: create my own recommender system.
    
    Navigate through menu on the left to explore more:
    - **Data Exploration** - see on which data the model was trained.
    - **Explore Models** - learn more about most successful models and how they work. 
    - **Get Recommendations** - and finally, try out the models! Get some music recommendations for yourself. 
"""
)
