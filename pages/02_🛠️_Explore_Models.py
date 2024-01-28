import streamlit as st

st.set_page_config(page_title="Explore Models", page_icon="📊")

model_names = [
    "Item-Item Collaborative Filtering",
    "KNN",
    "Slope One",
    "Gradient Boosting Classifier",
    "SVD",
]

(cf, knn, slope, gbc, svd) = st.tabs(model_names)

with cf:
    st.markdown(
        """In machine learning, the **perceptron** (or **McCulloch–Pitts neuron**) is an algorithm for supervised learning of 
        binary classifiers. A binary classifier is a function which can decide whether or not an input, represented 
        by a vector of numbers, belongs to some specific class. It is a type of linear classifier, 
        i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a 
        set of weights with the feature vector."""
    )

    st.image("https://upload.wikimedia.org/wikipedia/commons/8/8c/Perceptron_moj.png")

    st.latex(
        r"""{\displaystyle f(\mathbf {x} )=\theta (\mathbf {w} \cdot \mathbf {x} +b)}"""
    )

with knn:
    st.markdown(
        """
    knn
    """
    )

with slope:
    st.markdown(
        """
    slope
    """
    )

with gbc:
    st.markdown(
        """
    gbc
    """
    )

with svd:
    st.markdown(
        """
    svd
    """
    )
