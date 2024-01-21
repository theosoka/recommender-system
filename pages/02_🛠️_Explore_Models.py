import streamlit as st

st.set_page_config(page_title="Explore Models", page_icon="📊")

model_names = [
    "Perceptron",
    "KNN",
]

(
    perceptron,
    knn,
) = st.tabs(model_names)

with perceptron:
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
