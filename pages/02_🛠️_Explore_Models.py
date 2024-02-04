import streamlit as st

st.set_page_config(page_title="Explore Models", page_icon="📊")

model_names = [
    "Item-Item Collaborative Filtering",
    "KNN",
    "Slope One",
    "SVD",
]

(cf, knn, slope, svd) = st.tabs(model_names)

with cf:
    st.markdown(
        """**Collaborative filtering** is an effective approach that is widely used in the recommender systems. 
        It is based on the assumption that users who have similar preferences and history of interactions with items 
        in the past will have similar preferences in the future. The items don’t have to necessarily share many common 
        traits and be from the same category, like it’s in the content-based systems. And this is a noticeable appeal 
        of CF systems - it is domain-free and can address elusive data aspects."""
    )

    st.image("https://miro.medium.com/v2/resize:fit:649/1*Z8p9PAqx2dFfEn76B6juAw.png")
    st.markdown(
        "<small>Image source: https://medium.com/@ashmi_banerjee/understanding-collaborative-filtering-f1f496c673fd</small>",
        unsafe_allow_html=True,
    )

with knn:
    st.markdown(
        """
    The **k-nearest neighbors algorithm**, also known as KNN or k-NN, is a non-parametric, supervised learning 
    classifier, which uses proximity to make classifications or predictions about the grouping of an individual data 
    point. While it can be used for either regression or classification problems, it is typically used as a 
    classification algorithm, working off the assumption that similar points can be found near one another.
    """
    )
    st.image(
        "https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/ef/3a/KNN.component.complex-narrative-xl-retina.ts=1706298230790.png/content/adobe-cms/us/en/topics/knn/jcr:content/root/table_of_contents/body/content_section_styled/content-section-body/complex_narrative/items/content_group/image"
    )
    st.markdown(
        "<small>Image source: https://www.ibm.com/topics/knn</small>",
        unsafe_allow_html=True,
    )

with slope:
    st.markdown(
        """
    **Slope One** is a family of algorithms used for collaborative filtering, introduced in a 2005 paper by Daniel
     Lemire and Anna Maclachlan. Arguably, it is the simplest form of non-trivial item-based collaborative 
     filtering based on ratings. Their simplicity makes it especially easy to implement them efficiently while their 
     accuracy is often on par with more complicated and computationally expensive algorithms. They have also been used 
     as building blocks to improve other algorithms[Wikipedia].
    """
    )

with svd:
    st.markdown(
        """
    By definition, **Singular Value Decomposition**(more often called SVD) is a *generalisation of the eigen-decomposition 
    which can be used to analyse rectangular matrices (the eigen-decomposition is defined only for squared matrices)*.
It is a powerful matrix factorization technique commonly used in recommender systems to uncover underlying factors and 
patterns in user-item interaction data. This has proven effective in improving recommendation accuracy and providing 
personalised recommendations. SVD was a major breakthrough in the Netflix movie recommendation
 system that went by the name Netflix Prize. This method allows to have initial data in a compressed way, preserving 
 all information, but taking significantly less space.
    """
    )
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Singular-Value-Decomposition.svg/1920px-Singular-Value-Decomposition.svg.png"
    )
    st.markdown(
        "<small>Image source: https://en.wikipedia.org/wiki/Singular_value_decomposition#/media/File:Singular-Value-Decomposition.svg</small>",
        unsafe_allow_html=True,
    )
