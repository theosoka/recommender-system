import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from pages.streamlit_utils.make_data_from_s3 import get_dataframes

st.set_page_config(page_title="Data Exploration", page_icon="📊")
st.title("Datasets Profiles")

datasets_names = [
    "artists",
    "tags",
    "user_artists",
    "user_friends",
    "user_tagged_artists",
]

artists, tags, user_artists, user_friends, user_tagged_artists = st.tabs(datasets_names)
(
    artists_df,
    tags_df,
    user_artists_df,
    user_friends_df,
    user_tagged_artists_df,
) = get_dataframes()

with artists:
    pr = artists_df.profile_report()
    st_profile_report(pr)

with tags:
    pr_tags = tags_df.profile_report()
    st_profile_report(pr_tags)

with user_artists:
    pr_user_artists = user_artists_df.profile_report()
    st_profile_report(pr_user_artists)

with user_friends:
    pr_user_friends = user_friends_df.profile_report()
    st_profile_report(pr_user_friends)

with user_tagged_artists:
    pr_user_tagged_artists = user_tagged_artists_df.profile_report()
    st_profile_report(pr_user_tagged_artists)
