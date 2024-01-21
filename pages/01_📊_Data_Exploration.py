import pandas as pd
import ydata_profiling
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(page_title="Data Exploration", page_icon="📊")
st.title("Datasets Profiles")

datasets_names = [
    "artists",
    "tags",
    "user_artists",
    "user_friends",
    "user_taggedartists",
]

artists, tags, user_artists, user_friends, user_taggedartists = st.tabs(datasets_names)


with artists:
    artists_df = pd.read_csv(
        "/Users/polina/study/THESIS/recommender-system/data/processed/lastfm_2k/artists.csv"
    )
    pr = artists_df.profile_report()
    st_profile_report(pr)

with tags:
    tags_df = pd.read_csv(
        "/Users/polina/study/THESIS/recommender-system/data/processed/lastfm_2k/tags.csv"
    )
    pr_tags = tags_df.profile_report()
    st_profile_report(pr_tags)

with user_artists:
    user_artists_df = pd.read_csv(
        "/Users/polina/study/THESIS/recommender-system/data/processed/lastfm_2k/user_artists.csv"
    )
    pr_user_artists = user_artists_df.profile_report()
    st_profile_report(pr_user_artists)

with user_friends:
    user_friends_df = pd.read_csv(
        "/Users/polina/study/THESIS/recommender-system/data/processed/lastfm_2k/user_friends.csv"
    )
    pr_user_friends = user_friends_df.profile_report()
    st_profile_report(pr_user_friends)

with user_taggedartists:
    user_taggedartists_df = pd.read_csv(
        "/Users/polina/study/THESIS/recommender-system/data/processed/lastfm_2k/user_taggedartists.csv"
    )
    pr_user_taggedartists = user_taggedartists_df.profile_report()
    st_profile_report(pr_user_taggedartists)
