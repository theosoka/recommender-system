import time
from pathlib import Path

import pandas as pd
import streamlit as st
from src.models.streamlit_models.collaborative_filtering import CollaborativeFiltering
from pages.last_fm_api.get_artist_info import lastfm_get_artists_urls

st.set_page_config(page_title="Get Recommendations", page_icon="📊")

top_n_options = (10, 20, 30, 40)

top_n = st.selectbox(
    "How many recommended artists do you want to get?",
    top_n_options,
    index=None,
    placeholder="Select an amount..",
)


artists_df = pd.read_csv(Path() / "data/processed/lastfm_2k/artists.csv")
artists = artists_df.name
selected_artists = st.multiselect(
    "Choose min. 10 artists that you like",
    options=artists,
    placeholder="Choose artists...",
)
DATASET = pd.read_csv(Path() / "data/processed/lastfm_2k/user_artists_prepared.csv")


def append_new_user_data(selected_items: list[str]):
    ids = list(artists_df[artists_df.name.isin(selected_artists)].id)
    new_data = pd.DataFrame(
        {
            "userID": [1] * len(selected_items),
            "artistID": list(ids),
            "weight": [1] * len(selected_items),
        }
    )
    return pd.concat([DATASET, new_data], ignore_index=True)


def display_artists(artist_dict):
    heart_emojis = ["🧡", "💛", "💚", "💙", "💜", "❤️"]
    for i, (name, link) in enumerate(artist_dict.items()):
        heart_emoji = heart_emojis[i % len(heart_emojis)]
        st.markdown(f"{heart_emoji} {name} - {link}")


def display_recommendations(selected_items: list[str], top_n_rec: int):
    updated_dataset = append_new_user_data(selected_items)
    model = CollaborativeFiltering(
        model_name="Collaborative Filtering", dataset=updated_dataset
    )
    new_user_recommendations = model.get_recommendations(
        updated_dataset, top_n_rec
    ).iloc[0]
    artist_names = artists_df[artists_df.id.isin(list(new_user_recommendations))].name
    artists_urls = lastfm_get_artists_urls(artist_names)
    display_artists(artists_urls)


if st.button("Get artists recommendations", type="primary"):
    if not (len(selected_artists) > 9 or not top_n_options):
        st.write(
            "You have to choose number of recommendations and choose more than 10 artists."
        )
    else:
        computation_message = st.empty()

        computation_message.write("Recommendations will appear shortly.")
        time.sleep(3)
        computation_message.empty()
        display_recommendations(selected_artists, top_n)
