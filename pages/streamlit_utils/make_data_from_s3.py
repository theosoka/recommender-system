import streamlit as st
from st_files_connection import FilesConnection


conn = st.connection("s3", type=FilesConnection)
s3_folder_path = "plysenko-thesis-datasets/data/hetrec2011-lastfm-2k/"


def get_dataframes():
    artists = conn.read(f"{s3_folder_path}artists.csv", ttl=600)
    tags = conn.read(f"{s3_folder_path}tags.csv", ttl=600)
    user_artists = conn.read(f"{s3_folder_path}user_artists_prepared.csv", ttl=600)
    user_friends = conn.read(f"{s3_folder_path}user_friends.csv", ttl=600)
    user_tagged_artists = conn.read(f"{s3_folder_path}user_tagged_artists.csv", ttl=600)

    return artists, tags, user_artists, user_friends, user_tagged_artists
