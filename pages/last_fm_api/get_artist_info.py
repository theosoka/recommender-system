import requests
from env.constants import API_KEY


def get_top_tracks_for_artists(artist_names: list[str]) -> dict[str, list[str]]:
    top_tracks_dict = {}

    for artist_name in artist_names:
        payload = {
            "api_key": API_KEY,
            "method": "artist.gettoptracks",
            "artist": artist_name,
            "format": "json",
            "limit": 5,
        }

        r = requests.get("https://ws.audioscrobbler.com/2.0/", params=payload)
        top_tracks_data = r.json().get("toptracks", {}).get("track", [])

        top_tracks = [track["name"] for track in top_tracks_data]
        top_tracks_dict[artist_name] = top_tracks

    return top_tracks_dict
