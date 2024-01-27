import requests
from env.constants import API_KEY


def lastfm_get_artists_urls(artists_names: list[str]) -> dict[str, str]:
    artists_urls = dict()
    for artist_name in artists_names:
        payload = {
            "api_key": API_KEY,
            "method": "artist.getinfo",
            "artist": artist_name,
            "format": "json",
        }
        r = requests.get("https://ws.audioscrobbler.com/2.0/", params=payload)
        link = r.json()["artist"]["bio"]["links"]["link"]["href"]
        artists_urls[artist_name] = link
    return artists_urls
