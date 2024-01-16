import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

from train_model import ModelMixin


@dataclass
class CollaborativeFiltering(ModelMixin):
    user_artist_matrix: Optional[csr_matrix] = None
    artist_similarity: Optional[cosine_similarity] = None
    user_label_encoder: Optional = None
    artist_label_encoder: Optional = None

    @property
    def user_ids(self):
        pass

    @property
    def artists_ids(self):
        pass

    @property
    def group_by_user(self) -> pd.DataFrame:
        return self.dataset.groupby(["userID", "artistID"]).sum().reset_index()

    def get_item_item_similarity(self, user_ids, artist_ids):
        artist_user_matrix = csr_matrix(([1] * len(user_ids), (artist_ids, user_ids)))

        similarity = cosine_similarity(artist_user_matrix)
        return similarity, artist_user_matrix

    def get_recommendations_from_similarity(
        self, similarity_matrix, artist_user_matrix, top_n=10
    ):
        user_artist_matrix = csr_matrix(artist_user_matrix.T)
        user_artist_scores = user_artist_matrix.dot(similarity_matrix)
        rec_for_crust = []
        for user_id in range(user_artist_scores.shape[0]):
            scores = user_artist_scores[user_id, :]
        purchased_items = user_artist_matrix.indices[
            user_artist_matrix.indptr[user_id] : user_artist_matrix.indptr[user_id + 1]
        ]
        scores[purchased_items] = -1  # do not recommend already purchased SalesItems
        top_products_ids = np.argsort(scores)[-top_n:][::-1]
        recommendations = pd.DataFrame(
            top_products_ids.reshape(1, -1),
            index=[user_id],
            columns=["Top%s" % (i + 1) for i in range(top_n)],
        )
        rec_for_crust.append(recommendations)
        return pd.concat(rec_for_crust)

    def fit(self, data):
        self.dataset = data
        self.user_label_encoder = LabelEncoder()
        self.artist_label_encoder = LabelEncoder()

        user_ids = self.user_label_encoder.fit_transform(data.userID)
        artist_ids = self.artist_label_encoder.fit_transform(data.artistID)

        self.user_artist_matrix = csr_matrix(
            ([1] * len(user_ids), (artist_ids, user_ids))
        )

        self.artist_similarity = cosine_similarity(self.user_artist_matrix)

    def get_recommendations_for_new_user(self, favorite_artists: list[str], top_n=10):
        if self.user_artist_matrix is None or self.artist_similarity is None:
            raise ValueError("Model not fitted. Please call fit() method first.")

        # Transform favorite artists to encoded IDs
        favorite_artist_ids = self.artist_label_encoder.transform(favorite_artists)

        # Create a user vector for the new user
        new_user_vector = csr_matrix(
            (
                [1] * len(favorite_artist_ids),
                (favorite_artist_ids, [0] * len(favorite_artist_ids)),
            )
        )

        # Extend the new user vector to include all artists in the original matrix
        all_artist_ids = range(self.user_artist_matrix.shape[0])
        new_user_vector_extended = csr_matrix(
            ([1] * len(all_artist_ids), (all_artist_ids, [0] * len(all_artist_ids)))
        )

        # Calculate similarity between the new user and existing artists
        similarity_with_existing_artists = cosine_similarity(
            self.user_artist_matrix.T, new_user_vector_extended.T
        )

        print("User Artist Matrix Shape:", self.user_artist_matrix.shape)
        print("Similarity Matrix Shape:", similarity_with_existing_artists.shape)

        # Calculate recommendations for the new user
        scores = (
            self.user_artist_matrix.dot(similarity_with_existing_artists.T)
            .todense()
            .A.flatten()
        )
        top_products_ids = np.argsort(scores)[-top_n:][::-1]

        # Convert back to artist names
        recommendations = self.artist_label_encoder.inverse_transform(top_products_ids)

        return recommendations


dataset = pd.read_csv(Path() / "../../data/processed/lastfm_2k/user_artists.csv")

model = CollaborativeFiltering(model_name="Collaborative Filtering", dataset=dataset)
model.fit(
    dataset
)  # You need to fit the model with the existing dataset before making recommendations
artists = [
    "Moonspell",
    "Marilyn Manson",
    "MALICE MIZER",
    "Diary of Dreams",
    "Tamtrum",
    "Amduscia",
    "Covenant",
    "KMFDM",
    "Emperor",
    "Morcheeba",
]
artists_df = pd.read_csv(Path() / "../../data/processed/lastfm_2k/artists.csv")

ids = artists_df[artists_df.name.isin(artists)].id

recommendations = model.get_recommendations_for_new_user(list(ids))
print(recommendations)
