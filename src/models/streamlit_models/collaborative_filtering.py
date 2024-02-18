from dataclasses import dataclass
from typing import Optional, Set

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

from src.models.basic_models_exploration.model_mixin import ModelMixin


@dataclass
class CollaborativeFiltering(ModelMixin):
    """
    docs
    https://medium.com/towards-data-science/recommend-using-scikit-learn-and-tensorflow-recommender-bc659d91301a
    https://surprise.readthedocs.io/en/stable/FAQ.html
    """

    user_artist_matrix: Optional[csr_matrix] = None
    artist_user_matrix: Optional[csr_matrix] = None
    artist_similarity: Optional[cosine_similarity] = None
    user_label_encoder: Optional[LabelEncoder] = None
    artist_label_encoder: Optional[LabelEncoder] = None

    @property
    def unique_user_ids(self) -> Set:
        return set(self.dataset.userID)

    @property
    def unique_artist_ids(self) -> Set:
        return set(self.dataset.artistID)

    @property
    def user_artist_interaction(self) -> pd.DataFrame:
        return self.dataset.groupby(["userID", "artistID"]).sum().reset_index()

    def calculate_artist_similarity(
        self, user_ids: np.ndarray, artist_ids: np.ndarray
    ) -> tuple[cosine_similarity, csr_matrix]:
        self.artist_user_matrix = csr_matrix(
            ([1] * len(user_ids), (artist_ids, user_ids))
        )
        similarity_matrix = cosine_similarity(self.artist_user_matrix)
        return similarity_matrix, self.artist_user_matrix

    def generate_recommendations_from_similarity(
        self, similarity_matrix: cosine_similarity, top_n: int = 10
    ) -> pd.DataFrame:
        user_artist_matrix_transposed = csr_matrix(self.artist_user_matrix.T)
        user_artist_scores = user_artist_matrix_transposed.dot(similarity_matrix)

        recommendations_list = []
        for user_id in range(user_artist_scores.shape[0]):
            scores = user_artist_scores[user_id, :]
            listened_artists = user_artist_matrix_transposed.indices[
                user_artist_matrix_transposed.indptr[
                    user_id
                ] : user_artist_matrix_transposed.indptr[user_id + 1]
            ]
            scores[listened_artists] = -1
            top_artist_ids = np.argsort(scores)[-top_n:][::-1]
            recommendations = pd.DataFrame(
                top_artist_ids.reshape(1, -1),
                index=[user_id],
                columns=[f"Top{i + 1}" for i in range(top_n)],
            )
            recommendations_list.append(recommendations)

        return pd.concat(recommendations_list)

    def generate_recommendations(
        self, data: pd.DataFrame, top_n: int = 10
    ) -> pd.DataFrame:
        self.user_label_encoder = LabelEncoder()
        encoded_user_ids = self.user_label_encoder.fit_transform(data.userID)

        encoded_artist_encoder = LabelEncoder()
        encoded_artist_ids = encoded_artist_encoder.fit_transform(data.artistID)

        (
            similarity_matrix,
            artist_user_similarity_matrix,
        ) = self.calculate_artist_similarity(encoded_user_ids, encoded_artist_ids)

        recommendations_df = self.generate_recommendations_from_similarity(
            similarity_matrix, top_n
        )

        recommendations_df.index = self.user_label_encoder.inverse_transform(
            recommendations_df.index
        )

        for col in recommendations_df.columns:
            recommendations_df[col] = encoded_artist_encoder.inverse_transform(
                recommendations_df[col]
            )

        return recommendations_df
