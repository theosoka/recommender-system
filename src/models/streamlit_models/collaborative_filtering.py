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
    user_artist_matrix: Optional[csr_matrix] = None
    artist_user_matrix: Optional[csr_matrix] = None
    artist_similarity: Optional[cosine_similarity] = None
    user_label_encoder: Optional[LabelEncoder] = None
    artist_label_encoder: Optional[LabelEncoder] = None

    @property
    def user_ids(self) -> Set:
        return set(self.dataset.userID)

    @property
    def artists_ids(self) -> Set:
        return set(self.dataset.artistID)

    @property
    def group_by_user(self) -> pd.DataFrame:
        return self.dataset.groupby(["userID", "artistID"]).sum().reset_index()

    def get_item_item_similarity(
        self, user_ids: np.ndarray, artist_ids: np.ndarray
    ) -> tuple[cosine_similarity, csr_matrix]:
        self.artist_user_matrix = csr_matrix(
            ([1] * len(user_ids), (artist_ids, user_ids))
        )
        similarity = cosine_similarity(self.artist_user_matrix)
        return similarity, self.artist_user_matrix

    def get_recommendations_from_similarity(
        self, similarity_matrix: cosine_similarity, top_n: int = 10
    ) -> pd.DataFrame:
        user_artist_matrix = csr_matrix(self.artist_user_matrix.T)
        user_artist_scores = user_artist_matrix.dot(similarity_matrix)

        rec_for_crust = []
        for user_id in range(user_artist_scores.shape[0]):
            scores = user_artist_scores[user_id, :]
            purchased_items = user_artist_matrix.indices[
                user_artist_matrix.indptr[user_id] : user_artist_matrix.indptr[
                    user_id + 1
                ]
            ]
            scores[purchased_items] = -1
            top_products_ids = np.argsort(scores)[-top_n:][::-1]
            recommendations = pd.DataFrame(
                top_products_ids.reshape(1, -1),
                index=[user_id],
                columns=[f"Top{i + 1}" for i in range(top_n)],
            )
            rec_for_crust.append(recommendations)

        return pd.concat(rec_for_crust)

    def get_recommendations(self, data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        self.user_label_encoder = LabelEncoder()
        user_ids = self.user_label_encoder.fit_transform(data.userID)

        product_label_encoder = LabelEncoder()
        product_ids = product_label_encoder.fit_transform(data.artistID)

        similarity_matrix, artist_user_sim_matrix = self.get_item_item_similarity(
            user_ids, product_ids
        )

        recommendations = self.get_recommendations_from_similarity(
            similarity_matrix, top_n
        )

        recommendations.index = self.user_label_encoder.inverse_transform(
            recommendations.index
        )

        for col in recommendations.columns:
            recommendations[col] = product_label_encoder.inverse_transform(
                recommendations[col]
            )

        return recommendations
