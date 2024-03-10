from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class HybridRecommender:
    def __init__(self, users, items):
        self.users = users
        self.items = items
        self.user_similarity = cosine_similarity(users)
        self.item_similarity = cosine_similarity(items)

    def recommend(self, user_id, top_n=10):
        user_similarities = self.user_similarity[user_id]
        item_similarities = self.item_similarity[user_id]

        user_based_contrib = user_similarities.dot(self.users) / user_similarities.sum()
        item_based_contrib = item_similarities.dot(self.items) / item_similarities.sum()

        hybrid_score = user_based_contrib + item_based_contrib
        recommended_items = np.argsort(-hybrid_score)[:top_n]
        # return recommended_items
        
        # 忽略已经评分的物品
        rated_items = np.where(self.users[user_id] != 0)[0]
        recommended_items = [item for item in np.argsort(-hybrid_score) if item not in rated_items]
        return recommended_items[:top_n]
    
# user-item matrix
users = np.array([
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 0]
])

items = np.array([
    [1, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 1]
])

recommender = HybridRecommender(users, items)
recommendation  = recommender.recommend(1, 2)
print(recommendation)
