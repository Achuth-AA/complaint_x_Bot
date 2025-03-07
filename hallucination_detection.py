import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HallucinationDetector:
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def compute_similarity(self, retrieved_docs, generated_response):
        if not retrieved_docs:
            return 0.0
        retrieved_text = " ".join([doc for doc in retrieved_docs])
        if not retrieved_text.strip() or not generated_response.strip():
            return 0.0
        corpus = [retrieved_text, generated_response]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def is_hallucination(self, retrieved_docs, generated_response):
        similarity = self.compute_similarity(retrieved_docs, generated_response)
        is_hallucination = similarity < self.threshold
        return is_hallucination, similarity