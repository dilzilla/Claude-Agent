from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class TopicModeler:
    def __init__(self, num_topics=5):
        self.num_topics = num_topics
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    
    def extract_topics(self, documents):
        # Convert documents to TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Apply Latent Dirichlet Allocation
        lda_model = LatentDirichletAllocation(
            n_components=self.num_topics, 
            random_state=42, 
            learning_method='batch'
        )
        lda_output = lda_model.fit_transform(tfidf_matrix)
        
        # Extract top words for each topic
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_features_ind = topic.argsort()[:-10 - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topics.append({
                'topic_number': topic_idx,
                'top_words': top_features
            })
        
        return topics