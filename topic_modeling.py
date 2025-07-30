import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
warnings.filterwarnings('ignore')

class TopicModelingExtractor:
    """
    Implements LDA topic modeling for synopsis and reviews.
    Extracts topic distributions and calculates cosine similarity.
    """
    
    def __init__(self, n_topics: int = 10, max_features: int = 1000, random_state: int = 42):
        """
        Initialize topic modeling extractor.
        
        Args:
            n_topics: Number of topics for LDA
            max_features: Maximum features for vectorizer
            random_state: Random state for reproducibility
        """
        self.n_topics = n_topics
        self.max_features = max_features
        self.random_state = random_state
        
        # Initialize LDA models
        self.synopsis_lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=20,
            learning_method='batch'
        )
        self.review_lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=20,
            learning_method='batch'
        )
        
        # Initialize vectorizers
        self.synopsis_vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.review_vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        self.is_fitted = False
        
    def fit_topic_models(self, synopsis_texts: List[str], review_texts: List[str]) -> Dict:
        """
        Fit LDA models on synopsis and review texts.
        
        Args:
            synopsis_texts: List of synopsis texts
            review_texts: List of review texts
            
        Returns:
            Dictionary with fitted models and vectorizers
        """
        print(f"    📚 Fitting LDA topic models ({self.n_topics} topics)...")
        
        # Clean and prepare texts
        synopsis_texts = [text if text else "no synopsis" for text in synopsis_texts]
        review_texts = [text if text else "no reviews" for text in review_texts]
        
        # Fit synopsis model
        print(f"      📖 Fitting synopsis topic model...")
        synopsis_features = self.synopsis_vectorizer.fit_transform(synopsis_texts)
        self.synopsis_lda.fit(synopsis_features)
        
        # Fit review model
        print(f"      📝 Fitting review topic model...")
        review_features = self.review_vectorizer.fit_transform(review_texts)
        self.review_lda.fit(review_features)
        
        self.is_fitted = True
        
        return {
            'synopsis_lda': self.synopsis_lda,
            'review_lda': self.review_lda,
            'synopsis_vectorizer': self.synopsis_vectorizer,
            'review_vectorizer': self.review_vectorizer
        }
    
    def extract_topic_distributions(self, synopsis_text: str, review_text: str, drama_id: str = None) -> Dict:
        """
        Extract topic distributions for a single drama.
        
        Args:
            synopsis_text: Drama synopsis text
            review_text: Drama review text
            drama_id: Drama identifier (optional)
            
        Returns:
            Dictionary with topic distributions
        """
        if not self.is_fitted:
            print("    ⚠️ Topic models not fitted yet")
            return {'synopsis_topics': None, 'review_topics': None}
        
        # Clean texts
        synopsis_text = synopsis_text if synopsis_text else "no synopsis"
        review_text = review_text if review_text else "no reviews"
        
        # Extract synopsis topics
        synopsis_features = self.synopsis_vectorizer.transform([synopsis_text])
        synopsis_topics = self.synopsis_lda.transform(synopsis_features)[0]
        
        # Extract review topics
        review_features = self.review_vectorizer.transform([review_text])
        review_topics = self.review_lda.transform(review_features)[0]
        
        return {
            'synopsis_topics': synopsis_topics,
            'review_topics': review_topics
        }
    
    def build_profile_topic_distributions(self, dramas: List[Dict], weights: np.ndarray) -> Dict:
        """
        Build weighted average topic distributions for user profile.
        
        Args:
            dramas: List of drama dictionaries
            weights: Weight array for each drama
            
        Returns:
            Dictionary with profile topic distributions
        """
        if not self.is_fitted:
            print("    ⚠️ Topic models not fitted yet")
            return {'synopsis_topics': None, 'review_topics': None}
        
        print(f"    📚 Building profile topic distributions...")
        
        # Collect all texts
        synopsis_texts = [drama.get('synopsis', '') for drama in dramas]
        review_texts = [drama.get('reviews', '') for drama in dramas]
        
        # Fit models if not already fitted
        if not self.is_fitted:
            self.fit_topic_models(synopsis_texts, review_texts)
        
        # Calculate weighted average topic distributions
        synopsis_profile = np.zeros(self.n_topics)
        review_profile = np.zeros(self.n_topics)
        
        total_weight = np.sum(weights)
        
        for i, drama in enumerate(dramas):
            topic_dist = self.extract_topic_distributions(
                drama.get('synopsis', ''),
                drama.get('reviews', '')
            )
            
            weight = weights[i] / total_weight
            
            if topic_dist['synopsis_topics'] is not None:
                synopsis_profile += weight * topic_dist['synopsis_topics']
            
            if topic_dist['review_topics'] is not None:
                review_profile += weight * topic_dist['review_topics']
        
        return {
            'synopsis_topics': synopsis_profile,
            'review_topics': review_profile
        }
    
    def calculate_topic_similarity(self, drama: Dict) -> Dict:
        """
        Calculate cosine similarity between drama and profile topic distributions.
        
        Args:
            drama: Drama dictionary
            
        Returns:
            Dictionary with topic similarities
        """
        if not self.is_fitted or not hasattr(self, 'profile_topics'):
            return {'synopsis_topics': 0, 'review_topics': 0}
        
        # Extract drama topic distributions
        drama_topics = self.extract_topic_distributions(
            drama.get('synopsis', ''),
            drama.get('reviews', '')
        )
        
        # Calculate similarities
        synopsis_similarity = 0
        review_similarity = 0
        
        if (drama_topics['synopsis_topics'] is not None and 
            self.profile_topics['synopsis_topics'] is not None):
            synopsis_similarity = cosine_similarity(
                drama_topics['synopsis_topics'].reshape(1, -1),
                self.profile_topics['synopsis_topics'].reshape(1, -1)
            )[0, 0]
        
        if (drama_topics['review_topics'] is not None and 
            self.profile_topics['review_topics'] is not None):
            review_similarity = cosine_similarity(
                drama_topics['review_topics'].reshape(1, -1),
                self.profile_topics['review_topics'].reshape(1, -1)
            )[0, 0]
        
        return {
            'synopsis_topics': synopsis_similarity,
            'review_topics': review_similarity
        }
    
    def set_profile_topics(self, profile_topics: Dict):
        """Set the profile topic distributions for similarity calculation."""
        self.profile_topics = profile_topics
    
    def get_topic_keywords(self, top_n: int = 10) -> Dict:
        """
        Extract top keywords for each topic.
        
        Args:
            top_n: Number of top keywords to extract
            
        Returns:
            Dictionary with topic keywords
        """
        if not self.is_fitted:
            return {}
        
        keywords = {}
        
        # Get synopsis topic keywords
        if hasattr(self, 'synopsis_vectorizer'):
            feature_names = self.synopsis_vectorizer.get_feature_names_out()
            synopsis_keywords = {}
            
            for i in range(self.n_topics):
                topic_weights = self.synopsis_lda.components_[i]
                top_indices = np.argsort(topic_weights)[-top_n:][::-1]
                synopsis_keywords[f'topic_{i}'] = [feature_names[idx] for idx in top_indices]
            
            keywords['synopsis'] = synopsis_keywords
        
        # Get review topic keywords
        if hasattr(self, 'review_vectorizer'):
            feature_names = self.review_vectorizer.get_feature_names_out()
            review_keywords = {}
            
            for i in range(self.n_topics):
                topic_weights = self.review_lda.components_[i]
                top_indices = np.argsort(topic_weights)[-top_n:][::-1]
                review_keywords[f'topic_{i}'] = [feature_names[idx] for idx in top_indices]
            
            keywords['review'] = review_keywords
        
        return keywords
    
    def save_models(self, filepath: str = 'topic_models.pkl'):
        """Save fitted models to file."""
        if not self.is_fitted:
            print("No models to save")
            return
        
        models = {
            'synopsis_lda': self.synopsis_lda,
            'review_lda': self.review_lda,
            'synopsis_vectorizer': self.synopsis_vectorizer,
            'review_vectorizer': self.review_vectorizer,
            'n_topics': self.n_topics,
            'max_features': self.max_features,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(models, f)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str = 'topic_models.pkl'):
        """Load fitted models from file."""
        try:
            with open(filepath, 'rb') as f:
                models = pickle.load(f)
            
            self.synopsis_lda = models['synopsis_lda']
            self.review_lda = models['review_lda']
            self.synopsis_vectorizer = models['synopsis_vectorizer']
            self.review_vectorizer = models['review_vectorizer']
            self.n_topics = models['n_topics']
            self.max_features = models['max_features']
            self.random_state = models['random_state']
            self.is_fitted = True
            
            print(f"Models loaded from {filepath}")
            
        except FileNotFoundError:
            print(f"Model file {filepath} not found")
        except Exception as e:
            print(f"Error loading models: {e}") 