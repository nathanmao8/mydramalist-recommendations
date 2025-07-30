import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import json
import os
from topic_modeling import TopicModelingExtractor
from performance_weighting import PerformanceBasedWeighting

class UserTasteProfile:
    """
    Builds and analyzes user taste profiles from watched dramas.
    Provides similarity scoring for unwatched dramas against the user's preferences.
    """
    
    def __init__(self, semantic_extractor=None, topic_extractor=None):
        """
        Initialize the taste profile system.
        
        Args:
            semantic_extractor: SemanticSimilarityExtractor instance for embeddings
            topic_extractor: TopicModelingExtractor instance for topic modeling
        """
        self.semantic_extractor = semantic_extractor
        self.topic_extractor = topic_extractor or TopicModelingExtractor()
        self.taste_profile = {}
        self.profile_embeddings = {}
        self.scaler = StandardScaler()
        self.performance_weighting = PerformanceBasedWeighting()
        self.component_weights = {'categorical': 0.45, 'text': 0.20, 'semantic': 0.35}
    
    def _normalize_similarity(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Normalize a similarity value to 0-1 range.
        
        Args:
            value: Raw similarity value
            min_val: Minimum expected value (default 0.0)
            max_val: Maximum expected value (default 1.0)
            
        Returns:
            Normalized value between 0 and 1
        """
        if max_val <= min_val:
            return 0.0
        
        # Clip value to expected range
        clipped_value = np.clip(value, min_val, max_val)
        
        # Normalize to 0-1
        normalized = (clipped_value - min_val) / (max_val - min_val)
        
        return normalized
        
    def build_taste_profile(self, watched_dramas: List[Dict], ratings: List[float]) -> Dict:
        """
        Build comprehensive taste profile from watched dramas and ratings.
        
        Args:
            watched_dramas: List of drama dictionaries
            ratings: List of user ratings (1-10 scale)
            
        Returns:
            Dictionary containing taste profile components
        """
        if len(watched_dramas) != len(ratings):
            raise ValueError("Number of dramas must match number of ratings")
        
        # Store training dramas for later semantic similarity calculation
        self.training_dramas = watched_dramas
        
        # Normalize rating weights
        rating_weights = np.array(ratings) / np.sum(ratings)
        
        # 1. Categorical Preferences (genres, tags, cast, crew)
        categorical_profile = self._build_categorical_profile(watched_dramas, rating_weights)
        
        # 2. Text Content Preferences (synopsis, reviews)
        text_profile = self._build_text_profile(watched_dramas, rating_weights)
        
        # 3. Semantic Similarity Patterns
        semantic_profile = self._build_semantic_profile(watched_dramas, rating_weights)
        
        # 4. Rating Patterns
        rating_profile = self._build_rating_profile(ratings)
        
        # Build complete taste profile
        self.taste_profile = {
            'categorical': categorical_profile,
            'text': text_profile,
            'semantic': semantic_profile,
            'rating': rating_profile
        }
        
        return self.taste_profile
    
    def _build_categorical_profile(self, dramas: List[Dict], weights: np.ndarray) -> Dict:
        """Build profile of categorical preferences (genres, tags, cast, crew)."""
        profile = {
            'genres': defaultdict(float),
            'tags': defaultdict(float),
            'cast': defaultdict(float),
            'directors': defaultdict(float),
            'screenwriters': defaultdict(float),
            'composers': defaultdict(float)
        }
        
        for drama, weight in zip(dramas, weights):
            # Genres
            for genre in drama.get('genres', []):
                profile['genres'][genre] += weight
            
            # Tags
            for tag in drama.get('tags', []):
                profile['tags'][tag] += weight
            
            # Cast
            for actor in drama.get('main_cast', []):
                profile['cast'][actor] += weight
            
            # Crew
            for director in drama.get('directors', []):
                profile['directors'][director] += weight
            for screenwriter in drama.get('screenwriters', []):
                profile['screenwriters'][screenwriter] += weight
            for composer in drama.get('composers', []):
                profile['composers'][composer] += weight
        
        # Convert to sorted lists
        for category in profile:
            profile[category] = dict(sorted(
                profile[category].items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        
        return profile
    
    def _build_text_profile(self, dramas: List[Dict], weights: np.ndarray) -> Dict:
        """Build profile of text content preferences using embeddings and topic modeling."""
        print(f"    📝 Building text profile for {len(dramas)} dramas")
        
        # 1. Build embedding-based profile
        embedding_profile = self._build_embedding_profile(dramas, weights)
        
        # 2. Build topic modeling profile
        topic_profile = self._build_topic_profile(dramas, weights)
        
        return {
            'synopsis_embedding': embedding_profile['synopsis_embedding'],
            'review_embedding': embedding_profile['review_embedding'],
            'synopsis_topics': topic_profile['synopsis_topics'],
            'review_topics': topic_profile['review_topics']
        }
    
    def _build_embedding_profile(self, dramas: List[Dict], weights: np.ndarray) -> Dict:
        """Build profile using embeddings."""
        if not self.semantic_extractor:
            print("    ⚠️  No semantic extractor for embedding profile")
            return {'synopsis_embedding': None, 'review_embedding': None}
        
        synopsis_embeddings = []
        review_embeddings = []
        
        for i, (drama, weight) in enumerate(zip(dramas, weights)):
            drama_id = drama.get('slug', f"drama_{i}")
            
            # Check if drama has text content
            synopsis_text = drama.get('synopsis_clean', '')
            reviews_text = drama.get('reviews_combined', '')
            
            if not synopsis_text and not reviews_text:
                print(f"      ⚠️  Drama {drama.get('title', 'Unknown')} has no text content")
                continue
            
            # Get embeddings
            synopsis_emb = self.semantic_extractor._get_embedding(
                synopsis_text, 'synopsis', drama_id
            )
            review_emb = self.semantic_extractor._get_embedding(
                reviews_text, 'review', drama_id
            )
            
            # Weight by user rating
            synopsis_embeddings.append(synopsis_emb * weight)
            review_embeddings.append(review_emb * weight)
        
        if not synopsis_embeddings or not review_embeddings:
            print("    ⚠️  No valid text embeddings found")
            return {'synopsis_embedding': None, 'review_embedding': None}
        
        # Average weighted embeddings
        avg_synopsis = np.mean(synopsis_embeddings, axis=0)
        avg_review = np.mean(review_embeddings, axis=0)
        
        print(f"    ✅ Embedding profile built with {len(synopsis_embeddings)} synopsis and {len(review_embeddings)} review embeddings")
        
        return {
            'synopsis_embedding': avg_synopsis,
            'review_embedding': avg_review
        }
    
    def _build_topic_profile(self, dramas: List[Dict], weights: np.ndarray) -> Dict:
        """Build profile using topic modeling."""
        if not self.topic_extractor:
            print("    ⚠️  No topic extractor for topic profile")
            return {'synopsis_topics': None, 'review_topics': None}
        
        # Build topic distributions for the user profile
        topic_profile = self.topic_extractor.build_profile_topic_distributions(dramas, weights)
        
        return topic_profile
    
    def _build_semantic_profile(self, dramas: List[Dict], weights: np.ndarray) -> Dict:
        """Build profile of semantic similarity patterns."""
        if not self.semantic_extractor:
            return {}
        
        # Extract semantic similarity features for watched dramas
        semantic_features = self.semantic_extractor.extract_similarity_features(dramas)
        
        # Weight by user ratings
        weighted_features = semantic_features * weights.reshape(-1, 1)
        
        # Average weighted semantic features
        avg_semantic = np.mean(weighted_features, axis=0)
        
        return {
            'avg_synopsis_similarity': avg_semantic[0],
            'max_synopsis_similarity': avg_semantic[1],
            'max_review_similarity': avg_semantic[2]
        }
    
    def _build_rating_profile(self, ratings: List[float]) -> Dict:
        """Build profile of rating patterns."""
        ratings_array = np.array(ratings)
        
        return {
            'mean_rating': np.mean(ratings_array),
            'std_rating': np.std(ratings_array),
            'min_rating': np.min(ratings_array),
            'max_rating': np.max(ratings_array),
            'high_rated_threshold': np.percentile(ratings_array, 75),  # Top 25%
            'low_rated_threshold': np.percentile(ratings_array, 25),   # Bottom 25%
            'rating_distribution': {
                '1-3': np.sum(ratings_array <= 3),
                '3-5': np.sum((ratings_array > 3) & (ratings_array <= 5)),
                '5-7': np.sum((ratings_array > 5) & (ratings_array <= 7)),
                '7-9': np.sum((ratings_array > 7) & (ratings_array <= 9)),
                '9-10': np.sum(ratings_array > 9)
            }
        }
    
    def calculate_taste_similarity(self, drama: Dict) -> Dict:
        """
        Calculate similarity between a drama and the user's taste profile.
        
        Args:
            drama: Drama dictionary with features
            
        Returns:
            Dictionary with similarity scores for different aspects
        """
        if not self.taste_profile:
            raise ValueError("Taste profile not built. Call build_taste_profile() first.")
        
        similarities = {}
        
        # 1. Categorical Similarity
        similarities['categorical'] = self._calculate_categorical_similarity(drama)
        
        # 2. Text Similarity
        similarities['text'] = self._calculate_text_similarity(drama)
        
        # 3. Semantic Similarity
        similarities['semantic'] = self._calculate_semantic_similarity(drama)
        
        # 4. Overall Similarity (weighted combination)
        similarities['overall'] = self._calculate_overall_similarity(similarities)
        
        return similarities
    
    def _calculate_categorical_similarity(self, drama: Dict) -> Dict:
        """Calculate similarity for categorical features."""
        profile = self.taste_profile['categorical']
        similarities = {}
        
        # Genre similarity
        drama_genres = set(drama.get('genres', []))
        if drama_genres:
            genre_scores = [profile['genres'].get(genre, 0) for genre in drama_genres]
            similarities['genres'] = np.mean(genre_scores) if genre_scores else 0
        else:
            similarities['genres'] = 0
        
        # Tag similarity
        drama_tags = set(drama.get('tags', []))
        if drama_tags:
            tag_scores = [profile['tags'].get(tag, 0) for tag in drama_tags]
            similarities['tags'] = np.mean(tag_scores) if tag_scores else 0
        else:
            similarities['tags'] = 0
        
        # Cast similarity
        drama_cast = set(drama.get('main_cast', []))
        if drama_cast:
            cast_scores = [profile['cast'].get(actor, 0) for actor in drama_cast]
            similarities['cast'] = np.mean(cast_scores) if cast_scores else 0
        else:
            similarities['cast'] = 0
        
        # Crew similarity
        drama_directors = set(drama.get('directors', []))
        drama_screenwriters = set(drama.get('screenwriters', []))
        drama_composers = set(drama.get('composers', []))
        
        if drama_directors:
            director_scores = [profile['directors'].get(director, 0) for director in drama_directors]
            similarities['directors'] = np.mean(director_scores) if director_scores else 0
        else:
            similarities['directors'] = 0
            
        if drama_screenwriters:
            screenwriter_scores = [profile['screenwriters'].get(sw, 0) for sw in drama_screenwriters]
            similarities['screenwriters'] = np.mean(screenwriter_scores) if screenwriter_scores else 0
        else:
            similarities['screenwriters'] = 0
            
        if drama_composers:
            composer_scores = [profile['composers'].get(composer, 0) for composer in drama_composers]
            similarities['composers'] = np.mean(composer_scores) if composer_scores else 0
        else:
            similarities['composers'] = 0
        
        return similarities
    
    def _calculate_text_similarity(self, drama: Dict) -> Dict:
        """Calculate similarity for text content using embeddings and topic modeling."""
        text_similarities = {}
        
        # 1. Calculate embedding-based similarities
        embedding_similarities = self._calculate_embedding_similarity(drama)
        text_similarities.update(embedding_similarities)
        
        # 2. Calculate topic-based similarities
        topic_similarities = self._calculate_topic_similarity(drama)
        text_similarities.update(topic_similarities)
        
        return text_similarities
    
    def _calculate_embedding_similarity(self, drama: Dict) -> Dict:
        """Calculate similarity using embeddings."""
        if not self.semantic_extractor or self.taste_profile['text']['synopsis_embedding'] is None:
            return {'synopsis_embedding': 0, 'review_embedding': 0}
        
        drama_id = drama.get('slug', 'drama')
        
        # Get drama embeddings
        drama_synopsis_emb = self.semantic_extractor._get_embedding(
            drama.get('synopsis_clean', ''), 'synopsis', drama_id
        )
        drama_review_emb = self.semantic_extractor._get_embedding(
            drama.get('reviews_combined', ''), 'review', drama_id
        )
        
        # Calculate cosine similarities
        profile_synopsis = self.taste_profile['text']['synopsis_embedding']
        profile_review = self.taste_profile['text']['review_embedding']
        
        synopsis_sim = cosine_similarity([drama_synopsis_emb], [profile_synopsis])[0][0]
        review_sim = cosine_similarity([drama_review_emb], [profile_review])[0][0]
        
        return {
            'synopsis_embedding': synopsis_sim,
            'review_embedding': review_sim
        }
    
    def _calculate_topic_similarity(self, drama: Dict) -> Dict:
        """Calculate similarity using topic modeling."""
        if not self.topic_extractor or self.taste_profile['text']['synopsis_topics'] is None:
            return {'synopsis_topics': 0, 'review_topics': 0}
        
        # Calculate topic similarities
        topic_similarities = self.topic_extractor.calculate_topic_similarity(drama)
        
        return {
            'synopsis_topics': topic_similarities['synopsis_topics'],
            'review_topics': topic_similarities['review_topics']
        }
    
    def _calculate_semantic_similarity(self, drama: Dict) -> float:
        """Calculate semantic similarity."""
        if not self.semantic_extractor or not self.taste_profile.get('semantic'):
            return 0.0
        
        try:
            # Get training dramas from the taste profile
            training_dramas = getattr(self, 'training_dramas', [])
            
            if not training_dramas:
                return 0.0
            
            # Extract semantic features for this drama against training dramas
            drama_semantic_features = self.semantic_extractor.extract_single_drama_semantic_features(
                drama, 
                training_dramas
            )
            
            # Calculate similarities for each semantic feature
            similarities = []
            feature_names = [
                'avg_synopsis_similarity', 'max_synopsis_similarity', 'max_review_similarity'
            ]
            
            for i, feature in enumerate(feature_names):
                if feature in self.taste_profile['semantic']:
                    # Normalize and calculate similarity
                    drama_val = drama_semantic_features[i]
                    profile_val = self.taste_profile['semantic'][feature]
                    
                    # Simple similarity based on how close values are
                    max_val = max(abs(drama_val), abs(profile_val))
                    if max_val > 0:
                        similarity = 1 - abs(drama_val - profile_val) / max_val
                    else:
                        similarity = 1.0
                    
                    similarities.append(similarity)
            
            # Calculate average of semantic similarities
            semantic_similarity = np.mean(similarities) if similarities else 0.0
            
            # INVERT the semantic similarity since it measures "difference" not "similarity"
            # High semantic similarity = very different from watched dramas
            # Low semantic similarity = similar to watched dramas
            # We want dramas similar to watched dramas to have HIGH similarity scores
            inverted_semantic = 1.0 - semantic_similarity
            
            # Normalize the inverted semantic similarity to 0-1 range
            normalized_semantic = self._normalize_similarity(inverted_semantic, min_val=0.5, max_val=1.0)
            
            # Debug output
            if drama.get('title', '') in ['While You Were Sleeping', 'Sweet Home', 'Shogun']:
                print(f"    🧠 Semantic calculation for {drama.get('title')}:")
                print(f"      Raw similarity: {semantic_similarity:.3f}")
                print(f"      Inverted: {inverted_semantic:.3f}")
                print(f"      Normalized: {normalized_semantic:.3f}")
            
            return normalized_semantic
            
        except Exception as e:
            print(f"    ❌ Error calculating semantic similarity for {drama.get('title', 'Unknown')}: {str(e)}")
            return 0.0
    
    def _calculate_overall_similarity(self, similarities: Dict) -> float:
        """Calculate overall similarity score from all components."""
        overall_score = 0
        
        # Categorical similarity (average of all categorical components)
        if 'categorical' in similarities:
            cat_scores = list(similarities['categorical'].values())
            overall_score += self.component_weights['categorical'] * np.mean(cat_scores)
        
        # Text similarity (average of synopsis and review similarities)
        if 'text' in similarities:
            text_scores = list(similarities['text'].values())
            overall_score += self.component_weights['text'] * np.mean(text_scores)
        
        # Semantic similarity (now a single float value)
        if 'semantic' in similarities:
            overall_score += self.component_weights['semantic'] * similarities['semantic']
        
        return overall_score
    
    def get_taste_insights(self) -> Dict:
        """Get interpretable insights about the user's taste profile."""
        if not self.taste_profile:
            raise ValueError("Taste profile not built. Call build_taste_profile() first.")
        
        insights = {
            'top_preferences': {},
            'rating_patterns': {},
            'content_preferences': {}
        }
        
        # Top categorical preferences
        categorical = self.taste_profile['categorical']
        for category, items in categorical.items():
            top_items = list(items.items())[:10]  # Top 10
            insights['top_preferences'][category] = top_items
        
        # Rating patterns
        rating_profile = self.taste_profile['rating']
        insights['rating_patterns'] = {
            'average_rating': rating_profile['mean_rating'],
            'rating_consistency': 1 / (1 + rating_profile['std_rating']),  # Higher = more consistent
            'rating_distribution': rating_profile['rating_distribution'],
            'high_rated_threshold': rating_profile['high_rated_threshold']
        }
        
        # Content preferences (sentiment patterns)
        # Removed sentiment-related insights
        insights['content_preferences'] = {}
        
        return insights
    
    def update_component_weights(self, similarities_df: pd.DataFrame, ratings: List[float]) -> Dict:
        """
        Update component weights based on performance analysis.
        
        Args:
            similarities_df: DataFrame with similarity scores for each component
            ratings: List of user ratings corresponding to the dramas
            
        Returns:
            Dictionary with updated weights
        """
        print("    ⚖️ Analyzing component performance for optimal weighting...")
        
        # Calculate performance metrics
        performance_metrics = self.performance_weighting.calculate_component_performance(
            similarities_df, ratings
        )
        
        # Determine optimal weights
        optimal_weights = self.performance_weighting.determine_optimal_weights(performance_metrics)
        
        # Update component weights
        self.component_weights = optimal_weights
        
        # Display analysis
        analysis = self.performance_weighting.analyze_component_performance(performance_metrics)
        print(analysis)
        
        # Get recommendations
        recommendations = self.performance_weighting.get_weighting_recommendations(performance_metrics)
        if recommendations:
            print("    💡 WEIGHTING RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"      {rec}")
        
        return optimal_weights
    
    def get_component_weights(self) -> Dict:
        """Get current component weights."""
        return self.component_weights.copy()
    
    def set_component_weights(self, weights: Dict):
        """Set component weights manually."""
        self.component_weights = weights.copy()
        print(f"    ⚖️ Component weights updated: {weights}")
    
    def get_performance_analysis(self, similarities_df: pd.DataFrame, ratings: List[float]) -> str:
        """
        Get detailed performance analysis without updating weights.
        
        Args:
            similarities_df: DataFrame with similarity scores for each component
            ratings: List of user ratings corresponding to the dramas
            
        Returns:
            Formatted analysis string
        """
        performance_metrics = self.performance_weighting.calculate_component_performance(
            similarities_df, ratings
        )
        
        analysis = self.performance_weighting.analyze_component_performance(performance_metrics)
        recommendations = self.performance_weighting.get_weighting_recommendations(performance_metrics)
        
        if recommendations:
            analysis += "\n💡 RECOMMENDATIONS:\n"
            analysis += "-" * 20 + "\n"
            for rec in recommendations:
                analysis += f"   {rec}\n"
        
        return analysis
    
    def save_profile(self, filepath: str = 'user_taste_profile.json'):
        """Save taste profile to file."""
        # Convert numpy arrays to lists for JSON serialization
        profile_copy = self.taste_profile.copy()
        
        if 'text' in profile_copy:
            for key in ['synopsis_embedding', 'review_embedding', 'synopsis_topics', 'review_topics']:
                if profile_copy['text'].get(key) is not None:
                    if hasattr(profile_copy['text'][key], 'tolist'):
                        profile_copy['text'][key] = profile_copy['text'][key].tolist()
                    elif isinstance(profile_copy['text'][key], np.ndarray):
                        profile_copy['text'][key] = profile_copy['text'][key].tolist()
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar types
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        profile_copy = convert_numpy_types(profile_copy)
        
        with open(filepath, 'w') as f:
            json.dump(profile_copy, f, indent=2)
        print(f"Taste profile saved to {filepath}")
    
    def load_profile(self, filepath: str = 'user_taste_profile.json'):
        """Load taste profile from file."""
        try:
            with open(filepath, 'r') as f:
                self.taste_profile = json.load(f)
            print(f"✅ Taste profile loaded from {filepath}")
        except Exception as e:
            print(f"❌ Error loading taste profile: {e}")
    
    def get_raw_inverted_semantic(self, drama: Dict) -> float:
        """
        Returns the raw (inverted, pre-normalized) semantic similarity for a drama.
        This is used by TasteAnalyzer for dynamic normalization.
        """
        if not self.semantic_extractor or not self.taste_profile.get('semantic'):
            return 0.0
        
        try:
            # Get training dramas from the taste profile
            training_dramas = getattr(self, 'training_dramas', [])
            
            if not training_dramas:
                return 0.0
            
            # Extract semantic features for this drama against training dramas
            drama_semantic_features = self.semantic_extractor.extract_single_drama_semantic_features(
                drama, 
                training_dramas
            )
            
            # Calculate similarities for each semantic feature
            similarities = []
            feature_names = [
                'avg_synopsis_similarity', 'max_synopsis_similarity', 'max_review_similarity'
            ]
            
            for i, feature in enumerate(feature_names):
                if feature in self.taste_profile['semantic']:
                    # Normalize and calculate similarity
                    drama_val = drama_semantic_features[i]
                    profile_val = self.taste_profile['semantic'][feature]
                    
                    # Simple similarity based on how close values are
                    max_val = max(abs(drama_val), abs(profile_val))
                    if max_val > 0:
                        similarity = 1 - abs(drama_val - profile_val) / max_val
                    else:
                        similarity = 1.0
                    
                    similarities.append(similarity)
            
            # Calculate average of semantic similarities
            semantic_similarity = np.mean(similarities) if similarities else 0.0
            
            # Return the inverted semantic similarity (without normalization)
            inverted_semantic = 1.0 - semantic_similarity
            
            return inverted_semantic
            
        except Exception as e:
            print(f"    ❌ Error calculating raw semantic similarity for {drama.get('title', 'Unknown')}: {str(e)}")
            return 0.0 