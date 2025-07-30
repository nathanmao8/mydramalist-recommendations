# feature_engineer.py
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from typing import List, Dict, Tuple
from bert_extractor import BertFeatureExtractor

class FeatureEngineer:
    def __init__(self, feature_config: Dict = None):
        # Default configuration if none provided
        self.config = feature_config or {
            'use_bert': True, 
            'use_sentiment': True, 
            'use_tfidf': True,
            'use_position_weights': True, 
            'use_cast': True, 
            'use_crew': True,
            'use_genres': True, 
            'use_tags': True, 
            'tfidf_max_features': 1000,
            'bert_cache': True,
            'use_semantic_similarity': True,
            'semantic_model': 'all-MiniLM-L6-v2'
        }

        # Initialize components based on configuration
        if self.config['use_semantic_similarity']:
            from semantic_similarity import SemanticSimilarityExtractor
            self.semantic_extractor = SemanticSimilarityExtractor(
                model_name=self.config['semantic_model']
        )

        if self.config['use_tfidf']:
            max_features = self.config['tfidf_max_features']
            self.synopsis_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
            self.reviews_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        
        if self.config['use_genres']:
            self.genre_mlb = MultiLabelBinarizer()
        if self.config['use_tags']:
            self.tag_mlb = MultiLabelBinarizer()
        if self.config['use_cast']:
            self.cast_mlb = MultiLabelBinarizer()
        if self.config['use_crew']:
            self.director_mlb = MultiLabelBinarizer()
            self.screenwriter_mlb = MultiLabelBinarizer()
            self.composer_mlb = MultiLabelBinarizer()
        
        if self.config['use_sentiment']:
            self.sentiment_scaler = StandardScaler()
        
        if self.config['use_bert']:
            cache_path = 'bert_embeddings_cache.json' if self.config['bert_cache'] else None
            self.bert_extractor = BertFeatureExtractor(cache_path=cache_path)
        
        # Feature dimension tracking
        self.traditional_feature_dim = None
        self.bert_feature_dim = 768
        
        self.fitted = False

        self.actor_exponential_params = {'decay_factor': 0.7}
        self.genre_sigmoid_params = {'threshold': 2.5, 'steepness': 3}
        self.tag_exponential_params = {'decay_factor': 0.7}

        self._print_active_features()

    def _print_active_features(self):
        """Print which features are enabled for this run."""
        print("\n" + "="*60)
        print("ACTIVE FEATURE CONFIGURATION")
        print("="*60)
        active_features = [key for key, value in self.config.items() if value is True]
        inactive_features = [key for key, value in self.config.items() if value is False]
        
        print("ENABLED:")
        for feature in active_features:
            print(f"  ✓ {feature}")
        
        if inactive_features:
            print("\nDISABLED:")
            for feature in inactive_features:
                print(f"  ✗ {feature}")
        
        print("="*60)

    def _calculate_sigmoid_weights(self, positions: List[int], threshold: float, steepness: float) -> List[float]:
        """Calculate sigmoid-based position weights."""
        weights = []
        for pos in positions:
            weight = 1 / (1 + math.exp(steepness * (pos - threshold)))
            weights.append(weight)
        return weights
    
    def _calculate_exponential_weights(self, positions: List[int], decay_factor: float) -> List[float]:
        """Calculate exponential decay position weights."""
        weights = []
        for pos in positions:
            weight = decay_factor ** (pos - 1)
            weights.append(weight)
        return weights
    
    def _apply_position_weights(self, feature_matrix: np.ndarray, item_lists: List[List], 
                               weighting_func, weighting_params: Dict) -> np.ndarray:
        """Apply position-based weights to a feature matrix."""
        weighted_matrix = feature_matrix.copy().astype(float)
        
        for drama_idx, items in enumerate(item_lists):
            if not items:
                continue
                
            positions = list(range(1, len(items) + 1))
            weights = weighting_func(positions, **weighting_params)
            
            # Find which columns correspond to this drama's items
            for item_idx, item in enumerate(items):
                # Find the column index for this specific item
                for col_idx in range(feature_matrix.shape[1]):
                    if weighted_matrix[drama_idx, col_idx] == 1:  # This item is present
                        # Apply the position weight
                        weighted_matrix[drama_idx, col_idx] = weights[item_idx]
                        break
        
        return weighted_matrix
    
    def create_traditional_features(self, dramas: List[Dict], text_processor, is_training: bool = True):
        """Creates traditional features based on configuration."""
        feature_components = []
        
        # --- Text Features (TF-IDF) ---
        if self.config['use_tfidf']:
            print("Creating TF-IDF features...")
            synopsis_texts = [d.get('synopsis_clean', '') for d in dramas]
            review_texts = [d.get('reviews_combined', '') for d in dramas]
            
            if is_training:
                synopsis_features = self.synopsis_vectorizer.fit_transform(synopsis_texts)
                reviews_features = self.reviews_vectorizer.fit_transform(review_texts)
                self.fitted = True
            else:
                synopsis_features = self.synopsis_vectorizer.transform(synopsis_texts)
                reviews_features = self.reviews_vectorizer.transform(review_texts)
            
            text_features = np.hstack([synopsis_features.toarray(), reviews_features.toarray()])
            feature_components.append(text_features)
        
        # --- Categorical Features ---
        if any([self.config['use_genres'], self.config['use_tags'], self.config['use_cast'], self.config['use_crew']]):
            print("Creating categorical features...")
            categorical_features = self.create_categorical_features(dramas, is_training)
            if categorical_features.size > 0:
                feature_components.append(categorical_features)
        
        # --- Sentiment Features ---
        if self.config['use_sentiment']:
            print("Creating sentiment features...")
            sentiment_features = self.create_sentiment_features(dramas, text_processor, is_training)
            feature_components.append(sentiment_features)
        
        # --- Semantic Similarity Features ---
        if self.config['use_semantic_similarity']:
            print("Creating semantic similarity features...")
            semantic_features = self.semantic_extractor.extract_similarity_features(dramas)
            
            if is_training:
                # Scale semantic features
                if not hasattr(self, 'semantic_scaler'):
                    from sklearn.preprocessing import StandardScaler
                    self.semantic_scaler = StandardScaler()
                semantic_scaled = self.semantic_scaler.fit_transform(semantic_features)
            else:
                semantic_scaled = self.semantic_scaler.transform(semantic_features)
            
            feature_components.append(semantic_scaled)

        # Combine all enabled traditional features
        if feature_components:
            traditional_features = np.hstack(feature_components)
        else:
            # Fallback: create minimal dummy features if nothing is enabled
            print("WARNING: No traditional features enabled, creating dummy features")
            traditional_features = np.ones((len(dramas), 1))
        
        if is_training:
            ratings = np.array([d.get('user_rating', 0) for d in dramas])
            return traditional_features, ratings
        else:
            return traditional_features
    
    def create_bert_features(self, dramas: List[Dict]) -> np.ndarray:
        """Create BERT features only if enabled."""
        if not self.config['use_bert']:
            # Return empty array with correct number of rows
            return np.empty((len(dramas), 0))
        
        print("Creating BERT features...")
        return self.bert_extractor.extract_features(dramas)
    
    def create_all_feature_sets(self, dramas: List[Dict], text_processor, is_training: bool = True):
        """
        A single, efficient method to create all feature sets (traditional and hybrid).
        This is the only method that should be called from main.py to process a dataset.

        Args:
            dramas (List[Dict]): The list of drama data to process.
            text_processor: An instance of the TextProcessor.
            is_training (bool): Controls whether to fit the components or just transform.

        Returns:
            If is_training is True: (X_traditional, X_hybrid, y_ratings)
            If is_training is False: (X_traditional, X_hybrid)
        """
        print(f"Creating all feature sets... (is_training={is_training})")
        
        # 1. Create the base traditional features.
        #    If is_training is True, this will FIT the components.
        if is_training:
            traditional_features, ratings = self.create_traditional_features(dramas, text_processor, is_training=True)
        else:
            traditional_features = self.create_traditional_features(dramas, text_processor, is_training=False)
        
        # 2. Create the BERT features.
        bert_features = self.create_bert_features(dramas)
        
        # 3. Combine them to create the hybrid set.
        hybrid_features = np.hstack([traditional_features, bert_features])
        
        print("Feature creation complete.")
        
        # 4. Return the appropriate tuple based on the mode.
        if is_training:
            return traditional_features, hybrid_features, ratings
        else:
            return traditional_features, hybrid_features

    def create_categorical_features(self, dramas: List[Dict], is_training: bool) -> np.ndarray:
        """Creates categorical features based on configuration."""
        feature_components = []
        
        if self.config['use_genres']:
            genres_list = [d.get('genres', []) for d in dramas]
            genre_features = self._create_weighted_features(
                genres_list, 'genre', 
                'sigmoid' if self.config['use_position_weights'] else 'none',
                self.genre_sigmoid_params if self.config['use_position_weights'] else {},
                is_training
            )
            feature_components.append(genre_features)
        
        if self.config['use_tags']:
            tags_list = [d.get('tags', []) for d in dramas]
            tag_features = self._create_weighted_features(
                tags_list, 'tag',
                'exponential' if self.config['use_position_weights'] else 'none',
                self.tag_exponential_params if self.config['use_position_weights'] else {},
                is_training
            )
            feature_components.append(tag_features)
        
        if self.config['use_cast']:
            cast_list = [d.get('main_cast', []) for d in dramas]
            cast_features = self._create_weighted_features(
                cast_list, 'cast',
                'exponential' if self.config['use_position_weights'] else 'none',
                self.actor_exponential_params if self.config['use_position_weights'] else {},
                is_training
            )
            feature_components.append(cast_features)
        
        if self.config['use_crew']:
            directors_list = [d.get('directors', []) for d in dramas]
            screenwriters_list = [d.get('screenwriters', []) for d in dramas]
            composers_list = [d.get('composers', []) for d in dramas]
            
            director_features = self._create_weighted_features(directors_list, 'director', 'none', {}, is_training)
            screenwriter_features = self._create_weighted_features(screenwriters_list, 'screenwriter', 'none', {}, is_training)
            composer_features = self._create_weighted_features(composers_list, 'composer', 'none', {}, is_training)
            
            feature_components.extend([director_features, screenwriter_features, composer_features])
        
        if feature_components:
            return np.hstack(feature_components)
        else:
            # Return empty array with correct number of rows
            return np.empty((len(dramas), 0))

    def _create_weighted_features(self, item_lists: List[List], feature_type: str, 
                                weight_type: str, params: Dict, is_training: bool) -> np.ndarray:
        """
        Create weighted features for a specific category.
        """
        # Get the appropriate MultiLabelBinarizer
        mlb_map = {
            'genre': self.genre_mlb,
            'tag': self.tag_mlb,
            'cast': self.cast_mlb,
            'director': self.director_mlb,
            'screenwriter': self.screenwriter_mlb,
            'composer': self.composer_mlb
        }
        
        mlb = mlb_map[feature_type]
        
        # Create binary features
        if is_training:
            binary_features = mlb.fit_transform(item_lists)
        else:
            binary_features = mlb.transform(item_lists)
        
        # Apply weighting if specified
        if weight_type == 'none':
            return binary_features.astype(float)
        
        weighted_features = binary_features.astype(float)
        
        for drama_idx, items in enumerate(item_lists):
            if not items:
                continue
            
            # Calculate position weights
            positions = list(range(1, len(items) + 1))
            
            if weight_type == 'sigmoid':
                weights = self._calculate_sigmoid_weights(positions, params['threshold'], params['steepness'])
            elif weight_type == 'exponential':
                weights = self._calculate_exponential_weights(positions, params['decay_factor'])
            else:
                weights = [1.0] * len(items)
            
            # Apply weights to active features for this drama
            for item_idx, item in enumerate(items):
                # Find the column index for this item
                try:
                    col_idx = list(mlb.classes_).index(item)
                    if weighted_features[drama_idx, col_idx] > 0:
                        weighted_features[drama_idx, col_idx] = weights[item_idx]
                except ValueError:
                    # Item not found in classes (shouldn't happen with proper fitting)
                    continue
        
        return weighted_features

    def create_sentiment_features(self, dramas: List[Dict], text_processor, is_training: bool) -> np.ndarray:
        """Create sentiment features only if enabled."""
        if not self.config['use_sentiment']:
            return np.empty((len(dramas), 0))
        
        sentiment_features = []
        for drama in dramas:
            sentiment_dict = drama.get('sentiment_features', {})
            if sentiment_dict:
                sentiment_features.append(list(sentiment_dict.values()))
            else:
                # Fallback: extract sentiment features
                synopsis = drama.get('synopsis_clean', '')
                reviews = drama.get('reviews_combined', '')
                combined_text = synopsis + ' ' + reviews
                sentiment_dict = text_processor.extract_sentiment_features(combined_text)
                sentiment_features.append(list(sentiment_dict.values()))
        
        sentiment_array = np.array(sentiment_features)
        
        # Scale sentiment features
        if is_training:
            sentiment_scaled = self.sentiment_scaler.fit_transform(sentiment_array)
        else:
            sentiment_scaled = self.sentiment_scaler.transform(sentiment_array)
        
        return sentiment_scaled
    
    def get_feature_info(self) -> Dict:
        """Get information about feature dimensions."""
        return {
            'traditional_dim': self.traditional_feature_dim,
            'bert_dim': self.bert_feature_dim,
            'total_hybrid_dim': self.traditional_feature_dim + self.bert_feature_dim if self.traditional_feature_dim else None
        }

    # def get_feature_names(self) -> Tuple[List[str], List[str]]:
    #     """Returns lists of all feature names in the correct order."""
    #     if not self.fitted:
    #         raise RuntimeError("Must fit the feature engineer before getting feature names.")

    #     # Text features
    #     synopsis_features = self.synopsis_vectorizer.get_feature_names_out()
    #     reviews_features = [f"review_{name}" for name in self.reviews_vectorizer.get_feature_names_out()]

    #     # Get feature names from MultiLabelBinarizers
    #     genre_features = [f"genre_{name}" for name in self.genre_mlb.classes_]
    #     tag_features = [f"tag_{name}" for name in self.tag_mlb.classes_]
    #     cast_features = [f"cast_{name}" for name in self.cast_mlb.classes_]
    #     director_features = [f"director_{name}" for name in self.director_mlb.classes_]
    #     screenwriter_features = [f"screenwriter_{name}" for name in self.screenwriter_mlb.classes_]
    #     composer_features = [f"composer_{name}" for name in self.composer_mlb.classes_]
        
    #     # Sentiment features
    #     sentiment_feature_names = [
    #         'vader_positive', 'vader_negative', 'vader_neutral', 'vader_compound',
    #         'textblob_polarity', 'textblob_subjectivity', 'exclamation_ratio',
    #         'question_ratio', 'caps_ratio', 'avg_sentence_length', 'avg_word_length',
    #         'text_length', 'ensemble_sentiment', 'sentiment_agreement',
    #         'sentiment_confidence', 'sentiment_strength'
    #     ]

    #     # Combine for the traditional feature set
    #     traditional_feature_names = (
    #         list(synopsis_features) + reviews_features + genre_features + 
    #         tag_features + cast_features + director_features + 
    #         screenwriter_features + composer_features + sentiment_feature_names
    #     )

    #     # Add semantic similarity feature names
    #     if self.config['use_semantic_similarity']:
    #         semantic_names = self.semantic_extractor.get_feature_names()
    #         traditional_feature_names.extend(semantic_names)

    #     # Combine for the hybrid feature set
    #     bert_feature_names = [f"bert_{i}" for i in range(self.bert_extractor.get_feature_dimension())]
    #     hybrid_feature_names = traditional_feature_names + bert_feature_names

    #     return traditional_feature_names, hybrid_feature_names

    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        """Returns lists of all feature names that exactly match the created features."""
        if not self.fitted:
            raise RuntimeError("Must fit the feature engineer before getting feature names.")

        traditional_feature_names = []

        # TF-IDF features - ONLY if enabled AND vectorizers exist
        if self.config.get('use_tfidf', True) and hasattr(self, 'synopsis_vectorizer'):
            synopsis_features = list(self.synopsis_vectorizer.get_feature_names_out())
            reviews_features = [f"review_{name}" for name in self.reviews_vectorizer.get_feature_names_out()]
            traditional_feature_names.extend(synopsis_features)
            traditional_feature_names.extend(reviews_features)

        # Categorical features - ONLY if the corresponding component was actually created
        categorical_names = []
        
        if (self.config.get('use_genres', True) or 
            self.config.get('use_tags', True) or 
            self.config.get('use_cast', True) or 
            self.config.get('use_crew', True)):
            
            # This mirrors the exact order in create_categorical_features()
            if self.config.get('use_genres', True) and hasattr(self, 'genre_mlb') and hasattr(self.genre_mlb, 'classes_'):
                genre_features = [f"genre_{name}" for name in self.genre_mlb.classes_]
                categorical_names.extend(genre_features)

            if self.config.get('use_tags', True) and hasattr(self, 'tag_mlb') and hasattr(self.tag_mlb, 'classes_'):
                tag_features = [f"tag_{name}" for name in self.tag_mlb.classes_]
                categorical_names.extend(tag_features)

            if self.config.get('use_cast', True) and hasattr(self, 'mlb') and hasattr(self.cast_mlb, 'classes_'):
                cast_features = [f"cast_{name}" for name in self.cast_mlb.classes_]
                categorical_names.extend(cast_features)

            if self.config.get('use_crew', True):
                if hasattr(self, 'director_mlb') and hasattr(self.director_mlb, 'classes_'):
                    director_features = [f"director_{name}" for name in self.director_mlb.classes_]
                    categorical_names.extend(director_features)
                if hasattr(self, 'screenwriter_mlb') and hasattr(self.screenwriter_mlb, 'classes_'):
                    screenwriter_features = [f"screenwriter_{name}" for name in self.screenwriter_mlb.classes_]
                    categorical_names.extend(screenwriter_features)
                if hasattr(self, 'composer_mlb') and hasattr(self.composer_mlb, 'classes_'):
                    composer_features = [f"composer_{name}" for name in self.composer_mlb.classes_]
                    categorical_names.extend(composer_features)
        
        traditional_feature_names.extend(categorical_names)

        # Sentiment features - ONLY if enabled
        if self.config.get('use_sentiment', True):
            sentiment_feature_names = [
                'vader_positive', 'vader_negative', 'vader_neutral', 'vader_compound',
                'textblob_polarity', 'textblob_subjectivity', 'exclamation_ratio',
                'question_ratio', 'caps_ratio', 'avg_sentence_length', 'avg_word_length',
                'text_length', 'ensemble_sentiment', 'sentiment_agreement',
                'sentiment_confidence', 'sentiment_strength'
            ]
            traditional_feature_names.extend(sentiment_feature_names)

        # Semantic similarity features - ONLY if enabled AND extractor exists
        if (self.config.get('use_semantic_similarity', False) and 
            hasattr(self, 'semantic_extractor')):
            semantic_names = self.semantic_extractor.get_feature_names()
            traditional_feature_names.extend(semantic_names)

        # Create hybrid feature names
        hybrid_feature_names = traditional_feature_names.copy()
        if (self.config.get('use_bert', False) and 
            hasattr(self, 'bert_extractor')):
            bert_feature_names = [f"bert_{i}" for i in range(self.bert_extractor.get_feature_dimension())]
            hybrid_feature_names.extend(bert_feature_names)

        return traditional_feature_names, hybrid_feature_names
