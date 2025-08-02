# feature_engineer.py
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from typing import List, Dict, Tuple
from bert_extractor import BertFeatureExtractor

class FeatureEngineer:
    """
    FeatureEngineer handles all feature extraction and engineering for drama recommendation system.
    
    Features:
    - Traditional features: TF-IDF text features, categorical features, sentiment analysis
    - BERT embeddings for semantic understanding
    - Semantic similarity features
    - Position-weighted categorical features (exponential decay, sigmoid)
    - Numerical features with scaling
    - Configurable feature combinations
    
    Usage:
        engineer = FeatureEngineer(feature_config)
        X_traditional, X_hybrid, y = engineer.create_all_feature_sets(dramas, text_processor, is_training=True)
    """
    
    # Default configuration constants
    DEFAULT_CONFIG = {
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
        'semantic_model': 'all-MiniLM-L6-v2',
        'use_numerical_features': True,
        'use_country': True,
        'use_type': True
    }
    
    # Position weighting parameters
    ACTOR_EXPONENTIAL_PARAMS = {'decay_factor': 0.7}
    GENRE_SIGMOID_PARAMS = {'threshold': 2.5, 'steepness': 3}
    TAG_EXPONENTIAL_PARAMS = {'decay_factor': 0.7}
    
    # Feature dimensions
    BERT_FEATURE_DIMENSION = 768
    
    # Numerical feature names
    NUMERICAL_FEATURES = ['year', 'drama_rating', 'watchers']
    
    # Sentiment feature names
    SENTIMENT_FEATURES = [
        'vader_positive', 'vader_negative', 'vader_neutral', 'vader_compound',
        'textblob_polarity', 'textblob_subjectivity', 'exclamation_ratio',
        'question_ratio', 'caps_ratio', 'avg_sentence_length', 'avg_word_length',
        'text_length', 'ensemble_sentiment', 'sentiment_agreement',
        'sentiment_confidence', 'sentiment_strength'
    ]

    def __init__(self, feature_config: Dict = None):
        """
        Initialize FeatureEngineer with configurable parameters.
        
        Parameters
        ----------
        feature_config : Dict, optional
            Configuration dictionary specifying which features to use.
            Keys should match DEFAULT_CONFIG keys. Values will override defaults.
            If None, uses all default configuration values.
            
        Notes
        -----
        The configuration controls which feature types are enabled:
        - Text features: use_tfidf, use_bert, use_semantic_similarity
        - Categorical features: use_genres, use_tags, use_cast, use_crew, use_country, use_type
        - Numerical features: use_numerical_features
        - Other: use_sentiment, use_position_weights
        """
        # Merge user config with defaults
        self.config = {**self.DEFAULT_CONFIG, **(feature_config or {})}
        
        # Initialize tracking variables
        self.traditional_feature_dim = None
        self.bert_feature_dim = self.BERT_FEATURE_DIMENSION
        self.fitted = False
        
        # Store position weighting parameters
        self.actor_exponential_params = self.ACTOR_EXPONENTIAL_PARAMS.copy()
        self.genre_sigmoid_params = self.GENRE_SIGMOID_PARAMS.copy()
        self.tag_exponential_params = self.TAG_EXPONENTIAL_PARAMS.copy()
        
        # Initialize components
        self._initialize_components()
        
        # Print active features
        self._print_active_features()

    def _initialize_components(self):
        """
        Initialize feature extraction components based on configuration.
        
        Notes
        -----
        This method sets up all necessary components for feature extraction including:
        - TF-IDF vectorizers for text processing
        - MultiLabelBinarizers for categorical features
        - StandardScalers for numerical features
        - BERT feature extractor
        - Semantic similarity extractor
        """
        self._initialize_text_processors()
        self._initialize_categorical_encoders()
        self._initialize_scalers()
        self._initialize_bert_extractor()
        self._initialize_semantic_extractor()
    
    def _initialize_text_processors(self):
        """
        Initialize TF-IDF vectorizers if enabled.
        
        Notes
        -----
        Creates separate TfidfVectorizer instances for synopsis and reviews text.
        Both use English stop words and limit features to config['tfidf_max_features'].
        Only initializes if config['use_tfidf'] is True.
        """
        if self.config['use_tfidf']:
            max_features = self.config['tfidf_max_features']
            self.synopsis_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
            self.reviews_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    
    def _initialize_categorical_encoders(self):
        """
        Initialize MultiLabelBinarizers for categorical features.
        
        Notes
        -----
        Creates MultiLabelBinarizer instances for each enabled categorical feature type:
        - Genres, tags, cast (multi-value features)
        - Directors, screenwriters, composers (crew features)
        - Country, type (single-value features)
        Only initializes encoders for features enabled in configuration.
        """
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
        if self.config['use_country']:
            self.country_mlb = MultiLabelBinarizer()
        if self.config['use_type']:
            self.type_mlb = MultiLabelBinarizer()
    
    def _initialize_scalers(self):
        """
        Initialize feature scalers.
        
        Notes
        -----
        Creates StandardScaler instances for:
        - Sentiment features (if use_sentiment enabled)
        - Numerical features (if use_numerical_features enabled)
        Scalers are used to normalize feature values to improve model performance.
        """
        if self.config['use_sentiment']:
            self.sentiment_scaler = StandardScaler()
        if self.config['use_numerical_features']:
            self.numerical_scaler = StandardScaler()
    
    def _initialize_bert_extractor(self):
        """
        Initialize BERT feature extractor if enabled.
        
        Notes
        -----
        Creates BertFeatureExtractor instance with optional caching.
        Cache file is used if config['bert_cache'] is True to speed up
        repeated feature extraction on the same texts.
        Only initializes if config['use_bert'] is True.
        """
        if self.config['use_bert']:
            cache_path = 'bert_embeddings_cache.json' if self.config['bert_cache'] else None
            self.bert_extractor = BertFeatureExtractor(cache_path=cache_path)
    
    def _initialize_semantic_extractor(self):
        """
        Initialize semantic similarity extractor if enabled.
        
        Notes
        -----
        Creates SemanticSimilarityExtractor instance using the model specified
        in config['semantic_model']. Default model is 'all-MiniLM-L6-v2'.
        Only initializes if config['use_semantic_similarity'] is True.
        
        Raises
        ------
        ImportError
            If semantic_similarity module cannot be imported.
        """
        if self.config['use_semantic_similarity']:
            from semantic_similarity import SemanticSimilarityExtractor
            self.semantic_extractor = SemanticSimilarityExtractor(
                model_name=self.config['semantic_model']
            )

    def _print_active_features(self):
        """
        Print which features are enabled for this run.
        
        Notes
        -----
        Displays a formatted list of enabled and disabled features
        for transparency and debugging purposes. Separates boolean
        configuration values into enabled/disabled categories.
        """
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
        """
        Calculate sigmoid-based position weights with error handling.
        
        Parameters
        ----------
        positions : List[int]
            List of position indices (1-based) to calculate weights for.
        threshold : float
            Position threshold for sigmoid function. Positions below this get higher weights.
        steepness : float
            Controls how steep the sigmoid transition is. Higher values create sharper transitions.
            
        Returns
        -------
        List[float]
            List of weights corresponding to input positions, clamped to [0,1].
            Returns uniform weights (all 1.0) if calculation fails.
            
        Notes
        -----
        Uses sigmoid function: weight = 1 / (1 + exp(steepness * (pos - threshold)))
        Earlier positions (lower indices) receive higher weights.
        Weights are clamped to [0,1] range for stability.
        """
        try:
            weights = []
            for pos in positions:
                weight = 1 / (1 + math.exp(steepness * (pos - threshold)))
                weights.append(max(0.0, min(1.0, weight)))  # Clamp to [0,1]
            return weights
        except (OverflowError, ValueError, ZeroDivisionError) as e:
            print(f"Error calculating sigmoid weights: {e}, using uniform weights")
            return [1.0] * len(positions)
    
    def _calculate_exponential_weights(self, positions: List[int], decay_factor: float) -> List[float]:
        """
        Calculate exponential decay position weights with error handling.
        
        Parameters
        ----------
        positions : List[int]
            List of position indices (1-based) to calculate weights for.
        decay_factor : float
            Decay factor between 0 and 1. Controls how quickly weights decrease.
            Values closer to 1 create slower decay, closer to 0 create faster decay.
            
        Returns
        -------
        List[float]
            List of weights corresponding to input positions, clamped to [0,1].
            Returns uniform weights (all 1.0) if calculation fails.
            
        Notes
        -----
        Uses exponential decay: weight = decay_factor^(pos-1)
        Earlier positions (lower indices) receive higher weights.
        Decay factor is validated to be in (0,1] range.
        """
        try:
            if not 0 < decay_factor <= 1:
                print(f"Invalid decay factor {decay_factor}, using 0.7")
                decay_factor = 0.7
            
            weights = []
            for pos in positions:
                weight = decay_factor ** (pos - 1)
                weights.append(max(0.0, min(1.0, weight)))  # Clamp to [0,1]
            return weights
        except (OverflowError, ValueError) as e:
            print(f"Error calculating exponential weights: {e}, using uniform weights")
            return [1.0] * len(positions)

    def _get_weight_function_and_params(self, feature_type: str, use_position_weights: bool) -> Tuple:
        """
        Get the appropriate weight function and parameters for a feature type.
        
        Parameters
        ----------
        feature_type : str
            Type of feature to get weighting for. Options: 'genre', 'tag', 'cast',
            'director', 'screenwriter', 'composer', 'country', 'type'.
        use_position_weights : bool
            Whether position weighting is enabled globally.
            
        Returns
        -------
        Tuple
            (weight_type, params) where weight_type is str ('sigmoid', 'exponential', 'none')
            and params is Dict containing weight function parameters.
            Returns ('none', {}) if position weights disabled or feature type not configured.
            
        Notes
        -----
        Different feature types use different weighting strategies:
        - Genres: sigmoid weighting (important genres tend to be listed first)
        - Tags/Cast: exponential decay (first few items most important)
        - Crew/Country/Type: no weighting (all items equally important)
        """
        if not use_position_weights:
            return None, {}
        
        weight_config = {
            'genre': ('sigmoid', self.genre_sigmoid_params),
            'tag': ('exponential', self.tag_exponential_params),
            'cast': ('exponential', self.actor_exponential_params),
            'director': ('none', {}),
            'screenwriter': ('none', {}),
            'composer': ('none', {}),
            'country': ('none', {}),
            'type': ('none', {})
        }
        
        return weight_config.get(feature_type, ('none', {}))
    
    def _apply_weights_to_features(self, binary_features: np.ndarray, item_lists: List[List], 
                                  weight_type: str, params: Dict, mlb_classes: List) -> np.ndarray:
        """
        Apply position-based weights to binary features efficiently.
        
        Parameters
        ----------
        binary_features : np.ndarray
            Binary feature matrix of shape (n_samples, n_features).
        item_lists : List[List]
            List of item lists for each sample, maintaining original order.
        weight_type : str
            Type of weighting to apply ('sigmoid', 'exponential', 'none').
        params : Dict
            Parameters for the weight function.
        mlb_classes : List
            List of class names from MultiLabelBinarizer, corresponds to feature columns.
            
        Returns
        -------
        np.ndarray
            Weighted feature matrix of same shape as input, with values in [0,1].
            
        Notes
        -----
        Converts binary features to float and applies position-based weights.
        For each sample, finds the column index for each item and applies
        the corresponding position weight. More efficient than iterating over
        all features by using class-to-index mapping.
        """
        if weight_type == 'none' or not item_lists:
            return binary_features.astype(float)
        
        weighted_features = binary_features.astype(float)
        
        # Create class to index mapping for efficiency
        class_to_idx = {cls: idx for idx, cls in enumerate(mlb_classes)}
        
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
            
            # Apply weights efficiently
            for item_idx, item in enumerate(items):
                col_idx = class_to_idx.get(item)
                if col_idx is not None and weighted_features[drama_idx, col_idx] > 0:
                    weighted_features[drama_idx, col_idx] = weights[item_idx]
        
        return weighted_features
    
    def _create_tfidf_features(self, dramas: List[Dict], is_training: bool) -> np.ndarray:
        """
        Create TF-IDF features for synopsis and reviews.
        
        Parameters
        ----------
        dramas : List[Dict]
            List of drama dictionaries containing 'synopsis_clean' and 'reviews_combined' keys.
        is_training : bool
            If True, fits the TF-IDF vectorizers. If False, transforms using fitted vectorizers.
            
        Returns
        -------
        np.ndarray
            TF-IDF feature matrix of shape (n_dramas, n_tfidf_features).
            Returns empty array with shape (n_dramas, 0) if TF-IDF disabled.
            
        Notes
        -----
        Creates separate TF-IDF vectors for synopsis and reviews, then concatenates them.
        Uses pre-configured vectorizers with English stop words and max_features limit.
        Missing text fields default to empty strings.
        """
        if not self.config['use_tfidf']:
            return np.empty((len(dramas), 0))
        
        print("Creating TF-IDF features...")
        synopsis_texts = [d.get('synopsis_clean', '') for d in dramas]
        review_texts = [d.get('reviews_combined', '') for d in dramas]
        
        if is_training:
            synopsis_features = self.synopsis_vectorizer.fit_transform(synopsis_texts)
            reviews_features = self.reviews_vectorizer.fit_transform(review_texts)
        else:
            synopsis_features = self.synopsis_vectorizer.transform(synopsis_texts)
            reviews_features = self.reviews_vectorizer.transform(review_texts)
        
        return np.hstack([synopsis_features.toarray(), reviews_features.toarray()])
    
    def _create_semantic_features(self, dramas: List[Dict], is_training: bool) -> np.ndarray:
        """
        Create semantic similarity features.
        
        Parameters
        ----------
        dramas : List[Dict]
            List of drama dictionaries to extract semantic features from.
        is_training : bool
            If True, fits the semantic feature scaler. If False, transforms using fitted scaler.
            
        Returns
        -------
        np.ndarray
            Scaled semantic similarity feature matrix.
            Returns empty array with shape (n_dramas, 0) if semantic similarity disabled.
            
        Notes
        -----
        Uses semantic similarity extractor to compute features based on text similarity.
        Features are scaled using StandardScaler for consistent range with other features.
        Creates scaler if it doesn't exist during training.
        """
        if not self.config['use_semantic_similarity']:
            return np.empty((len(dramas), 0))
        
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
        
        return semantic_scaled

    def create_traditional_features(self, dramas: List[Dict], text_processor, is_training: bool = True):
        """
        Create traditional features based on configuration.
        
        Parameters
        ----------
        dramas : List[Dict]
            List of drama dictionaries containing feature data.
        text_processor : object
            TextProcessor instance for sentiment feature extraction.
        is_training : bool, default True
            If True, fits feature extractors and returns ratings. If False, only transforms.
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            If is_training=True: (traditional_features, ratings)
            If is_training=False: traditional_features
            Traditional features matrix combining enabled feature types.
            
        Notes
        -----
        Combines all enabled traditional feature types:
        - TF-IDF text features
        - Categorical features (genres, tags, cast, crew, country, type)
        - Sentiment features
        - Semantic similarity features  
        - Numerical features (year, rating, watchers)
        
        If no features are enabled, creates dummy features with warning.
        """
        feature_components = []
        
        # Text Features (TF-IDF)
        tfidf_features = self._create_tfidf_features(dramas, is_training)
        if tfidf_features.size > 0:
            feature_components.append(tfidf_features)
        
        # Categorical Features
        if self._has_categorical_features_enabled():
            print("Creating categorical features...")
            categorical_features = self.create_categorical_features(dramas, is_training)
            if categorical_features.size > 0:
                feature_components.append(categorical_features)
        
        # Sentiment Features
        if self.config['use_sentiment']:
            print("Creating sentiment features...")
            sentiment_features = self.create_sentiment_features(dramas, text_processor, is_training)
            if sentiment_features.size > 0:
                feature_components.append(sentiment_features)
        
        # Semantic Similarity Features
        semantic_features = self._create_semantic_features(dramas, is_training)
        if semantic_features.size > 0:
            feature_components.append(semantic_features)
        
        # Numerical Features
        if self.config['use_numerical_features']:
            print("Creating numerical features...")
            numerical_features = self.create_numerical_features(dramas, is_training)
            if numerical_features.size > 0:
                feature_components.append(numerical_features)

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
    
    def _has_categorical_features_enabled(self) -> bool:
        """
        Check if any categorical features are enabled.
        
        Returns
        -------
        bool
            True if at least one categorical feature type is enabled in configuration.
            
        Notes
        -----
        Checks configuration for: genres, tags, cast, crew, country, type features.
        Used to determine whether to run categorical feature extraction.
        """
        return any([
            self.config['use_genres'], 
            self.config['use_tags'], 
            self.config['use_cast'],
            self.config['use_crew'], 
            self.config['use_country'], 
            self.config['use_type']
        ])
    
    def create_bert_features(self, dramas: List[Dict]) -> np.ndarray:
        """
        Create BERT features only if enabled.
        
        Parameters
        ----------
        dramas : List[Dict]
            List of drama dictionaries containing text data for BERT embedding.
            
        Returns
        -------
        np.ndarray
            BERT feature matrix of shape (n_dramas, 768) if enabled.
            Returns empty array with shape (n_dramas, 0) if BERT disabled.
            Returns zeros with correct shape if extraction fails.
            
        Notes
        -----
        Uses BertFeatureExtractor to generate semantic embeddings from drama text.
        BERT features provide rich semantic representation but are computationally expensive.
        Includes error handling with fallback to zero features if extraction fails.
        """
        if not self.config['use_bert']:
            # Return empty array with correct number of rows
            return np.empty((len(dramas), 0))
        
        try:
            print("Creating BERT features...")
            return self.bert_extractor.extract_features(dramas)
        except Exception as e:
            print(f"Error creating BERT features: {e}")
            # Return zeros with correct shape
            return np.zeros((len(dramas), self.BERT_FEATURE_DIMENSION))
    
    def create_all_feature_sets(self, dramas: List[Dict], text_processor, is_training: bool = True):
        """
        A single, efficient method to create all feature sets (traditional and hybrid).
        This is the only method that should be called from main.py to process a dataset.

        Parameters
        ----------
        dramas : List[Dict]
            The list of drama data to process.
        text_processor : object
            An instance of the TextProcessor.
        is_training : bool, default True
            Controls whether to fit the components or just transform.

        Returns
        -------
        Tuple
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
        
        # 4. Set fitted flag and update dimensions
        if is_training:
            self.fitted = True
            self.traditional_feature_dim = traditional_features.shape[1]
            return traditional_features, hybrid_features, ratings
        else:
            return traditional_features, hybrid_features

    def create_categorical_features(self, dramas: List[Dict], is_training: bool) -> np.ndarray:
        """
        Create categorical features based on configuration.
        
        Parameters
        ----------
        dramas : List[Dict]
            List of drama dictionaries containing categorical data.
        is_training : bool
            If True, fits MultiLabelBinarizers. If False, transforms using fitted binarizers.
            
        Returns
        -------
        np.ndarray
            Categorical feature matrix combining all enabled categorical features.
            Returns empty array with shape (n_dramas, 0) if no categorical features enabled.
            
        Notes
        -----
        Creates binary encoded features for categorical data with optional position weighting:
        - Simple features: genres, tags, cast, country, type
        - Crew features: directors, screenwriters, composers (handled separately)
        
        Position weights are applied based on feature type and configuration.
        Features are concatenated horizontally to form final categorical matrix.
        """
        feature_components = []
        
        # Define feature mappings for cleaner code
        feature_mappings = [
            ('use_genres', 'genres', 'genre', 'genre_mlb'),
            ('use_tags', 'tags', 'tag', 'tag_mlb'),
            ('use_cast', 'main_cast', 'cast', 'cast_mlb'),
            ('use_country', 'country', 'country', 'country_mlb'),
            ('use_type', 'drama_type', 'type', 'type_mlb')
        ]
        
        # Process simple categorical features
        for config_key, data_key, feature_type, mlb_attr in feature_mappings:
            if self.config[config_key]:
                if feature_type in ['country', 'type']:
                    # Handle single-value features
                    data_list = [[d.get(data_key, '')] if d.get(data_key, '') else [] for d in dramas]
                else:
                    # Handle multi-value features
                    data_list = [d.get(data_key, []) for d in dramas]
                
                features = self._create_single_categorical_feature(
                    data_list, feature_type, is_training
                )
                if features.size > 0:
                    feature_components.append(features)
        
        # Process crew features (special case with multiple components)
        if self.config['use_crew']:
            crew_features = self._create_crew_features(dramas, is_training)
            if crew_features.size > 0:
                feature_components.append(crew_features)
        
        if feature_components:
            return np.hstack(feature_components)
        else:
            # Return empty array with correct number of rows
            return np.empty((len(dramas), 0))
    
    def _create_single_categorical_feature(self, data_list: List[List], feature_type: str, 
                                         is_training: bool) -> np.ndarray:
        """
        Create a single categorical feature with appropriate weighting.
        
        Parameters
        ----------
        data_list : List[List]
            List of item lists for each sample, maintaining original order for position weighting.
        feature_type : str
            Type of categorical feature being created (e.g., 'genre', 'tag', 'cast').
        is_training : bool
            If True, fits the MultiLabelBinarizer. If False, transforms using fitted binarizer.
            
        Returns
        -------
        np.ndarray
            Binary or weighted feature matrix for the categorical feature.
            Returns empty array if binarizer not available.
            
        Notes
        -----
        Creates binary encoded features using MultiLabelBinarizer, then applies
        position-based weights if configured. Weight function depends on feature type:
        - Genres: sigmoid weighting
        - Tags/Cast: exponential decay
        - Others: no weighting
        """
        mlb = self._get_multilabel_binarizer(feature_type)
        if mlb is None:
            return np.empty((len(data_list), 0))
        
        # Create binary features
        if is_training:
            binary_features = mlb.fit_transform(data_list)
        else:
            binary_features = mlb.transform(data_list)
        
        # Apply position weights if configured
        weight_type, params = self._get_weight_function_and_params(feature_type, self.config['use_position_weights'])
        
        return self._apply_weights_to_features(
            binary_features, data_list, weight_type, params, mlb.classes_
        )
    
    def _create_crew_features(self, dramas: List[Dict], is_training: bool) -> np.ndarray:
        """
        Create crew-related features (directors, screenwriters, composers).
        
        Parameters
        ----------
        dramas : List[Dict]
            List of drama dictionaries containing crew information.
        is_training : bool
            If True, fits the crew MultiLabelBinarizers. If False, transforms using fitted binarizers.
            
        Returns
        -------
        np.ndarray
            Crew feature matrix combining director, screenwriter, and composer features.
            Returns empty array with shape (n_dramas, 0) if no crew features available.
            
        Notes
        -----
        Creates separate binary features for each crew type, then concatenates them.
        Crew features typically don't use position weighting as all crew members
        are considered equally important.
        """
        crew_components = []
        
        crew_mappings = [
            ('directors', 'director'),
            ('screenwriters', 'screenwriter'),
            ('composers', 'composer')
        ]
        
        for data_key, feature_type in crew_mappings:
            data_list = [d.get(data_key, []) for d in dramas]
            features = self._create_single_categorical_feature(data_list, feature_type, is_training)
            if features.size > 0:
                crew_components.append(features)
        
        if crew_components:
            return np.hstack(crew_components)
        else:
            return np.empty((len(dramas), 0))
    
    def _get_multilabel_binarizer(self, feature_type: str):
        """
        Get the MultiLabelBinarizer for a specific feature type.
        
        Parameters
        ----------
        feature_type : str
            Type of feature to get binarizer for. Options: 'genre', 'tag', 'cast',
            'director', 'screenwriter', 'composer', 'country', 'type'.
            
        Returns
        -------
        MultiLabelBinarizer or None
            The corresponding MultiLabelBinarizer instance if available and feature enabled.
            Returns None if feature type not recognized or not enabled.
            
        Notes
        -----
        Provides centralized access to MultiLabelBinarizer instances.
        Uses getattr with None default to safely handle missing attributes
        when features are disabled.
        """
        mlb_mapping = {
            'genre': getattr(self, 'genre_mlb', None),
            'tag': getattr(self, 'tag_mlb', None),
            'cast': getattr(self, 'cast_mlb', None),
            'director': getattr(self, 'director_mlb', None),
            'screenwriter': getattr(self, 'screenwriter_mlb', None),
            'composer': getattr(self, 'composer_mlb', None),
            'country': getattr(self, 'country_mlb', None),
            'type': getattr(self, 'type_mlb', None)
        }
        
        return mlb_mapping.get(feature_type)

    def create_sentiment_features(self, dramas: List[Dict], text_processor, is_training: bool) -> np.ndarray:
        """
        Create sentiment features only if enabled.
        
        Parameters
        ----------
        dramas : List[Dict]
            List of drama dictionaries containing text data for sentiment analysis.
        text_processor : object
            TextProcessor instance with extract_sentiment_features method.
        is_training : bool
            If True, fits the sentiment feature scaler. If False, transforms using fitted scaler.
            
        Returns
        -------
        np.ndarray
            Scaled sentiment feature matrix of shape (n_dramas, n_sentiment_features).
            Returns empty array with shape (n_dramas, 0) if sentiment features disabled.
            Returns zeros with correct shape if extraction fails.
            
        Notes
        -----
        Extracts sentiment features from combined synopsis and reviews text.
        Uses pre-computed sentiment features if available in drama data.
        Falls back to text processor extraction if not available.
        Features are scaled using StandardScaler for consistent range.
        """
        if not self.config['use_sentiment']:
            return np.empty((len(dramas), 0))
        
        try:
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
        except Exception as e:
            print(f"Error creating sentiment features: {e}")
            # Return zeros with correct shape
            return np.zeros((len(dramas), len(self.SENTIMENT_FEATURES)))
    
    def create_numerical_features(self, dramas: List[Dict], is_training: bool) -> np.ndarray:
        """
        Create numerical features (year, drama_rating, watchers) only if enabled.
        
        Parameters
        ----------
        dramas : List[Dict]
            List of drama dictionaries containing numerical data.
        is_training : bool
            If True, fits the numerical feature scaler. If False, transforms using fitted scaler.
            
        Returns
        -------
        np.ndarray
            Scaled numerical feature matrix of shape (n_dramas, 3).
            Returns empty array with shape (n_dramas, 0) if numerical features disabled.
            Returns zeros with correct shape if extraction fails.
            
        Notes
        -----
        Extracts three numerical features:
        - year: Release year of the drama
        - drama_rating: Overall rating/score of the drama
        - watchers: Number of people who watched the drama
        
        Missing values default to 0.0. Features are scaled using StandardScaler.
        """
        if not self.config['use_numerical_features']:
            return np.empty((len(dramas), 0))
        
        try:
            numerical_features = []
            for drama in dramas:
                features = [
                    drama.get('year', 0.0),
                    drama.get('drama_rating', 0.0),
                    drama.get('watchers', 0.0)
                ]
                numerical_features.append(features)
            
            numerical_array = np.array(numerical_features)
            
            # Scale numerical features
            if is_training:
                numerical_scaled = self.numerical_scaler.fit_transform(numerical_array)
            else:
                numerical_scaled = self.numerical_scaler.transform(numerical_array)
            
            return numerical_scaled
        except Exception as e:
            print(f"Error creating numerical features: {e}")
            # Return zeros with correct shape
            return np.zeros((len(dramas), len(self.NUMERICAL_FEATURES)))
    

    def get_feature_info(self) -> Dict:
        """
        Get information about feature dimensions.
        
        Returns
        -------
        Dict
            Dictionary containing feature dimension information:
            - 'traditional_dim': Number of traditional features
            - 'bert_dim': Number of BERT features (768 if enabled, else None)
            - 'total_hybrid_dim': Total features in hybrid set (traditional + BERT)
            
        Notes
        -----
        Must be called after fitting the feature engineer to get accurate dimensions.
        Used for model initialization and debugging feature extraction.
        """
        return {
            'traditional_dim': self.traditional_feature_dim,
            'bert_dim': self.bert_feature_dim,
            'total_hybrid_dim': self.traditional_feature_dim + self.bert_feature_dim if self.traditional_feature_dim else None
        }

    def _get_tfidf_feature_names(self) -> List[str]:
        """
        Get TF-IDF feature names.
        
        Returns
        -------
        List[str]
            List of TF-IDF feature names combining synopsis and review features.
            Returns empty list if TF-IDF disabled or vectorizers not fitted.
            
        Notes
        -----
        Combines feature names from both synopsis and reviews vectorizers.
        Review features are prefixed with 'review_' to distinguish them.
        Requires vectorizers to be fitted before calling.
        """
        if not (self.config.get('use_tfidf', True) and hasattr(self, 'synopsis_vectorizer')):
            return []
        
        synopsis_features = list(self.synopsis_vectorizer.get_feature_names_out())
        reviews_features = [f"review_{name}" for name in self.reviews_vectorizer.get_feature_names_out()]
        return synopsis_features + reviews_features
    
    def _get_categorical_feature_names(self) -> List[str]:
        """
        Get categorical feature names in the order they are created.
        
        Returns
        -------
        List[str]
            List of categorical feature names in creation order.
            Each feature name is prefixed with its category type.
            
        Notes
        -----
        Returns feature names for all enabled categorical features:
        - Genres: 'genre_<name>'
        - Tags: 'tag_<name>'  
        - Cast: 'cast_<name>'
        - Country: 'country_<name>'
        - Type: 'type_<name>'
        - Crew: 'director_<name>', 'screenwriter_<name>', 'composer_<name>'
        
        Order matches the feature creation order in create_categorical_features.
        """
        categorical_names = []
        
        # Define feature mappings matching create_categorical_features
        feature_mappings = [
            ('use_genres', 'genre_mlb', 'genre'),
            ('use_tags', 'tag_mlb', 'tag'),
            ('use_cast', 'cast_mlb', 'cast'),
            ('use_country', 'country_mlb', 'country'),
            ('use_type', 'type_mlb', 'type')
        ]
        
        # Process simple categorical features
        for config_key, mlb_attr, prefix in feature_mappings:
            if self.config.get(config_key, True):
                mlb = getattr(self, mlb_attr, None)
                if mlb and hasattr(mlb, 'classes_'):
                    feature_names = [f"{prefix}_{name}" for name in mlb.classes_]
                    categorical_names.extend(feature_names)
        
        # Process crew features
        if self.config.get('use_crew', True):
            crew_names = self._get_crew_feature_names()
            categorical_names.extend(crew_names)
        
        return categorical_names
    
    def _get_crew_feature_names(self) -> List[str]:
        """
        Get crew feature names.
        
        Returns
        -------
        List[str]
            List of crew feature names with appropriate prefixes:
            'director_<name>', 'screenwriter_<name>', 'composer_<name>'.
            
        Notes
        -----
        Returns names for all fitted crew MultiLabelBinarizers.
        Order matches the creation order in _create_crew_features method.
        Only includes features for binarizers that have been fitted.
        """
        crew_names = []
        crew_mappings = [
            ('director_mlb', 'director'),
            ('screenwriter_mlb', 'screenwriter'),
            ('composer_mlb', 'composer')
        ]
        
        for mlb_attr, prefix in crew_mappings:
            mlb = getattr(self, mlb_attr, None)
            if mlb and hasattr(mlb, 'classes_'):
                feature_names = [f"{prefix}_{name}" for name in mlb.classes_]
                crew_names.extend(feature_names)
        
        return crew_names
    
    def _get_semantic_feature_names(self) -> List[str]:
        """
        Get semantic similarity feature names.
        
        Returns
        -------
        List[str]
            List of semantic similarity feature names.
            Returns empty list if semantic similarity disabled or extractor not available.
            
        Notes
        -----
        Delegates to semantic extractor's get_feature_names method.
        Feature names depend on the semantic similarity implementation.
        """
        if not (self.config.get('use_semantic_similarity', False) and hasattr(self, 'semantic_extractor')):
            return []
        return self.semantic_extractor.get_feature_names()
    
    def _get_bert_feature_names(self) -> List[str]:
        """
        Get BERT feature names.
        
        Returns
        -------
        List[str]
            List of BERT feature names in format 'bert_0', 'bert_1', ..., 'bert_767'.
            Returns empty list if BERT disabled or extractor not available.
            
        Notes
        -----
        Creates sequential names for each BERT embedding dimension.
        BERT features are dense 768-dimensional vectors from transformer model.
        """
        if not (self.config.get('use_bert', False) and hasattr(self, 'bert_extractor')):
            return []
        return [f"bert_{i}" for i in range(self.bert_extractor.get_feature_dimension())]

    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        """
        Return lists of all feature names that exactly match the created features.
        
        Returns
        -------
        Tuple[List[str], List[str]]
            (traditional_feature_names, hybrid_feature_names) where:
            - traditional_feature_names: Names for all traditional features
            - hybrid_feature_names: Names for traditional + BERT features
            
        Raises
        ------
        RuntimeError
            If feature engineer has not been fitted yet.
            
        Notes
        -----
        Feature names are returned in the exact order they appear in feature matrices.
        This order matches the concatenation order in create_all_feature_sets:
        1. TF-IDF features (synopsis + reviews)
        2. Categorical features (genres, tags, cast, crew, country, type)
        3. Sentiment features
        4. Semantic similarity features
        5. Numerical features
        6. BERT features (hybrid only)
        """
        if not self.fitted:
            raise RuntimeError("Must fit the feature engineer before getting feature names.")

        traditional_feature_names = []

        # TF-IDF features
        traditional_feature_names.extend(self._get_tfidf_feature_names())
        
        # Categorical features
        if self._has_categorical_features_enabled():
            traditional_feature_names.extend(self._get_categorical_feature_names())

        # Sentiment features
        if self.config.get('use_sentiment', True):
            traditional_feature_names.extend(self.SENTIMENT_FEATURES)

        # Semantic similarity features
        traditional_feature_names.extend(self._get_semantic_feature_names())

        # Numerical features
        if self.config.get('use_numerical_features', True):
            traditional_feature_names.extend(self.NUMERICAL_FEATURES)

        # Create hybrid feature names
        hybrid_feature_names = traditional_feature_names.copy()
        hybrid_feature_names.extend(self._get_bert_feature_names())

        return traditional_feature_names, hybrid_feature_names

    def get_configuration(self) -> Dict[str, any]:
        """
        Get current configuration settings for debugging and monitoring.
        
        Returns
        -------
        Dict[str, any]
            Dictionary containing complete configuration information:
            - 'config': Current feature configuration settings
            - 'fitted': Whether feature engineer has been fitted
            - 'traditional_feature_dim': Number of traditional features
            - 'bert_feature_dim': Number of BERT features
            - 'position_weights': Position weighting parameters
            - 'available_components': Status of initialized components
            
        Notes
        -----
        Useful for debugging feature engineering issues and monitoring
        which components are properly initialized. Returns deep copy
        of configuration to prevent accidental modification.
        """
        return {
            'config': self.config.copy(),
            'fitted': self.fitted,
            'traditional_feature_dim': self.traditional_feature_dim,
            'bert_feature_dim': self.bert_feature_dim,
            'position_weights': {
                'actor_exponential_params': self.actor_exponential_params,
                'genre_sigmoid_params': self.genre_sigmoid_params,
                'tag_exponential_params': self.tag_exponential_params
            },
            'available_components': self._get_available_components()
        }
    
    def _get_available_components(self) -> Dict[str, bool]:
        """
        Check which components are available and properly initialized.
        
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping component names to availability status.
            True indicates component is initialized and available for use.
            
        Notes
        -----
        Checks for presence of all possible feature engineering components:
        - Text processors: synopsis_vectorizer, reviews_vectorizer
        - Categorical encoders: genre_mlb, tag_mlb, cast_mlb, crew MLBs, etc.
        - Scalers: sentiment_scaler, numerical_scaler, semantic_scaler
        - Advanced extractors: bert_extractor, semantic_extractor
        
        Used for debugging and validation of component initialization.
        """
        return {
            'synopsis_vectorizer': hasattr(self, 'synopsis_vectorizer'),
            'reviews_vectorizer': hasattr(self, 'reviews_vectorizer'),
            'genre_mlb': hasattr(self, 'genre_mlb'),
            'tag_mlb': hasattr(self, 'tag_mlb'),
            'cast_mlb': hasattr(self, 'cast_mlb'),
            'director_mlb': hasattr(self, 'director_mlb'),
            'screenwriter_mlb': hasattr(self, 'screenwriter_mlb'),
            'composer_mlb': hasattr(self, 'composer_mlb'),
            'country_mlb': hasattr(self, 'country_mlb'),
            'type_mlb': hasattr(self, 'type_mlb'),
            'sentiment_scaler': hasattr(self, 'sentiment_scaler'),
            'numerical_scaler': hasattr(self, 'numerical_scaler'),
            'semantic_scaler': hasattr(self, 'semantic_scaler'),
            'bert_extractor': hasattr(self, 'bert_extractor'),
            'semantic_extractor': hasattr(self, 'semantic_extractor')
        }

    def get_feature_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive feature engineering statistics.
        
        Returns
        -------
        Dict[str, any]
            Dictionary containing detailed feature statistics:
            - 'feature_counts': Counts for different feature categories
            - 'feature_breakdown': Detailed breakdown by feature type
            - 'enabled_features': List of enabled feature configuration keys
            - 'disabled_features': List of disabled feature configuration keys
            - 'error': Error message if statistics cannot be generated
            
        Notes
        -----
        Requires feature engineer to be fitted before generating statistics.
        Provides comprehensive overview of feature composition for analysis
        and debugging. Includes error handling for robustness.
        """
        if not self.fitted:
            return {'error': 'Feature engineer not fitted yet'}
        
        try:
            traditional_names, hybrid_names = self.get_feature_names()
            
            stats = {
                'feature_counts': {
                    'traditional_features': len(traditional_names),
                    'bert_features': len(self._get_bert_feature_names()),
                    'total_hybrid_features': len(hybrid_names)
                },
                'feature_breakdown': {
                    'tfidf_features': len(self._get_tfidf_feature_names()),
                    'categorical_features': len(self._get_categorical_feature_names()),
                    'sentiment_features': len(self.SENTIMENT_FEATURES) if self.config['use_sentiment'] else 0,
                    'semantic_features': len(self._get_semantic_feature_names()),
                    'numerical_features': len(self.NUMERICAL_FEATURES) if self.config['use_numerical_features'] else 0
                },
                'enabled_features': [key for key, value in self.config.items() if key.startswith('use_') and value],
                'disabled_features': [key for key, value in self.config.items() if key.startswith('use_') and not value]
            }
            
            return stats
        except Exception as e:
            return {'error': f'Error generating statistics: {e}'}

    def validate_configuration(self) -> Dict[str, any]:
        """
        Validate the current configuration and report any issues.
        
        Returns
        -------
        Dict[str, any]
            Dictionary containing validation results:
            - 'is_valid': Overall validation status (always True, warnings don't invalidate)
            - 'warnings': List of configuration warnings
            - 'errors': List of configuration errors (currently unused)
            - 'recommendations': List of optimization recommendations
            
        Notes
        -----
        Performs comprehensive configuration validation including:
        - Checking for minimum feature requirements
        - Identifying potentially suboptimal settings
        - Recommending configuration improvements
        - Detecting conflicting or insufficient feature combinations
        
        Validation is non-blocking - warnings and recommendations don't prevent usage.
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check for conflicting configurations
        if not self.config['use_bert'] and not any([
            self.config['use_tfidf'], self.config['use_sentiment'], 
            self.config['use_semantic_similarity']
        ]):
            validation_results['warnings'].append(
                "No text-based features enabled - consider enabling TF-IDF, BERT, or sentiment analysis"
            )
        
        # Check for minimum feature requirements
        enabled_features = sum(1 for key, value in self.config.items() if key.startswith('use_') and value)
        if enabled_features < 2:
            validation_results['warnings'].append(
                f"Only {enabled_features} feature type(s) enabled - consider enabling more for better performance"
            )
        
        # Check TF-IDF configuration
        if self.config['use_tfidf'] and self.config['tfidf_max_features'] < 100:
            validation_results['recommendations'].append(
                "TF-IDF max_features is quite low - consider increasing for richer text representation"
            )
        
        # Check position weights configuration
        if not self.config['use_position_weights'] and any([
            self.config['use_cast'], self.config['use_tags'], self.config['use_genres']
        ]):
            validation_results['recommendations'].append(
                "Position weights disabled - consider enabling for better handling of ordered categorical features"
            )
        
        return validation_results
