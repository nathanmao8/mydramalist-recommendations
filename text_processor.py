import numpy as np
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class TextProcessor:
    """
    TextProcessor handles all text cleaning, preprocessing, and sentiment analysis.
    
    Features:
    - Comprehensive text cleaning (spoiler removal, non-English filtering)
    - Multi-method sentiment analysis (VADER + TextBlob)
    - Linguistic feature extraction
    - Review processing with helpfulness weighting
    - Configurable ensemble sentiment scoring
    """
    
    # Text cleaning patterns and configuration
    SPOILER_PATTERNS = [
        r'This review may contain spoilers',
        r'Spoiler Alert',
        r'Warning:.*spoiler',
        r'Disclaimer:.*',
    ]
    
    # Sentiment analysis configuration
    VADER_WEIGHT = 0.6  # Weight for VADER in ensemble scoring
    TEXTBLOB_WEIGHT = 0.4  # Weight for TextBlob in ensemble scoring
    MIN_TEXT_LENGTH = 5  # Minimum text length for sentiment analysis
    
    # Review processing configuration
    MAX_REVIEW_REPETITIONS = 3  # Maximum times a review can be repeated based on helpfulness
    HELPFULNESS_LOG_BASE = 1  # Base for logarithmic helpfulness scaling
    
    def __init__(self, vader_weight: float = None, textblob_weight: float = None, 
                 min_text_length: int = None, max_review_repetitions: int = None):
        """
        Initialize TextProcessor with configurable parameters.
        
        Parameters
        ----------
        vader_weight : float, optional
            Weight for VADER sentiment scores in ensemble. Must be between 0 and 1.
            Default is 0.6.
        textblob_weight : float, optional
            Weight for TextBlob sentiment scores in ensemble. Must be between 0 and 1.
            Default is 0.4.
        min_text_length : int, optional
            Minimum text length required for sentiment analysis. Default is 5.
        max_review_repetitions : int, optional
            Maximum number of times a review can be repeated based on helpfulness.
            Default is 3.
        
        Notes
        -----
        If both vader_weight and textblob_weight are provided, they will be normalized
        to sum to 1.0 automatically.
        """
        self.english_pattern = re.compile(r'[a-zA-Z\s\.,!?;:\-\'\"()]+')
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Allow customization of ensemble weights
        self.vader_weight = vader_weight if vader_weight is not None else self.VADER_WEIGHT
        self.textblob_weight = textblob_weight if textblob_weight is not None else self.TEXTBLOB_WEIGHT
        
        # Ensure weights sum to 1.0
        total_weight = self.vader_weight + self.textblob_weight
        if total_weight != 1.0:
            self.vader_weight /= total_weight
            self.textblob_weight /= total_weight
            
        # Allow customization of processing parameters
        self.min_text_length = min_text_length if min_text_length is not None else self.MIN_TEXT_LENGTH
        self.max_review_repetitions = max_review_repetitions if max_review_repetitions is not None else self.MAX_REVIEW_REPETITIONS

    def get_configuration(self) -> Dict[str, any]:
        """
        Get current configuration settings for debugging and monitoring.
        
        Returns
        -------
        dict
            Dictionary containing current configuration parameters including:
            - vader_weight : float
                Current weight for VADER sentiment analysis
            - textblob_weight : float
                Current weight for TextBlob sentiment analysis
            - min_text_length : int
                Minimum text length for processing
            - max_review_repetitions : int
                Maximum review repetitions for helpfulness weighting
            - helpfulness_log_base : int
                Base for logarithmic helpfulness scaling
            - spoiler_patterns_count : int
                Number of spoiler patterns configured
        """
        return {
            'vader_weight': self.vader_weight,
            'textblob_weight': self.textblob_weight,
            'min_text_length': self.min_text_length,
            'max_review_repetitions': self.max_review_repetitions,
            'helpfulness_log_base': self.HELPFULNESS_LOG_BASE,
            'spoiler_patterns_count': len(self.SPOILER_PATTERNS)
        }

    def update_configuration(self, **kwargs):
        """
        Update configuration parameters at runtime.
        
        Parameters
        ----------
        vader_weight : float, optional
            New weight for VADER sentiment analysis (0-1)
        textblob_weight : float, optional
            New weight for TextBlob sentiment analysis (0-1)
        min_text_length : int, optional
            New minimum text length for processing
        max_review_repetitions : int, optional
            New maximum review repetitions for helpfulness weighting
            
        Notes
        -----
        If both vader_weight and textblob_weight are updated, they will be
        automatically normalized to sum to 1.0.
        """
        if 'vader_weight' in kwargs or 'textblob_weight' in kwargs:
            self.vader_weight = kwargs.get('vader_weight', self.vader_weight)
            self.textblob_weight = kwargs.get('textblob_weight', self.textblob_weight)
            
            # Ensure weights sum to 1.0
            total_weight = self.vader_weight + self.textblob_weight
            if total_weight != 1.0:
                self.vader_weight /= total_weight
                self.textblob_weight /= total_weight
                
        if 'min_text_length' in kwargs:
            self.min_text_length = kwargs['min_text_length']
            
        if 'max_review_repetitions' in kwargs:
            self.max_review_repetitions = kwargs['max_review_repetitions']
    
    def remove_non_english(self, text: str) -> str:
        """
        Remove non-English characters from text using regex pattern matching.
        
        Parameters
        ----------
        text : str
            Input text that may contain non-English characters
            
        Returns
        -------
        str
            Text with only English characters, spaces, and common punctuation
            preserved. Returns empty string if input is None or empty.
            
        Notes
        -----
        Preserves letters (a-z, A-Z), spaces, and punctuation: .,!?;:-'"()
        All other characters including emojis, non-Latin scripts, and special
        symbols are removed.
        """
        try:
            if not text:
                return ""
            
            # Extract English text
            english_parts = self.english_pattern.findall(text)
            return ' '.join(english_parts).strip()
        except Exception as e:
            print(f"Error removing non-English characters: {e}")
            return text  # Return original text if processing fails
    
    def remove_spoiler_warnings(self, text: str) -> str:
        """
        Remove spoiler warnings and disclaimers from text.
        
        Parameters
        ----------
        text : str
            Input text that may contain spoiler warnings or disclaimers
            
        Returns
        -------
        str
            Text with spoiler warnings removed and whitespace stripped.
            Returns original text if processing fails.
            
        Notes
        -----
        Removes patterns defined in SPOILER_PATTERNS class variable using
        case-insensitive matching. Common patterns include:
        - "This review may contain spoilers"
        - "Spoiler Alert"
        - "Warning:.*spoiler"
        - "Disclaimer:.*"
        """
        try:
            for pattern in self.SPOILER_PATTERNS:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
            return text.strip()
        except Exception as e:
            print(f"Error removing spoiler warnings: {e}")
            return text  # Return original text if processing fails
    
    def clean_text(self, text: str) -> str:
        """
        Comprehensive text cleaning with error handling.
        
        Parameters
        ----------
        text : str
            Raw input text to be cleaned
            
        Returns
        -------
        str
            Cleaned text with spoiler warnings removed, non-English characters
            filtered, special characters normalized, and extra whitespace collapsed.
            Returns empty string for None/empty input.
            
        Notes
        -----
        Cleaning pipeline:
        1. Remove spoiler warnings
        2. Filter non-English characters 
        3. Normalize special characters to spaces
        4. Collapse multiple whitespace to single spaces
        5. Strip leading/trailing whitespace
        """
        try:
            if not text:
                return ""
            
            # Remove spoiler warnings
            text = self.remove_spoiler_warnings(text)
            
            # Remove non-English text
            text = self.remove_non_english(text)
            
            # Handle special characters and formatting
            text = re.sub(r'[^\w\s\.,!?;:\-\'\"()]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
        except Exception as e:
            print(f"Error during text cleaning: {e}")
            return str(text).strip()  # Return basic cleaned version
    
    def extract_sentiment_features(self, text: str) -> Dict:
        """
        Extract comprehensive sentiment features using VADER and TextBlob.
        
        Parameters
        ----------
        text : str
            Input text for sentiment analysis
            
        Returns
        -------
        dict
            Dictionary containing sentiment features:
            - VADER features: vader_positive, vader_negative, vader_neutral, vader_compound
            - TextBlob features: textblob_polarity, textblob_subjectivity  
            - Linguistic features: exclamation_ratio, question_ratio, caps_ratio,
              avg_sentence_length, avg_word_length, text_length
            - Ensemble features: ensemble_sentiment, sentiment_agreement,
              sentiment_confidence, sentiment_strength
              
        Notes
        -----
        Returns empty features (zeros/neutral values) if text is shorter than
        min_text_length or if any processing step fails. VADER compound scores
        and TextBlob polarity are combined using configurable weights.
        """
        
        if not text or len(text.strip()) < self.min_text_length:
            return self._get_empty_sentiment_features()
        
        # VADER Sentiment Analysis
        vader_features = self._extract_vader_features(text)
        
        # TextBlob Sentiment Analysis
        textblob_features = self._extract_textblob_features(text)
        
        # Traditional linguistic features (without word lists)
        linguistic_features = self._extract_linguistic_features(text)
        
        # Ensemble sentiment scores
        ensemble_features = self._calculate_ensemble_features(vader_features, textblob_features)
        
        # Combine all features
        return {
            **vader_features,
            **textblob_features,
            **linguistic_features,
            **ensemble_features
        }
    
    def _extract_vader_features(self, text: str) -> Dict:
        """
        Extract VADER sentiment scores with error handling.
        
        Parameters
        ----------
        text : str
            Input text for VADER sentiment analysis
            
        Returns
        -------
        dict
            Dictionary containing VADER sentiment scores:
            - vader_positive : float (0-1)
                Proportion of positive sentiment
            - vader_negative : float (0-1) 
                Proportion of negative sentiment
            - vader_neutral : float (0-1)
                Proportion of neutral sentiment
            - vader_compound : float (-1 to 1)
                Overall sentiment score (negative to positive)
                
        Notes
        -----
        Returns neutral scores (pos=0, neg=0, neu=1, compound=0) if VADER
        analysis fails for any reason.
        """
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            return {
                'vader_positive': scores['pos'],
                'vader_negative': scores['neg'],
                'vader_neutral': scores['neu'],
                'vader_compound': scores['compound']
            }
        except Exception as e:
            print(f"VADER analysis error: {e}")
            return {
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 1.0,
                'vader_compound': 0.0
            }
    
    def _extract_textblob_features(self, text: str) -> Dict:
        """
        Extract TextBlob sentiment scores with error handling.
        
        Parameters
        ----------
        text : str
            Input text for TextBlob sentiment analysis
            
        Returns
        -------
        dict
            Dictionary containing TextBlob sentiment scores:
            - textblob_polarity : float (-1 to 1)
                Sentiment polarity from negative to positive
            - textblob_subjectivity : float (0 to 1)
                Subjectivity from objective to subjective
                
        Notes
        -----
        Returns neutral scores (polarity=0, subjectivity=0) if TextBlob
        analysis fails for any reason.
        """
        
        try:
            blob = TextBlob(text)
            
            return {
                'textblob_polarity': blob.sentiment.polarity,      # -1 to 1
                'textblob_subjectivity': blob.sentiment.subjectivity  # 0 to 1
            }
        except Exception as e:
            print(f"TextBlob error: {e}")
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0
            }
    
    def _extract_linguistic_features(self, text: str) -> Dict:
        """
        Extract linguistic features with comprehensive error handling.
        
        Parameters
        ----------
        text : str
            Input text for linguistic feature extraction
            
        Returns
        -------
        dict
            Dictionary containing linguistic features:
            - exclamation_ratio : float
                Ratio of exclamation marks to total characters
            - question_ratio : float
                Ratio of question marks to total characters  
            - caps_ratio : float
                Ratio of uppercase letters to total characters
            - avg_sentence_length : float
                Average number of words per sentence
            - avg_word_length : float
                Average number of characters per word
            - text_length : int
                Total number of words in text
                
        Notes
        -----
        Returns zero values for all features if linguistic analysis fails.
        Sentences are split on periods, words on whitespace.
        """
        try:
            # Punctuation-based sentiment indicators
            exclamation_count = text.count('!')
            question_count = text.count('?')
            caps_count = sum(1 for c in text if c.isupper() and c.isalpha())
            
            # Text structure features
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            words = text.split()
            
            avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            total_chars = len(text)
            total_words = len(words)
            
            return {
                'exclamation_ratio': exclamation_count / max(1, total_chars),
                'question_ratio': question_count / max(1, total_chars),
                'caps_ratio': caps_count / max(1, total_chars),
                'avg_sentence_length': avg_sentence_length,
                'avg_word_length': avg_word_length,
                'text_length': total_words
            }
        except Exception as e:
            print(f"Linguistic features extraction error: {e}")
            return {
                'exclamation_ratio': 0.0,
                'question_ratio': 0.0,
                'caps_ratio': 0.0,
                'avg_sentence_length': 0.0,
                'avg_word_length': 0.0,
                'text_length': 0
            }
    
    def _calculate_ensemble_features(self, vader_features: Dict, textblob_features: Dict) -> Dict:
        """
        Calculate ensemble sentiment scores with error handling.
        
        Parameters
        ----------
        vader_features : dict
            Dictionary containing VADER sentiment scores
        textblob_features : dict
            Dictionary containing TextBlob sentiment scores
            
        Returns
        -------
        dict
            Dictionary containing ensemble sentiment features:
            - ensemble_sentiment : float (-1 to 1)
                Weighted combination of VADER and TextBlob sentiment
            - sentiment_agreement : float (0 to 1)
                Measure of agreement between VADER and TextBlob
            - sentiment_confidence : float (0 to 1)
                Confidence in sentiment prediction based on agreement and strength
            - sentiment_strength : float (0 to 1)
                Absolute strength of the sentiment signal
                
        Notes
        -----
        Uses configurable weights for VADER and TextBlob scores. Agreement is
        calculated as 1 minus the absolute difference between normalized scores.
        Returns zero values if calculation fails.
        """
        try:
            # Normalize scores to same scale (-1 to 1)
            vader_normalized = vader_features['vader_compound']
            textblob_normalized = textblob_features['textblob_polarity']
            
            # Weighted ensemble
            weighted_sentiment = (vader_normalized * self.vader_weight) + (textblob_normalized * self.textblob_weight)
            
            # Agreement measure (how much the methods agree)
            agreement = 1 - abs(vader_normalized - textblob_normalized) / 2
            
            # Confidence based on agreement and strength of signal
            confidence = agreement * (abs(weighted_sentiment) + 0.1)
            
            return {
                'ensemble_sentiment': weighted_sentiment,
                'sentiment_agreement': agreement,
                'sentiment_confidence': min(confidence, 1.0),
                'sentiment_strength': abs(weighted_sentiment)
            }
        except Exception as e:
            print(f"Ensemble features calculation error: {e}")
            return {
                'ensemble_sentiment': 0.0,
                'sentiment_agreement': 0.0,
                'sentiment_confidence': 0.0,
                'sentiment_strength': 0.0
            }
    
    def _get_empty_sentiment_features(self) -> Dict:
        """
        Return empty sentiment features for missing/invalid text.
        
        Returns
        -------
        dict
            Dictionary with all sentiment features set to neutral/zero values:
            - VADER features set to neutral (neu=1.0, others=0.0)
            - TextBlob features set to zero
            - Linguistic features set to zero  
            - Ensemble features set to zero
            
        Notes
        -----
        Used as fallback when text is too short, empty, or processing fails.
        Ensures consistent feature structure across all processing scenarios.
        """
        
        return {
            'vader_positive': 0.0, 'vader_negative': 0.0, 'vader_neutral': 1.0, 'vader_compound': 0.0,
            'textblob_polarity': 0.0, 'textblob_subjectivity': 0.0,
            'exclamation_ratio': 0.0, 'question_ratio': 0.0, 'caps_ratio': 0.0,
            'avg_sentence_length': 0.0, 'avg_word_length': 0.0, 'text_length': 0,
            'ensemble_sentiment': 0.0, 'sentiment_agreement': 0.0, 
            'sentiment_confidence': 0.0, 'sentiment_strength': 0.0
        }
    
    def _extract_review_text(self, review: Dict) -> str:
        """
        Extract and clean text content from a single review.
        
        Parameters
        ----------
        review : dict
            Review dictionary containing 'review' field with text content
            
        Returns
        -------
        str
            Cleaned review text. Returns empty string if extraction fails
            or review structure is invalid.
            
        Notes
        -----
        Handles both list and string formats for review content. Applies
        comprehensive text cleaning pipeline including spoiler removal
        and character filtering.
        """
        try:
            review_content = review.get('review', [])
            if isinstance(review_content, list):
                review_text = ' '.join(review_content)
            else:
                review_text = str(review_content)
            
            return self.clean_text(review_text)
        except Exception as e:
            print(f"Error extracting review text: {e}")
            return ""
    
    def _calculate_review_weight(self, review: Dict, current_weights: List[float]) -> float:
        """
        Calculate the helpfulness weight for a review.
        
        Parameters
        ----------
        review : dict
            Review dictionary containing reviewer information and helpfulness data
        current_weights : list of float
            List of previously calculated weights (unused in current implementation)
            
        Returns
        -------
        float
            Logarithmic weight based on helpfulness score. Returns 1.0 if
            calculation fails or helpfulness information is missing.
            
        Notes
        -----
        Extracts helpfulness score from reviewer info string, then applies
        logarithmic scaling using np.log1p for smooth weight distribution.
        Helpfulness format expected: "X people found this review helpful"
        """
        try:
            reviewer_info = review.get('reviewer', {})
            helpfulness_info = reviewer_info.get('info', '0 people found this review helpful')
            helpfulness_score = self.extract_helpfulness_score(helpfulness_info)
            
            # Apply logarithmic scaling
            log_weight = np.log1p(helpfulness_score * self.HELPFULNESS_LOG_BASE)
            
            return log_weight
        except Exception as e:
            print(f"Error calculating review weight: {e}")
            return 1.0  # Default weight
    
    def _normalize_review_weight(self, log_weight: float, max_weight: float) -> int:
        """
        Normalize review weight to determine repetition count.
        
        Parameters
        ----------
        log_weight : float
            Logarithmic weight for the current review
        max_weight : float
            Maximum weight across all reviews for normalization
            
        Returns
        -------
        int
            Number of times to repeat the review text (1 to max_review_repetitions+1).
            Returns 1 if normalization fails.
            
        Notes
        -----
        Normalizes weights to range [1, max_review_repetitions+1] based on
        relative weight compared to maximum. Higher helpfulness scores result
        in more repetitions, giving them greater influence in analysis.
        """
        try:
            if max_weight <= 0:
                return 1
            
            normalized_weight = max(1, int((log_weight / max_weight) * self.max_review_repetitions) + 1)
            return min(normalized_weight, self.max_review_repetitions + 1)
        except Exception as e:
            print(f"Error normalizing review weight: {e}")
            return 1  # Default to single repetition
    
    def _apply_helpfulness_weighting(self, cleaned_text: str, weight_multiplier: int) -> str:
        """
        Apply helpfulness weighting by repeating review text.
        
        Parameters
        ----------
        cleaned_text : str
            Cleaned review text to be weighted
        weight_multiplier : int
            Number of times to repeat the text
            
        Returns
        -------
        str
            Text repeated weight_multiplier times, joined with spaces.
            Returns original text if weighting fails.
            
        Notes
        -----
        Simple but effective method to increase influence of helpful reviews
        in aggregate sentiment analysis. Higher helpfulness scores lead to
        more repetitions, amplifying their impact on final sentiment.
        """
        try:
            if weight_multiplier <= 0:
                weight_multiplier = 1
            return ' '.join([cleaned_text] * weight_multiplier)
        except Exception as e:
            print(f"Error applying helpfulness weighting: {e}")
            return cleaned_text  # Return original text
    
    def _validate_reviews_data(self, reviews_data: Dict) -> List[Dict]:
        """
        Validate and extract reviews list from reviews data.
        
        Parameters
        ----------
        reviews_data : dict
            Reviews data structure containing nested 'data' and 'reviews' fields
            
        Returns
        -------
        list of dict
            List of individual review dictionaries. Returns empty list if
            validation fails or structure is invalid.
            
        Notes
        -----
        Expected structure: {'data': {'reviews': [review1, review2, ...]}}
        Handles missing fields gracefully and provides defensive validation
        for downstream processing steps.
        """
        try:
            if not reviews_data or not reviews_data.get('data'):
                return []
            
            reviews_list = reviews_data.get('data', {}).get('reviews', [])
            return reviews_list if reviews_list else []
        except Exception as e:
            print(f"Error validating reviews data: {e}")
            return []

    def process_reviews(self, reviews_data: Dict) -> Tuple[str, List[float]]:
        """
        Process all reviews for a drama with helpfulness weighting applied.
        
        Parameters
        ----------
        reviews_data : dict
            Complete reviews data structure containing list of reviews with
            helpfulness information
            
        Returns
        -------
        tuple of (str, list of float)
            - Combined weighted review text with helpful reviews repeated
            - List of helpfulness weights for each review
            Returns ("", []) if processing fails or no valid reviews found.
            
        Notes
        -----
        Two-pass algorithm:
        1. Extract text and calculate logarithmic helpfulness weights
        2. Normalize weights and apply repetition-based weighting
        More helpful reviews are repeated more times in the final text,
        giving them proportionally greater influence on sentiment analysis.
        """
        try:
            reviews_list = self._validate_reviews_data(reviews_data)
            if not reviews_list:
                return "", []
            
            weighted_reviews = []
            helpfulness_weights = []
            
            # First pass: extract text and calculate weights
            for review in reviews_list:
                cleaned_text = self._extract_review_text(review)
                log_weight = self._calculate_review_weight(review, helpfulness_weights)
                helpfulness_weights.append(log_weight)
            
            # Second pass: normalize weights and apply weighting
            max_weight = max(helpfulness_weights) if helpfulness_weights else 1
            
            for i, review in enumerate(reviews_list):
                cleaned_text = self._extract_review_text(review)
                log_weight = helpfulness_weights[i]
                
                # Normalize weight and apply to text
                weight_multiplier = self._normalize_review_weight(log_weight, max_weight)
                weighted_text = self._apply_helpfulness_weighting(cleaned_text, weight_multiplier)
                weighted_reviews.append(weighted_text)
            
            # Combine all weighted reviews
            combined_reviews = ' '.join(weighted_reviews)
            
            return combined_reviews, helpfulness_weights
        except Exception as e:
            print(f"Error processing reviews: {e}")
            return "", []
    
    def extract_helpfulness_score(self, helpfulness_info: str) -> int:
        """
        Extract numerical helpfulness score from string with error handling.
        
        Parameters
        ----------
        helpfulness_info : str
            String containing helpfulness information in format:
            "X people found this review helpful"
            
        Returns
        -------
        int
            Numerical helpfulness score extracted from string. Returns 0 if
            extraction fails or no number is found.
            
        Notes
        -----
        Uses regex pattern to extract the first number from the helpfulness
        string. Handles various formats gracefully and provides fallback
        for malformed or missing helpfulness data.
        """
        try:
            match = re.search(r'(\d+) people found this review helpful', helpfulness_info)
            return int(match.group(1)) if match else 0
        except Exception as e:
            print(f"Helpfulness score extraction error: {e}")
            return 0

    def get_processing_stats(self, text: str) -> Dict[str, any]:
        """
        Get comprehensive processing statistics for a given text.
        
        Parameters
        ----------
        text : str
            Input text to analyze and generate statistics for
            
        Returns
        -------
        dict
            Dictionary containing comprehensive processing statistics:
            - original_length : int
                Character count of original text
            - cleaned_length : int  
                Character count after cleaning
            - reduction_ratio : float
                Proportion of text removed during cleaning
            - english_chars : int
                Count of English characters in original text
            - non_english_chars : int
                Count of non-English characters removed
            - non_english_ratio : float
                Proportion of non-English characters
            - meets_min_length : bool
                Whether cleaned text meets minimum length requirement
            - sentiment_summary : dict
                Key sentiment scores from all analysis methods
            Returns {'error': str} if processing fails.
            
        Notes
        -----
        Provides detailed insights into text processing pipeline effectiveness
        and sentiment analysis results. Useful for debugging, monitoring,
        and understanding data quality issues.
        """
        try:
            if not text:
                return {'error': 'Empty text provided'}
            
            # Basic text statistics
            original_length = len(text)
            cleaned_text = self.clean_text(text)
            cleaned_length = len(cleaned_text)
            
            # Character analysis
            english_chars = len(self.remove_non_english(text))
            non_english_chars = original_length - english_chars
            
            # Sentiment analysis
            sentiment_features = self.extract_sentiment_features(cleaned_text)
            
            return {
                'original_length': original_length,
                'cleaned_length': cleaned_length,
                'reduction_ratio': (original_length - cleaned_length) / max(1, original_length),
                'english_chars': english_chars,
                'non_english_chars': non_english_chars,
                'non_english_ratio': non_english_chars / max(1, original_length),
                'meets_min_length': cleaned_length >= self.min_text_length,
                'sentiment_summary': {
                    'vader_compound': sentiment_features.get('vader_compound', 0),
                    'textblob_polarity': sentiment_features.get('textblob_polarity', 0),
                    'ensemble_sentiment': sentiment_features.get('ensemble_sentiment', 0),
                    'sentiment_confidence': sentiment_features.get('sentiment_confidence', 0)
                }
            }
        except Exception as e:
            return {'error': f'Processing stats error: {e}'}
