import numpy as np
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class TextProcessor:
    def __init__(self):
        self.english_pattern = re.compile(r'[a-zA-Z\s\.,!?;:\-\'\"()]+')
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def remove_non_english(self, text: str) -> str:
        """Remove non-English characters from text."""
        if not text:
            return ""
        
        # Extract English text
        english_parts = self.english_pattern.findall(text)
        return ' '.join(english_parts).strip()
    
    def remove_spoiler_warnings(self, text: str) -> str:
        """Remove spoiler warnings and disclaimers."""
        patterns = [
            r'This review may contain spoilers',
            r'Spoiler Alert',
            r'Warning:.*spoiler',
            r'Disclaimer:.*',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def clean_text(self, text: str) -> str:
        """Comprehensive text cleaning."""
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
    
    def extract_sentiment_features(self, text: str) -> Dict:
        """Extract comprehensive sentiment features using VADER and TextBlob."""
        
        if not text or len(text.strip()) < 5:
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
        """Extract VADER sentiment scores."""
        
        scores = self.vader_analyzer.polarity_scores(text)
        
        return {
            'vader_positive': scores['pos'],
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu'],
            'vader_compound': scores['compound']
        }
    
    def _extract_textblob_features(self, text: str) -> Dict:
        """Extract TextBlob sentiment scores."""
        
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
        """Extract linguistic features without word lists."""
        
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
    
    def _calculate_ensemble_features(self, vader_features: Dict, textblob_features: Dict) -> Dict:
        """Calculate ensemble sentiment scores from multiple methods."""
        
        # Normalize scores to same scale (-1 to 1)
        vader_normalized = vader_features['vader_compound']
        textblob_normalized = textblob_features['textblob_polarity']
        
        # Weighted ensemble (VADER tends to be more accurate for social media text)
        weighted_sentiment = (vader_normalized * 0.6) + (textblob_normalized * 0.4)
        
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
    
    def _get_empty_sentiment_features(self) -> Dict:
        """Return empty sentiment features for missing/invalid text."""
        
        return {
            'vader_positive': 0.0, 'vader_negative': 0.0, 'vader_neutral': 1.0, 'vader_compound': 0.0,
            'textblob_polarity': 0.0, 'textblob_subjectivity': 0.0,
            'exclamation_ratio': 0.0, 'question_ratio': 0.0, 'caps_ratio': 0.0,
            'avg_sentence_length': 0.0, 'avg_word_length': 0.0, 'text_length': 0,
            'ensemble_sentiment': 0.0, 'sentiment_agreement': 0.0, 
            'sentiment_confidence': 0.0, 'sentiment_strength': 0.0
        }
    
    def process_reviews(self, reviews_data: Dict) -> Tuple[str, List[float]]:
        """Process all reviews for a drama with helpfulness weighting applied."""
        
        if not reviews_data or not reviews_data.get('data'):
            return "", []
        
        reviews_list = reviews_data.get('data', {}).get('reviews', [])
        
        if not reviews_list:
            return "", []
        
        weighted_reviews = []
        helpfulness_weights = []
        
        for review in reviews_list:
            # Clean review text
            review_content = review.get('review', [])
            if isinstance(review_content, list):
                review_text = ' '.join(review_content)
            else:
                review_text = str(review_content)
            
            cleaned_text = self.clean_text(review_text)
            
            # Extract helpfulness score
            reviewer_info = review.get('reviewer', {})
            helpfulness_info = reviewer_info.get('info', '0 people found this review helpful')
            helpfulness_score = self.extract_helpfulness_score(helpfulness_info)
            
            # Apply logarithmic scaling
            log_weight = np.log1p(helpfulness_score)
            helpfulness_weights.append(log_weight)
            
            # Weight by repeating text based on helpfulness (normalized)
            # Normalize weights to reasonable repetition counts (1-3 times)
            max_weight = max(helpfulness_weights) if helpfulness_weights else 1
            normalized_weight = max(1, int((log_weight / max_weight) * 2) + 1)
            
            # Repeat the review text based on its helpfulness
            weighted_text = ' '.join([cleaned_text] * normalized_weight)
            weighted_reviews.append(weighted_text)
        
        # Combine weighted reviews
        combined_reviews = ' '.join(weighted_reviews)
        
        return combined_reviews, helpfulness_weights
    
    def extract_helpfulness_score(self, helpfulness_info: str) -> int:
        """Extract numerical helpfulness score from string."""
        match = re.search(r'(\d+) people found this review helpful', helpfulness_info)
        return int(match.group(1)) if match else 0
