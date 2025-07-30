# taste_profile_example.py
"""
Example script demonstrating the User Taste Profile System.

This script shows how to:
1. Build a taste profile from watched dramas
2. Calculate similarity scores for unwatched dramas
3. Get taste-based recommendations
4. Interpret the taste profile
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from taste_analyzer import TasteAnalyzer
from semantic_similarity import SemanticSimilarityExtractor
import json

def load_sample_data():
    """Load sample drama data for demonstration."""
    
    # Sample watched dramas with ratings (you would load this from your actual data)
    watched_dramas = [
        {
            'title': 'Goblin',
            'slug': 'goblin',
            'synopsis_clean': 'A modern-day goblin seeks to end his immortal life by finding a human bride.',
            'reviews_combined': 'Amazing chemistry between leads. Beautiful cinematography and emotional story.',
            'genres': ['Fantasy', 'Romance', 'Drama'],
            'tags': ['Supernatural', 'Romance', 'Fantasy', 'Emotional'],
            'main_cast': ['Gong Yoo', 'Kim Go-eun'],
            'directors': ['Lee Eung-bok'],
            'screenwriters': ['Kim Eun-sook'],
            'composers': ['Nam Hye-seung'],
            'sentiment_features': {
                'ensemble_sentiment': 0.8,
                'vader_positive': 0.7,
                'textblob_polarity': 0.6
            }
        },
        {
            'title': 'Descendants of the Sun',
            'slug': 'descendants-of-the-sun',
            'synopsis_clean': 'A love story between a special forces captain and a doctor.',
            'reviews_combined': 'Great action scenes and romantic storyline. Strong performances.',
            'genres': ['Romance', 'Action', 'Drama'],
            'tags': ['Military', 'Medical', 'Romance', 'Action'],
            'main_cast': ['Song Joong-ki', 'Song Hye-kyo'],
            'directors': ['Lee Eung-bok'],
            'screenwriters': ['Kim Eun-sook'],
            'composers': ['Gaemi'],
            'sentiment_features': {
                'ensemble_sentiment': 0.7,
                'vader_positive': 0.6,
                'textblob_polarity': 0.5
            }
        },
        {
            'title': 'Signal',
            'slug': 'signal',
            'synopsis_clean': 'A detective communicates with the past through a mysterious radio.',
            'reviews_combined': 'Intense crime thriller with time travel elements. Gripping storyline.',
            'genres': ['Crime', 'Thriller', 'Mystery'],
            'tags': ['Time Travel', 'Crime', 'Thriller', 'Mystery'],
            'main_cast': ['Lee Je-hoon', 'Kim Hye-soo'],
            'directors': ['Kim Won-seok'],
            'screenwriters': ['Kim Eun-hee'],
            'composers': ['Kim Joon-seok'],
            'sentiment_features': {
                'ensemble_sentiment': 0.3,
                'vader_positive': 0.4,
                'textblob_polarity': 0.2
            }
        }
    ]
    
    # Sample unwatched dramas
    unwatched_dramas = [
        {
            'title': 'The King: Eternal Monarch',
            'slug': 'the-king-eternal-monarch',
            'synopsis_clean': 'A modern-day Korean emperor passes through a mysterious door into a parallel world.',
            'reviews_combined': 'Complex fantasy romance with beautiful visuals and strong leads.',
            'genres': ['Fantasy', 'Romance', 'Drama'],
            'tags': ['Parallel World', 'Fantasy', 'Romance', 'Royalty'],
            'main_cast': ['Lee Min-ho', 'Kim Go-eun'],
            'directors': ['Baek Sang-hoon'],
            'screenwriters': ['Kim Eun-sook'],
            'composers': ['Nam Hye-seung'],
            'sentiment_features': {
                'ensemble_sentiment': 0.6,
                'vader_positive': 0.5,
                'textblob_polarity': 0.4
            }
        },
        {
            'title': 'Stranger',
            'slug': 'stranger',
            'synopsis_clean': 'A prosecutor with no emotions teams up with a passionate detective to solve crimes.',
            'reviews_combined': 'Brilliant crime drama with complex characters and intricate plot.',
            'genres': ['Crime', 'Drama', 'Thriller'],
            'tags': ['Crime', 'Prosecutor', 'Thriller', 'Mystery'],
            'main_cast': ['Cho Seung-woo', 'Bae Doona'],
            'directors': ['Ahn Gil-ho'],
            'screenwriters': ['Lee Soo-yeon'],
            'composers': ['Jang Young-gyu'],
            'sentiment_features': {
                'ensemble_sentiment': 0.2,
                'vader_positive': 0.3,
                'textblob_polarity': 0.1
            }
        },
        {
            'title': 'Crash Landing on You',
            'slug': 'crash-landing-on-you',
            'synopsis_clean': 'A South Korean heiress accidentally paraglides into North Korea and falls in love with a military officer.',
            'reviews_combined': 'Epic romance with amazing chemistry and beautiful story. Perfect blend of comedy and drama.',
            'genres': ['Romance', 'Drama', 'Comedy'],
            'tags': ['Military', 'Romance', 'Cross-border', 'Comedy'],
            'main_cast': ['Hyun Bin', 'Son Ye-jin'],
            'directors': ['Lee Jung-hyo'],
            'screenwriters': ['Park Ji-eun'],
            'composers': ['Nam Hye-seung'],
            'sentiment_features': {
                'ensemble_sentiment': 0.8,
                'vader_positive': 0.9,
                'textblob_polarity': 0.7
            }
        },
        {
            'title': 'Kingdom',
            'slug': 'kingdom',
            'synopsis_clean': 'A crown prince investigates a mysterious plague that turns people into zombies in Joseon-era Korea.',
            'reviews_combined': 'Gripping historical zombie thriller with stunning cinematography and intense action.',
            'genres': ['Historical', 'Thriller', 'Horror'],
            'tags': ['Zombie', 'Historical', 'Thriller', 'Political'],
            'main_cast': ['Ju Ji-hoon', 'Bae Doona'],
            'directors': ['Kim Seong-hun'],
            'screenwriters': ['Kim Eun-hee'],
            'composers': ['Mowg'],
            'sentiment_features': {
                'ensemble_sentiment': 0.1,
                'vader_positive': 0.2,
                'textblob_polarity': 0.0
            }
        },
        {
            'title': 'My Mister',
            'slug': 'my-mister',
            'synopsis_clean': 'A middle-aged man and a young woman form an unlikely friendship that helps them overcome life\'s hardships.',
            'reviews_combined': 'Deeply emotional and realistic portrayal of human relationships. Masterful storytelling.',
            'genres': ['Drama', 'Slice of Life'],
            'tags': ['Slice of Life', 'Drama', 'Friendship', 'Emotional'],
            'main_cast': ['Lee Sun-kyun', 'IU'],
            'directors': ['Kim Won-seok'],
            'screenwriters': ['Park Hae-young'],
            'composers': ['Jang Young-gyu'],
            'sentiment_features': {
                'ensemble_sentiment': 0.3,
                'vader_positive': 0.4,
                'textblob_polarity': 0.2
            }
        },
        {
            'title': 'What\'s Wrong with Secretary Kim',
            'slug': 'whats-wrong-with-secretary-kim',
            'synopsis_clean': 'A narcissistic CEO tries to prevent his perfect secretary from quitting after nine years.',
            'reviews_combined': 'Funny and sweet office romance with great chemistry and lighthearted humor.',
            'genres': ['Romance', 'Comedy', 'Drama'],
            'tags': ['Office Romance', 'Comedy', 'Romance', 'Workplace'],
            'main_cast': ['Park Seo-joon', 'Park Min-young'],
            'directors': ['Park Joon-hwa'],
            'screenwriters': ['Jung Kyung-yoon'],
            'composers': ['Nam Hye-seung'],
            'sentiment_features': {
                'ensemble_sentiment': 0.7,
                'vader_positive': 0.8,
                'textblob_polarity': 0.6
            }
        },
        {
            'title': 'Signal',
            'slug': 'signal',
            'synopsis_clean': 'A detective communicates with the past through a mysterious radio to solve cold cases.',
            'reviews_combined': 'Intense crime thriller with time travel elements. Gripping storyline and strong performances.',
            'genres': ['Crime', 'Thriller', 'Mystery'],
            'tags': ['Time Travel', 'Crime', 'Thriller', 'Mystery'],
            'main_cast': ['Lee Je-hoon', 'Kim Hye-soo'],
            'directors': ['Kim Won-seok'],
            'screenwriters': ['Kim Eun-hee'],
            'composers': ['Kim Joon-seok'],
            'sentiment_features': {
                'ensemble_sentiment': 0.3,
                'vader_positive': 0.4,
                'textblob_polarity': 0.2
            }
        },
        {
            'title': 'Hotel del Luna',
            'slug': 'hotel-del-luna',
            'synopsis_clean': 'A mysterious hotel for ghosts is run by a beautiful but ill-tempered CEO who has been cursed for 1300 years.',
            'reviews_combined': 'Beautiful fantasy drama with stunning visuals and emotional depth. Unique concept.',
            'genres': ['Fantasy', 'Romance', 'Drama'],
            'tags': ['Supernatural', 'Fantasy', 'Romance', 'Hotel'],
            'main_cast': ['IU', 'Yeo Jin-goo'],
            'directors': ['Oh Choong-hwan'],
            'screenwriters': ['Hong Jung-eun'],
            'composers': ['Nam Hye-seung'],
            'sentiment_features': {
                'ensemble_sentiment': 0.6,
                'vader_positive': 0.7,
                'textblob_polarity': 0.5
            }
        },
        {
            'title': 'Itaewon Class',
            'slug': 'itaewon-class',
            'synopsis_clean': 'A young man opens a restaurant in Itaewon to seek revenge against a powerful food company.',
            'reviews_combined': 'Inspiring underdog story with great character development and social commentary.',
            'genres': ['Drama', 'Business'],
            'tags': ['Revenge', 'Business', 'Drama', 'Underdog'],
            'main_cast': ['Park Seo-joon', 'Kim Da-mi'],
            'directors': ['Kim Sung-yoon'],
            'screenwriters': ['Jo Kwang-jin'],
            'composers': ['Park Se-joon'],
            'sentiment_features': {
                'ensemble_sentiment': 0.4,
                'vader_positive': 0.5,
                'textblob_polarity': 0.3
            }
        },
        {
            'title': 'Vincenzo',
            'slug': 'vincenzo',
            'synopsis_clean': 'A Korean-Italian lawyer returns to Korea and uses his mafia connections to fight corruption.',
            'reviews_combined': 'Stylish revenge drama with dark humor and intense action. Song Joong-ki shines.',
            'genres': ['Crime', 'Drama', 'Action'],
            'tags': ['Revenge', 'Crime', 'Action', 'Lawyer'],
            'main_cast': ['Song Joong-ki', 'Jeon Yeo-been'],
            'directors': ['Kim Hee-won'],
            'screenwriters': ['Park Jae-bum'],
            'composers': ['Park Se-joon'],
            'sentiment_features': {
                'ensemble_sentiment': 0.5,
                'vader_positive': 0.6,
                'textblob_polarity': 0.4
            }
        }
    ]
    
    # Sample ratings (1-10 scale)
    ratings = [9.0, 8.5, 9.2]  # Ratings for the 3 watched dramas
    
    return watched_dramas, unwatched_dramas, ratings

def main():
    """Main function demonstrating taste profile functionality."""
    
    print("🎭 USER TASTE PROFILE SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Load sample data
    watched_dramas, unwatched_dramas, ratings = load_sample_data()
    
    print(f"Loaded {len(watched_dramas)} watched dramas and {len(unwatched_dramas)} unwatched dramas")
    
    # Initialize semantic extractor (for text embeddings)
    semantic_extractor = SemanticSimilarityExtractor()
    
    # Initialize taste analyzer
    taste_analyzer = TasteAnalyzer(semantic_extractor)
    
    # 1. Build and analyze taste profile
    print("\n1️⃣ BUILDING TASTE PROFILE...")
    taste_analysis = taste_analyzer.analyze_user_taste(watched_dramas, ratings)
    
    # 2. Calculate taste similarities for unwatched dramas
    print("\n2️⃣ CALCULATING TASTE SIMILARITIES...")
    taste_similarities = taste_analyzer.calculate_taste_similarities(unwatched_dramas)
    
    print("\nTaste Similarity Results:")
    for _, row in taste_similarities.iterrows():
        print(f"• {row['Drama_Title']:<30} Overall: {row['Overall_Taste_Similarity']:.3f}")
        print(f"  Categorical: {row['Categorical_Similarity']:.3f}, Text: {row['Text_Similarity']:.3f}")
    
    # 3. Get taste-based recommendations
    print("\n3️⃣ GENERATING TASTE-BASED RECOMMENDATIONS...")
    # Use the already calculated similarities instead of recalculating
    recommendations = taste_similarities.head(5).copy()
    
    print("\nTop Taste-Based Recommendations:")
    for _, row in recommendations.iterrows():
        print(f"• {row['Drama_Title']}")
        print(f"  Similarity: {row['Overall_Taste_Similarity']:.3f}")
        print(f"  Reasoning: {row['Taste_Reasoning']}")
        print()
    
    # 4. Save results
    print("\n4️⃣ SAVING RESULTS...")
    taste_analyzer.save_taste_analysis('example_taste_analysis.csv')
    taste_similarities.to_csv('example_taste_similarities.csv', index=False)
    
    print("✅ Taste profile analysis complete!")
    print("\nGenerated files:")
    print("• example_taste_analysis.csv - Taste profile insights")
    print("• example_taste_similarities.csv - Similarity scores and recommendations")
    print("• user_taste_profile.json - Complete taste profile")

if __name__ == "__main__":
    main() 