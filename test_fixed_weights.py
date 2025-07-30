#!/usr/bin/env python3
"""
Test script for different fixed component weight configurations.
"""

import argparse
import numpy as np
import pandas as pd
from taste_analyzer import TasteAnalyzer
from feature_engineer import FeatureEngineer
from topic_modeling import TopicModelingExtractor
from semantic_similarity import SemanticSimilarityExtractor
from data_loader import DataLoader

def load_data():
    """Load drama data for testing."""
    from main_taste_eval import KuryanaAPI
    
    # Initialize API and data loader
    api = KuryanaAPI()
    data_loader = DataLoader(api)
    
    # Load data
    print("Loading drama data...")
    watched_dramas, unwatched_dramas = data_loader.load_all_drama_data(user_id='oamen')
    
    # Extract ratings from watched dramas
    watched_ratings = [d.get('user_rating', 0) for d in watched_dramas]
    
    print(f"Loaded {len(watched_dramas)} watched dramas with ratings")
    print(f"Loaded {len(unwatched_dramas)} unwatched dramas for prediction")
    print(f"Rating range: {min(watched_ratings):.1f} - {max(watched_ratings):.1f}")
    print(f"Average rating: {sum(watched_ratings)/len(watched_ratings):.2f}")
    
    return watched_dramas, unwatched_dramas, watched_ratings

def test_weight_configuration(watched_dramas, unwatched_dramas, watched_ratings, 
                            categorical_weight, text_weight, semantic_weight, 
                            topic_extractor, semantic_extractor):
    """
    Test a specific weight configuration and return evaluation metrics.
    """
    print(f"\n🔧 Testing weights: Categorical={categorical_weight}%, Text={text_weight}%, Semantic={semantic_weight}%")
    print("=" * 80)
    
    # Create taste analyzer
    taste_analyzer = TasteAnalyzer(semantic_extractor, topic_extractor)
    
    # Set fixed weights
    weights = {
        'categorical': categorical_weight / 100.0,
        'text': text_weight / 100.0,
        'semantic': semantic_weight / 100.0
    }
    taste_analyzer.taste_profile.set_component_weights(weights)
    
    # Build taste profile
    taste_analyzer.taste_profile.build_taste_profile(watched_dramas, watched_ratings)
    
    # Calculate similarities for watched dramas (for evaluation)
    similarities_data = []
    for drama in watched_dramas:
        similarities = taste_analyzer.taste_profile.calculate_taste_similarity(drama)
        drama_data = {
            'Drama_Title': drama.get('title', ''),
            'Drama_ID': drama.get('slug', ''),
            'Categorical_Similarity': np.mean(list(similarities['categorical'].values())),
            'Text_Similarity': np.mean(list(similarities['text'].values())),
            'Semantic_Similarity': similarities['semantic'],
            'Overall_Taste_Similarity': similarities['overall']
        }
        similarities_data.append(drama_data)
    
    similarities_df = pd.DataFrame(similarities_data)
    
    # Calculate correlation with ratings
    overall_corr = np.corrcoef(similarities_df['Overall_Taste_Similarity'], watched_ratings)[0, 1]
    categorical_corr = np.corrcoef(similarities_df['Categorical_Similarity'], watched_ratings)[0, 1]
    text_corr = np.corrcoef(similarities_df['Text_Similarity'], watched_ratings)[0, 1]
    semantic_corr = np.corrcoef(similarities_df['Semantic_Similarity'], watched_ratings)[0, 1]
    
    # Calculate ranking accuracy (how well high-rated dramas rank high)
    sorted_indices = np.argsort(similarities_df['Overall_Taste_Similarity'])[::-1]
    top_25_percent = int(len(watched_ratings) * 0.25)
    top_25_dramas = sorted_indices[:top_25_percent]
    
    # Find high-rated dramas (rating >= 8.0)
    high_rated_indices = [i for i, rating in enumerate(watched_ratings) if rating >= 8.0]
    
    # Calculate precision: how many of top 25% are high-rated
    high_rated_in_top = len(set(top_25_dramas) & set(high_rated_indices))
    precision = high_rated_in_top / len(top_25_dramas) if len(top_25_dramas) > 0 else 0
    
    # Calculate recall: how many high-rated dramas are in top 25%
    recall = high_rated_in_top / len(high_rated_indices) if len(high_rated_indices) > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate similarity statistics
    mean_similarity = np.mean(similarities_df['Overall_Taste_Similarity'])
    std_similarity = np.std(similarities_df['Overall_Taste_Similarity'])
    
    results = {
        'weights': weights,
        'overall_correlation': overall_corr,
        'categorical_correlation': categorical_corr,
        'text_correlation': text_corr,
        'semantic_correlation': semantic_corr,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mean_similarity': mean_similarity,
        'std_similarity': std_similarity,
        'high_rated_count': len(high_rated_indices),
        'top_25_high_rated': high_rated_in_top
    }
    
    print(f"📊 RESULTS:")
    print(f"   Overall Correlation: {overall_corr:.3f}")
    print(f"   Categorical Correlation: {categorical_corr:.3f}")
    print(f"   Text Correlation: {text_corr:.3f}")
    print(f"   Semantic Correlation: {semantic_corr:.3f}")
    print(f"   Precision (Top 25%): {precision:.3f}")
    print(f"   Recall (High-rated): {recall:.3f}")
    print(f"   F1 Score: {f1:.3f}")
    print(f"   Mean Similarity: {mean_similarity:.3f}")
    print(f"   Std Similarity: {std_similarity:.3f}")
    print(f"   High-rated dramas: {len(high_rated_indices)}")
    print(f"   High-rated in top 25%: {high_rated_in_top}")
    
    return results

def main():
    """Main function to test different weight configurations."""
    print("🎯 FIXED WEIGHT CONFIGURATION TESTING")
    print("=" * 80)
    
    # Load data
    print("Loading data...")
    watched_dramas, unwatched_dramas, watched_ratings = load_data()
    
    # Initialize components
    feature_engineer = FeatureEngineer()
    topic_extractor = TopicModelingExtractor(n_topics=10, max_features=1000)
    semantic_extractor = SemanticSimilarityExtractor()
    
    # Test configurations
    configurations = [
        (100, 0, 0, "100% Categorical, 0% Text, 0% Semantic"),
        (80, 20, 0, "80% Categorical, 20% Text, 0% Semantic"),
        (70, 20, 10, "70% Categorical, 20% Text, 10% Semantic"),
        (70, 30, 0, "70% Categorical, 30% Text, 0% Semantic"),
        (60, 40, 0, "60% Categorical, 40% Text, 0% Semantic")
    ]
    
    all_results = []
    
    for cat_weight, text_weight, sem_weight, description in configurations:
        print(f"\n{'='*80}")
        print(f"🧪 TESTING: {description}")
        print(f"{'='*80}")
        
        results = test_weight_configuration(
            watched_dramas, unwatched_dramas, watched_ratings,
            cat_weight, text_weight, sem_weight,
            topic_extractor, semantic_extractor
        )
        
        results['description'] = description
        all_results.append(results)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("📊 SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    summary_data = []
    for results in all_results:
        summary_data.append({
            'Configuration': results['description'],
            'Overall_Correlation': results['overall_correlation'],
            'Categorical_Correlation': results['categorical_correlation'],
            'Text_Correlation': results['text_correlation'],
            'Semantic_Correlation': results['semantic_correlation'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1_Score': results['f1_score'],
            'Mean_Similarity': results['mean_similarity'],
            'Std_Similarity': results['std_similarity']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Find best configuration
    best_f1_idx = np.argmax([r['f1_score'] for r in all_results])
    best_corr_idx = np.argmax([r['overall_correlation'] for r in all_results])
    
    print(f"\n🏆 BEST CONFIGURATIONS:")
    print(f"   Best F1 Score: {all_results[best_f1_idx]['description']} (F1: {all_results[best_f1_idx]['f1_score']:.3f})")
    print(f"   Best Correlation: {all_results[best_corr_idx]['description']} (Corr: {all_results[best_corr_idx]['overall_correlation']:.3f})")
    
    # Save results
    summary_df.to_csv('fixed_weight_test_results.csv', index=False)
    print(f"\n✅ Results saved to fixed_weight_test_results.csv")

if __name__ == "__main__":
    main() 