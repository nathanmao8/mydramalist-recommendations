import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import requests
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')
import argparse
import pandas as pd
import numpy as np

# API Class
class KuryanaAPI:
    def __init__(self, base_url: str = "https://kuryana.tbdh.app"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """Make a request to the API with error handling."""
        try:
            response = self.session.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        
    def get_drama_info(self, slug: str) -> Optional[Dict]:
        """Get detailed information about a drama. Primarily to access: slug_query, title, synopsis, genres, tags. Do not use for cast."""
        return self._make_request(f"/id/{slug}")

    def get_cast(self, slug: str) -> Optional[Dict]:
        """Get cast information for a drama. Primarily to access: Main Role cast."""
        return self._make_request(f"/id/{slug}/cast")

    def get_reviews(self, slug: str) -> Optional[Dict]:
        """Get reviews for a drama. Primarily to access: review text and number of people who found each review helpful."""
        return self._make_request(f"/id/{slug}/reviews")
        
    def get_user_dramalist(self, user_id: str) -> Optional[Dict]:
        """Get user's drama list. Primarily to access: rating of watched dramas."""
        return self._make_request(f"/dramalist/{user_id}")

# Import all modules
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from predictor import Predictor
from evaluator import Evaluator
from interpretability import ShapExplainer
from taste_analyzer import TasteAnalyzer
from taste_stratified_evaluator import TasteStratifiedEvaluator
from taste_kfold_evaluator import TasteKFoldEvaluator
from taste_loocv_evaluator import TasteLOOCVEvaluator
from topic_modeling import TopicModelingExtractor

def parse_arguments():
    parser = argparse.ArgumentParser(description='Drama Recommendation with Taste Profile Analysis')
    
    # Data and model parameters
    parser.add_argument('--user-id', type=str, required=True, help='User ID for drama data')
    parser.add_argument('--output', type=str, default='drama_predictions.csv', help='Output file path')
    
    # # Feature configuration
    parser.add_argument('--use-bert', action='store_true', help='Use BERT embeddings')
    parser.add_argument('--use-sentiment', action='store_true', help='Use sentiment analysis')
    parser.add_argument('--use-tfidf', action='store_true', help='Use TF-IDF features')
    parser.add_argument('--use-position-weights', action='store_true', help='Use position-based weighting')
    parser.add_argument('--use-cast', action='store_true', help='Use cast features')
    parser.add_argument('--use-crew', action='store_true', help='Use crew features')
    parser.add_argument('--use-genres', action='store_true', help='Use genre features')
    parser.add_argument('--use-tags', action='store_true', help='Use tag features')
    parser.add_argument('--tfidf-max-features', type=int, default=1000, help='TF-IDF max features')
    parser.add_argument('--bert-cache', action='store_true', help='Cache BERT embeddings')
    parser.add_argument('--use-semantic-similarity', action='store_true', help='Use semantic similarity')
    parser.add_argument('--semantic-model', type=str, default='all-MiniLM-L6-v2', help='Semantic similarity model')
    
    # Evaluation parameters
    parser.add_argument('--run-stratified-eval', action='store_true', help='Run stratified holdout evaluation')
    parser.add_argument('--run-kfold-eval', action='store_true', default=True, help='Disable K-fold cross validation (enabled by default)')
    parser.add_argument('--run-loocv-eval', action='store_true', default=True, help='Disable LOOCV evaluation (enabled by default)')
    parser.add_argument('--eval-test-size', type=float, default=0.2, help='Test size for stratified evaluation')
    
    # Performance-based weighting
    parser.add_argument('--use-performance-weighting', action='store_true',
                       help='Use performance-based weighting for components')
    parser.add_argument('--use-dynamic-thresholding', default=True, action='store_true',
                       help='Disable dynamic thresholding at 75th percentile (enabled by default)')
    parser.add_argument('--use-fixed-weights', default=True, action='store_true',
                       help='Disable optimal fixed weights (enabled by default)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    feature_config = {
        'use_bert': False,
        'use_sentiment': False,
        'use_tfidf': False,
        'use_position_weights': False,
        'use_cast': False,
        'use_crew': False,
        'use_genres': False,
        'use_tags': False,
        'tfidf_max_features': 1000,
        'bert_cache': False,
        'use_semantic_similarity': True,  # Enable by default for text similarity
        'semantic_model': 'all-MiniLM-L6-v2'
    }

    # Initialize API
    api = KuryanaAPI()
    
    # Initialize components
    data_loader = DataLoader(api)
    feature_engineer = FeatureEngineer(feature_config)
    
    # Initialize topic modeling extractor
    topic_extractor = TopicModelingExtractor(n_topics=10, max_features=1000)
    
    # Initialize taste analyzer with semantic extractor
    semantic_extractor = getattr(feature_engineer, 'semantic_extractor', None)
    if semantic_extractor is None and args.use_semantic_similarity:
        from semantic_similarity import SemanticSimilarityExtractor
        semantic_extractor = SemanticSimilarityExtractor()
    
    taste_analyzer = TasteAnalyzer(semantic_extractor, topic_extractor)
    
    # Apply fixed weights if requested (default: True)
    if not args.no_fixed_weights:
        print("\n" + "="*80)
        print("FIXED WEIGHTS CONFIGURATION")
        print("="*80)
        fixed_weights = {
            'categorical': 0.80,
            'text': 0.20,
            'semantic': 0.00
        }
        taste_analyzer.taste_profile.set_component_weights(fixed_weights)
        print(f"✅ Applied optimal fixed weights: {fixed_weights}")
        print(f"   Based on testing: 80% categorical, 20% text, 0% semantic")
        print(f"   Best F1 score: 0.444, Best precision: 0.632")
    
    # Load and process data
    print("Loading drama data...")
    watched_dramas, unwatched_dramas = data_loader.load_all_drama_data(user_id=args.user_id)
    
    print(f"Loaded {len(watched_dramas)} watched dramas with ratings")
    print(f"Loaded {len(unwatched_dramas)} unwatched dramas for prediction")
    
    # Extract ratings from watched dramas
    watched_ratings = [d.get('user_rating', 0) for d in watched_dramas]
    print(f"Rating range: {min(watched_ratings):.1f} - {max(watched_ratings):.1f}")
    print(f"Average rating: {sum(watched_ratings)/len(watched_ratings):.2f}")
    
    # Taste Profile Analysis
    print("\n" + "="*80)
    print("TASTE PROFILE ANALYSIS")
    print("="*80)
    
    # Analyze user taste (builds and displays taste profile)
    taste_analyzer.analyze_user_taste(watched_dramas, watched_ratings)
    
    # Performance-based weighting analysis
    if args.use_performance_weighting:
        print("\n" + "="*80)
        print("PERFORMANCE-BASED WEIGHTING")
        print("="*80)
        
        # Perform performance-based weighting analysis
        weighting_results = taste_analyzer.perform_performance_weighting(watched_dramas, watched_ratings)
        
        print(f"\n✅ Performance-based weighting complete!")
        print(f"   Optimal weights: {weighting_results['optimal_weights']}")
    
    # Dynamic thresholding
    if not args.no_dynamic_thresholding:
        print("\n" + "="*80)
        print("DYNAMIC THRESHOLDING")
        print("="*80)
        
        # Calculate similarities for watched dramas to fit thresholds
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
        thresholds = taste_analyzer.fit_dynamic_thresholds(similarities_df)
        print(f"\n✅ Dynamic thresholding complete!")
        print(f"   Using 75th percentile thresholds for taste reasoning")
    
    # Calculate taste similarities for unwatched dramas
    taste_similarities = taste_analyzer.calculate_taste_similarities(unwatched_dramas)
    
    # Get pure taste-based recommendations
    taste_recommendations = taste_similarities.head(20).copy()
    
    # Save taste analysis results
    taste_analyzer.save_taste_analysis('taste_analysis_results.csv')
    taste_similarities.to_csv('taste_similarities.csv', index=False)
    
    # Display top taste-based recommendations
    print("\n🎯 TOP TASTE-BASED RECOMMENDATIONS:")
    print("-" * 60)
    for _, row in taste_recommendations.head(10).iterrows():
        print(f"• {row['Drama_Title']:<40} (Similarity: {row['Overall_Taste_Similarity']:.3f})")
        print(f"  Reason: {row['Taste_Reasoning']}")
    
    # Taste Profile Evaluation
    if args.run_stratified_eval or not args.no_kfold_eval or not args.no_loocv_eval:
        print("\n" + "="*80)
        print("TASTE PROFILE EVALUATION")
        print("="*80)
        
        # Stratified Evaluation
        if args.run_stratified_eval:
            print("\n📊 RUNNING STRATIFIED HOLDOUT EVALUATION...")
            stratified_evaluator = TasteStratifiedEvaluator(
                test_size=args.eval_test_size, 
                random_state=42
            )
            stratified_results = stratified_evaluator.run_stratified_evaluation(watched_dramas, watched_ratings)
            print("\n" + stratified_results['summary'])
            stratified_evaluator.save_results('stratified_evaluation_results')
            # stratified_evaluator.plot_results('stratified_evaluation_visualization.png')
        
        # K-Fold Evaluation
        if not args.no_kfold_eval:
            print("\n📊 RUNNING K-FOLD CROSS VALIDATION...")
            kfold_evaluator = TasteKFoldEvaluator(n_folds=5)
            kfold_results = kfold_evaluator.run_kfold_evaluation(watched_dramas, watched_ratings)
            print("\n" + kfold_results['summary'])
            kfold_evaluator.save_results('kfold_evaluation_results')
            kfold_evaluator.plot_results('kfold_evaluation_visualization.png')
        
        # LOOCV Evaluation
        if not args.no_loocv_eval:
            print("\n📊 RUNNING LOOCV EVALUATION...")
            loocv_evaluator = TasteLOOCVEvaluator()
            loocv_results = loocv_evaluator.run_loocv(watched_dramas, watched_ratings)
            print("\n" + loocv_results['summary'])
            loocv_evaluator.save_results('loocv_evaluation_results')
            loocv_evaluator.plot_results('loocv_evaluation_visualization.png')
    
    # Save final results
    print("\nSaving results...")
    taste_similarities.to_csv(args.output, index=False)
    print(f"Taste similarities saved to {args.output}")
    
    print("\n✅ Taste profile analysis and evaluation complete!")

if __name__ == "__main__":
    main() 