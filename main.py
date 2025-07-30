import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import requests
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')
import argparse
import pandas as pd

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
from text_processor import TextProcessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from predictor import Predictor
from evaluator import Evaluator
from interpretability import ShapExplainer
from taste_analyzer import TasteAnalyzer
from taste_stratified_evaluator import TasteStratifiedEvaluator
from taste_loocv_evaluator import TasteLOOCVEvaluator

def parse_arguments():
    parser = argparse.ArgumentParser(description='Drama Rating Prediction System')
    # parser.add_argument('--user_id', type=str, required=True, help='User ID for predictions')
    parser.add_argument('--user_id', type=str, default='Oamen', help='User ID for predictions')
    parser.add_argument('--output', '-o', type=str, default='drama_predictions.csv', help='Output CSV file')
    
    parser.add_argument('--use-bert', action='store_true', default=True, 
                       help='Enable BERT embeddings (computationally expensive)')
    parser.add_argument('--use-sentiment', action='store_true', default=False,
                       help='Enable sentiment analysis features')
    parser.add_argument('--use-tfidf', action='store_true', default=True,
                       help='Enable TF-IDF text features')
    parser.add_argument('--use-position-weights', action='store_true', default=True,
                       help='Enable position-based weighting for actors/genres/tags')
    parser.add_argument('--use-cast', action='store_true', default=True,
                       help='Enable cast/actor features')
    parser.add_argument('--use-crew', action='store_true', default=True,
                       help='Enable director/screenwriter/composer features')
    parser.add_argument('--use-genres', action='store_true', default=True,
                       help='Enable genre features')
    parser.add_argument('--use-tags', action='store_true', default=True,
                       help='Enable tag features')
    parser.add_argument('--use-semantic-similarity', action='store_true', default=True,
                   help='Enable semantic similarity features (synopsis-review, synopsis-synopsis, review-review)')
    parser.add_argument('--semantic-model', type=str, default='all-MiniLM-L6-v2',
                   choices=['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'distilbert-base-nli-mean-tokens'],
                   help='Sentence transformer model for semantic similarity')

    # Taste Profile Arguments
    parser.add_argument('--enable-taste-profile', action='store_true', default=True,
                       help='Enable user taste profile analysis and recommendations')
    parser.add_argument('--taste-weight', type=float, default=0.3,
                       help='Weight for taste similarity in enhanced predictions (0.0-1.0)')
    parser.add_argument('--run-stratified-eval', action='store_true', default=True,
                       help='Run stratified holdout evaluation on taste profile')
    parser.add_argument('--run-loocv-eval', action='store_true', default=False,
                       help='Run LOOCV evaluation on taste profile (comprehensive but slow)')
    parser.add_argument('--eval-test-size', type=float, default=0.3,
                       help='Test size for stratified evaluation (0.1-0.5)')

    # Advanced toggles
    parser.add_argument('--tfidf-max-features', type=int, default=1000,
                       help='Maximum features for TF-IDF vectorization')
    parser.add_argument('--bert-cache', action='store_true', default=True,
                       help='Use BERT embedding cache (recommended)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    feature_config = {
        'use_bert': args.use_bert,
        'use_sentiment': args.use_sentiment,
        'use_tfidf': args.use_tfidf,
        'use_position_weights': args.use_position_weights,
        'use_cast': args.use_cast,
        'use_crew': args.use_crew,
        'use_genres': args.use_genres,
        'use_tags': args.use_tags,
        'tfidf_max_features': args.tfidf_max_features,
        'bert_cache': args.bert_cache,
        'use_semantic_similarity': args.use_semantic_similarity,
        'semantic_model': args.semantic_model
    }

    # Initialize API
    api = KuryanaAPI()
    
    # Initialize components
    data_loader = DataLoader(api)
    text_processor = TextProcessor()
    feature_engineer = FeatureEngineer(feature_config)
    model_trainer = ModelTrainer()
    predictor = Predictor()
    evaluator = Evaluator()
    
    # Initialize taste analyzer if enabled
    taste_analyzer = None
    if args.enable_taste_profile:
        taste_analyzer = TasteAnalyzer(feature_engineer.semantic_extractor)
    
    # Load and process data
    print("Loading drama data...")
    watched_dramas, unwatched_dramas = data_loader.load_all_drama_data(user_id=args.user_id)
    
    print(f"Loaded {len(watched_dramas)} watched dramas with ratings")
    print(f"Loaded {len(unwatched_dramas)} unwatched dramas for prediction")
    
    # Create both traditional and hybrid features
    print("Creating features for watched dramas (training)...")
    X_traditional, X_hybrid, y_train = feature_engineer.create_all_feature_sets(
        watched_dramas, text_processor, is_training=True
    )

    # Get feature names for display
    print("Retrieving feature names...")
    feature_names_trad, feature_names_hybrid = feature_engineer.get_feature_names()
    feature_names_dict = {'traditional': feature_names_trad, 'hybrid': feature_names_hybrid}

    # Create features for prediction
    print("Creating features for unwatched dramas (prediction)...")
    X_predict_traditional, X_predict_hybrid = feature_engineer.create_all_feature_sets(
        unwatched_dramas, text_processor, is_training=False
    )
        
    # Display feature information
    feature_info = feature_engineer.get_feature_info()
    print(f"Feature dimensions: {feature_info}")
    
    # Train models
    print("Training all models with LOOCV...")
    trained_models, evaluation_results, loocv_predictions = model_trainer.train_models(X_traditional, X_hybrid, y_train)
        

    # Generate predictions
    print("Generating predictions with all models...")
    predictions_unwatched = predictor.predict_all_dramas(
        unwatched_dramas, trained_models, X_predict_traditional, X_predict_hybrid
    )
    
    # Taste Profile Analysis and Enhancement
    if taste_analyzer:
        print("\n" + "="*80)
        print("TASTE PROFILE ANALYSIS")
        print("="*80)
        
        # Extract ratings from watched dramas
        watched_ratings = [d.get('user_rating', 0) for d in watched_dramas]
        
        # Analyze user taste
        taste_analysis = taste_analyzer.analyze_user_taste(watched_dramas, watched_ratings)
        
        # Calculate taste similarities for unwatched dramas
        taste_similarities = taste_analyzer.calculate_taste_similarities(unwatched_dramas)
        
        # Enhance predictions with taste similarity
        enhanced_predictions = taste_analyzer.enhance_predictions_with_taste(
            predictions_unwatched, taste_similarities
        )
        
        # Get pure taste-based recommendations
        taste_recommendations = taste_similarities.head(20).copy()
        
        # Compare taste vs predictions
        comparison_analysis = taste_analyzer.compare_taste_vs_predictions(
            taste_similarities, predictions_unwatched
        )
        
        # Save taste analysis results
        taste_analyzer.save_taste_analysis('taste_analysis_results.csv')
        taste_similarities.to_csv('taste_similarities.csv', index=False)
        enhanced_predictions.to_csv('enhanced_predictions.csv', index=False)
        
        # Display top taste-based recommendations
        print("\n🎯 TOP TASTE-BASED RECOMMENDATIONS:")
        print("-" * 60)
        for _, row in taste_recommendations.head(10).iterrows():
            print(f"• {row['Drama_Title']:<40} (Similarity: {row['Overall_Taste_Similarity']:.3f})")
            print(f"  Reason: {row['Taste_Reasoning']}")
        
        # Use enhanced predictions as final output
        predictions_unwatched = enhanced_predictions
    
    # Save results
    print("Saving results...")
    predictions_unwatched.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")

    # Display evaluation results
    evaluator.display_results(evaluation_results)
    evaluator.save_loocv_predictions(watched_dramas, loocv_predictions)

    # Display feature importances
    print("\nFeature Importances:")
    evaluator.display_feature_importances(evaluation_results, feature_names_dict)

    # Display permutation importance
    print("\nPermutation Importance:")
    X_dict_for_eval = {'traditional': X_traditional, 'hybrid': X_hybrid}
    evaluator.display_permutation_importance(trained_models, X_dict_for_eval, y_train, feature_names_dict)

    # Run SHAP Explainer (only if dimensions match)
    try:
    X_df_dict_for_shap = {
        'traditional': pd.DataFrame(X_traditional, columns=feature_names_trad),
        'hybrid': pd.DataFrame(X_hybrid, columns=feature_names_hybrid)
    }
    shap_explainer = ShapExplainer(trained_models, X_df_dict_for_shap, feature_names_dict)
    shap_explainer.explain_models()
    except ValueError as e:
        print(f"\n⚠️  SHAP Analysis skipped due to dimension mismatch: {e}")
        print("This is likely due to feature engineering differences between training and prediction.")
        print("The taste profile system and predictions are still working correctly.")
    
    # After all processing is done, save the potentially updated BERT and semantic similarity caches.
    if feature_config['use_bert']:
        feature_engineer.bert_extractor.save_cache()
    
    if feature_config['use_semantic_similarity']:
        feature_engineer.semantic_extractor.save_cache()

    # Display top 10 predictions from each model type
    print("\nTop 10 Predictions by Model Type:")
    print("-" * 80)
    
    # Check if we have taste-adjusted predictions
    if 'Taste_Adjusted_Prediction' in predictions_unwatched.columns:
        prediction_columns = ['Traditional_Ensemble', 'BERT_Ensemble', 'Final_Prediction', 'Taste_Adjusted_Prediction']
    else:
        prediction_columns = ['Traditional_Ensemble', 'BERT_Ensemble', 'Final_Prediction']
    
    for col in prediction_columns:
        print(f"\n{col}:")
        top_10 = predictions_unwatched.nlargest(10, col)
        for _, row in top_10.iterrows():
            print(f"  {row['Drama_Title']:<40} {row[col]:<8.2f}")

if __name__ == "__main__":
    main()