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
        """
        Initialize the Kuryana API client.
        
        Parameters
        ----------
        base_url : str, optional
            Base URL for the Kuryana API, by default "https://kuryana.tbdh.app"
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """
        Make a request to the API with error handling.
        
        Parameters
        ----------
        endpoint : str
            API endpoint path to request (e.g., "/id/drama-slug")
            
        Returns
        -------
        Optional[Dict]
            JSON response as dictionary if successful, None if request failed
            
        Notes
        -----
        This is a private method that handles HTTP errors and network issues
        gracefully by catching RequestException and returning None on failure.
        """
        try:
            response = self.session.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        
    def get_drama_info(self, slug: str) -> Optional[Dict]:
        """
        Get detailed information about a drama.
        
        Parameters
        ----------
        slug : str
            Drama slug identifier for the API request
            
        Returns
        -------
        Optional[Dict]
            Drama information including slug_query, title, synopsis, genres, 
            and tags if successful, None if request failed
            
        Notes
        -----
        Used to access: slug_query, title, synopsis, genres, tags, year, rating, watchers, country, type.
        Do not use for cast information - use get_cast() instead.
        """
        return self._make_request(f"/id/{slug}")

    def get_cast(self, slug: str) -> Optional[Dict]:
        """
        Get cast information for a drama.
        
        Parameters
        ----------
        slug : str
            Drama slug identifier for the API request
            
        Returns
        -------
        Optional[Dict]
            Cast information with main role actors if successful, 
            None if request failed
            
        Notes
        -----
        Used to access main role cast members. For general drama
        information, use get_drama_info() instead.
        """
        return self._make_request(f"/id/{slug}/cast")

    def get_reviews(self, slug: str) -> Optional[Dict]:
        """
        Get reviews for a drama.
        
        Parameters
        ----------
        slug : str
            Drama slug identifier for the API request
            
        Returns
        -------
        Optional[Dict]
            Review data including review text and helpfulness ratings 
            if successful, None if request failed
            
        Notes
        -----
        Primarily used to access review text and number of people who 
        found each review helpful for sentiment analysis and text processing.
        """
        return self._make_request(f"/id/{slug}/reviews")
        
    def get_user_dramalist(self, user_id: str) -> Optional[Dict]:
        """
        Get user's drama list with ratings.
        
        Parameters
        ----------
        user_id : str
            User identifier for the API request
            
        Returns
        -------
        Optional[Dict]
            User's drama list with ratings for watched dramas if successful,
            None if request failed
            
        Notes
        -----
        Primarily used to access user ratings of watched dramas for training
        the recommendation model.
        """
        return self._make_request(f"/dramalist/{user_id}")

# Import all modules
from data_loader import DataLoader
from text_processor import TextProcessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from predictor import Predictor
from evaluator import Evaluator
from interpretability import ShapExplainer

def add_basic_arguments(parser):
    """
    Add basic required arguments to the argument parser.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        ArgumentParser instance to add arguments to
        
    Notes
    -----
    Adds user-id (required) and output file arguments. These are the
    minimum required arguments for the recommendation system to function.
    """
    parser.add_argument('--user-id', type=str, required=True, help='User ID for predictions')
    parser.add_argument('--output', '-o', type=str, default='drama_predictions.csv', help='Output CSV file')

def add_feature_arguments(parser):
    """
    Add feature toggle arguments to enable/disable different feature types.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        ArgumentParser instance to add feature toggle arguments to
        
    Notes
    -----
    Creates a feature options group with toggles for BERT embeddings,
    sentiment analysis, TF-IDF, position weights, cast/crew features,
    genres, tags, semantic similarity, numerical features, country, and type.
    All features are enabled by default for optimal performance.
    """
    feature_group = parser.add_argument_group('Feature Options', 'Enable/disable different feature types')
    
    feature_group.add_argument('--use-bert', action='store_true', default=True, 
                              help='Enable BERT embeddings (computationally expensive)')
    feature_group.add_argument('--use-sentiment', action='store_true', default=True,
                              help='Enable sentiment analysis features')
    feature_group.add_argument('--use-tfidf', action='store_true', default=True,
                              help='Enable TF-IDF text features')
    feature_group.add_argument('--use-position-weights', action='store_true', default=True,
                              help='Enable position-based weighting for actors/genres/tags')
    feature_group.add_argument('--use-cast', action='store_true', default=True,
                              help='Enable cast/actor features')
    feature_group.add_argument('--use-crew', action='store_true', default=True,
                              help='Enable director/screenwriter/composer features')
    feature_group.add_argument('--use-genres', action='store_true', default=True,
                              help='Enable genre features')
    feature_group.add_argument('--use-tags', action='store_true', default=True,
                              help='Enable tag features')
    feature_group.add_argument('--use-semantic-similarity', action='store_true', default=True,
                              help='Enable semantic similarity features (synopsis-review, synopsis-synopsis, review-review)')
    feature_group.add_argument('--use-numerical-features', action='store_true', default=True,
                              help='Enable numerical features (year, rating, watchers)')
    feature_group.add_argument('--use-country', action='store_true', default=True,
                              help='Enable country features')
    feature_group.add_argument('--use-type', action='store_true', default=True,
                              help='Enable drama type features')

def add_advanced_arguments(parser):
    """
    Add advanced configuration arguments for model tuning.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        ArgumentParser instance to add advanced configuration arguments to
        
    Notes
    -----
    Creates an advanced options group with semantic model selection,
    TF-IDF feature limits, and BERT caching options. These settings
    allow fine-tuning of computational performance and model behavior.
    """
    advanced_group = parser.add_argument_group('Advanced Options', 'Advanced configuration settings')
    
    advanced_group.add_argument('--semantic-model', type=str, default='all-MiniLM-L6-v2',
                               choices=['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'distilbert-base-nli-mean-tokens'],
                               help='Sentence transformer model for semantic similarity')
    advanced_group.add_argument('--tfidf-max-features', type=int, default=1000,
                               help='Maximum features for TF-IDF vectorization')
    advanced_group.add_argument('--bert-cache', action='store_true', default=True,
                               help='Use BERT embedding cache (recommended)')

def parse_arguments():
    """
    Parse command line arguments with organized argument groups.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments containing all configuration options
        
    Notes
    -----
    Creates an ArgumentParser with three organized groups: basic arguments,
    feature options, and advanced options. This provides a clean CLI interface
    for configuring the drama recommendation system.
    """
    parser = argparse.ArgumentParser(description='Drama Rating Prediction System')
    
    add_basic_arguments(parser)
    add_feature_arguments(parser)
    add_advanced_arguments(parser)
    
    return parser.parse_args()

def create_feature_config(args):
    """
    Create feature configuration dictionary from parsed command line arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments containing feature toggles and settings
        
    Returns
    -------
    dict
        Feature configuration dictionary with boolean flags and settings
        for all feature types and advanced options
        
    Notes
    -----
    Converts command line arguments into a structured configuration dictionary
    that can be passed to the FeatureEngineer for consistent feature processing.
    """
    return {
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
        'semantic_model': args.semantic_model,
        'use_numerical_features': args.use_numerical_features,
        'use_country': args.use_country,
        'use_type': args.use_type
    }

def initialize_components(feature_config):
    """
    Initialize all system components with the given feature configuration.
    
    Parameters
    ----------
    feature_config : dict
        Feature configuration dictionary containing all feature flags and settings
        
    Returns
    -------
    tuple
        6-element tuple containing initialized components:
        (data_loader, text_processor, feature_engineer, model_trainer, predictor, evaluator)
        
    Notes
    -----
    Creates instances of all major system components in the correct order
    with proper configuration. The feature_config is only passed to
    FeatureEngineer, while other components use default configurations.
    """
    api = KuryanaAPI()
    data_loader = DataLoader(api)
    text_processor = TextProcessor()
    feature_engineer = FeatureEngineer(feature_config)
    model_trainer = ModelTrainer()
    predictor = Predictor()
    evaluator = Evaluator()
    
    return data_loader, text_processor, feature_engineer, model_trainer, predictor, evaluator

def load_and_prepare_data(data_loader, user_id):
    """
    Load drama data for both training and prediction.
    
    Parameters
    ----------
    data_loader : DataLoader
        Initialized DataLoader instance for API communication
    user_id : str
        User identifier for loading personal drama ratings
        
    Returns
    -------
    watched_dramas : list
        List of watched drama dictionaries with user ratings for training
    unwatched_dramas : list
        List of unwatched drama dictionaries for generating predictions
        
    Notes
    -----
    Calls data_loader.load_all_drama_data() to fetch both watched dramas
    (for training) and unwatched dramas (for prediction) from the API.
    Prints summary statistics about the loaded data.
    """
    print("Loading drama data...")
    watched_dramas, unwatched_dramas = data_loader.load_all_drama_data(user_id=user_id)
    
    print(f"Loaded {len(watched_dramas)} watched dramas with ratings")
    print(f"Loaded {len(unwatched_dramas)} unwatched dramas for prediction")
    
    return watched_dramas, unwatched_dramas

def create_features(feature_engineer, text_processor, watched_dramas, unwatched_dramas):
    """
    Create features for training and prediction datasets.
    
    Processes watched dramas to create training features and unwatched dramas 
    to create prediction features using the specified feature engineering pipeline.

    Parameters
    ----------
    feature_engineer : FeatureEngineer
        Configured feature engineering instance
    text_processor : TextProcessor
        Text processing utilities instance
    watched_dramas : list
        List of watched drama dictionaries with ratings
    unwatched_dramas : list
        List of unwatched drama dictionaries for prediction

    Returns
    -------
    X_traditional : array
        Traditional features for training
    X_hybrid : array
        Hybrid features for training
    y_train : array
        Training target ratings
    X_predict_traditional : array
        Traditional features for prediction
    X_predict_hybrid : array
        Hybrid features for prediction
    feature_names_dict : dict
        Feature names for both model types

    Notes
    -----
    This function calls feature_engineer.create_all_feature_sets() twice,
    once for training data (is_training=True) and once for prediction data
    (is_training=False). Also retrieves and displays feature information.
    """
    print("Creating features for watched dramas (training)...")
    X_traditional, X_hybrid, y_train = feature_engineer.create_all_feature_sets(
        watched_dramas, text_processor, is_training=True
    )

    print("Retrieving feature names...")
    feature_names_trad, feature_names_hybrid = feature_engineer.get_feature_names()
    feature_names_dict = {'traditional': feature_names_trad, 'hybrid': feature_names_hybrid}

    print("Creating features for unwatched dramas (prediction)...")
    X_predict_traditional, X_predict_hybrid = feature_engineer.create_all_feature_sets(
        unwatched_dramas, text_processor, is_training=False
    )
    
    feature_info = feature_engineer.get_feature_info()
    print(f"Feature dimensions: {feature_info}")
    
    return (X_traditional, X_hybrid, y_train, 
            X_predict_traditional, X_predict_hybrid, 
            feature_names_dict)

def train_and_predict(model_trainer, predictor, X_traditional, X_hybrid, y_train, 
                     X_predict_traditional, X_predict_hybrid, unwatched_dramas):
    """
    Train models using LOOCV and generate predictions for unwatched dramas.
    
    Parameters
    ----------
    model_trainer : ModelTrainer
        Initialized model training instance
    predictor : Predictor
        Initialized prediction instance
    X_traditional : array
        Traditional features for training
    X_hybrid : array
        Hybrid features for training
    y_train : array
        Training target ratings
    X_predict_traditional : array
        Traditional features for prediction
    X_predict_hybrid : array
        Hybrid features for prediction
    unwatched_dramas : list
        List of unwatched drama dictionaries for prediction
        
    Returns
    -------
    trained_models : dict
        Dictionary of trained models by type
    evaluation_results : dict
        LOOCV evaluation metrics for all models
    loocv_predictions : dict
        Leave-one-out cross-validation predictions
    predictions_unwatched : pd.DataFrame
        Predictions for all unwatched dramas with model scores
        
    Notes
    -----
    Uses Leave-One-Out Cross-Validation (LOOCV) for model training and evaluation.
    Generates predictions using both traditional and hybrid model ensembles.
    """
    print("Training all models with LOOCV...")
    trained_models, evaluation_results, loocv_predictions = model_trainer.train_models(
        X_traditional, X_hybrid, y_train
    )

    print("Generating predictions with all models...")
    predictions_unwatched = predictor.predict_all_dramas(
        unwatched_dramas, trained_models, X_predict_traditional, X_predict_hybrid
    )
    
    return trained_models, evaluation_results, loocv_predictions, predictions_unwatched

def display_and_analyze_results(evaluator, evaluation_results, loocv_predictions, 
                               watched_dramas, feature_names_dict, trained_models, 
                               X_traditional, X_hybrid, y_train, predictions_unwatched):
    """
    Display evaluation results and perform comprehensive model analysis.
    
    Parameters
    ----------
    evaluator : Evaluator
        Initialized evaluator instance for result analysis
    evaluation_results : dict
        LOOCV evaluation metrics for all models
    loocv_predictions : dict
        Leave-one-out cross-validation predictions
    watched_dramas : list
        List of watched drama dictionaries with ratings
    feature_names_dict : dict
        Feature names for both traditional and hybrid models
    trained_models : dict
        Dictionary of trained models by type
    X_traditional : array
        Traditional features for training
    X_hybrid : array
        Hybrid features for training
    y_train : array
        Training target ratings
    predictions_unwatched : pd.DataFrame
        Predictions for unwatched dramas
        
    Notes
    -----
    Displays model performance metrics, feature importances, permutation importance,
    and attempts SHAP analysis. SHAP analysis is wrapped in try-catch to handle
    potential dimension mismatch errors gracefully without stopping execution.
    """
    evaluator.display_results(evaluation_results)
    evaluator.save_loocv_predictions(watched_dramas, loocv_predictions)

    print("\nFeature Importances:")
    evaluator.display_feature_importances(evaluation_results, feature_names_dict)

    print("\nPermutation Importance:")
    X_dict_for_eval = {'traditional': X_traditional, 'hybrid': X_hybrid}
    evaluator.display_permutation_importance(trained_models, X_dict_for_eval, y_train, feature_names_dict)

    # Run SHAP Explainer with error handling
    try:
        X_df_dict_for_shap = {
            'traditional': pd.DataFrame(X_traditional, columns=feature_names_dict['traditional']),
            'hybrid': pd.DataFrame(X_hybrid, columns=feature_names_dict['hybrid'])
        }
        shap_explainer = ShapExplainer(trained_models, X_df_dict_for_shap, feature_names_dict)
        shap_explainer.explain_models()
    except ValueError as e:
        print(f"\n⚠️  SHAP Analysis skipped due to dimension mismatch: {e}")
        print("This is likely due to feature engineering differences between training and prediction.")
        print("The predictions are still working correctly.")

def display_top_predictions(predictions_unwatched):
    """
    Display top 10 predictions from each model type.
    
    Parameters
    ----------
    predictions_unwatched : pd.DataFrame
        DataFrame containing predictions for unwatched dramas with columns:
        Traditional_Ensemble, BERT_Ensemble, Final_Prediction, Drama_Title
        
    Notes
    -----
    Shows the highest-rated drama recommendations from traditional ensemble,
    BERT ensemble, and final prediction models. Provides a quick overview
    of model differences and top recommendations for the user.
    """
    print("\nTop 10 Predictions by Model Type:")
    print("-" * 80)
    
    prediction_columns = ['Traditional_Ensemble', 'BERT_Ensemble', 'Final_Prediction']
    
    for col in prediction_columns:
        print(f"\n{col}:")
        top_10 = predictions_unwatched.nlargest(10, col)
        for _, row in top_10.iterrows():
            print(f"  {row['Drama_Title']:<40} {row[col]:<8.2f}")

def save_caches(feature_config, feature_engineer):
    """
    Save BERT and semantic similarity caches for future runs.
    
    Parameters
    ----------
    feature_config : dict
        Feature configuration dictionary with cache settings
    feature_engineer : FeatureEngineer
        Feature engineering instance containing extractors with caches
        
    Notes
    -----
    Conditionally saves caches based on feature configuration to improve
    performance in subsequent runs. Only saves caches for enabled features
    to avoid unnecessary disk usage.
    """
    if feature_config['use_bert']:
        feature_engineer.bert_extractor.save_cache()
    
    if feature_config['use_semantic_similarity']:
        feature_engineer.semantic_extractor.save_cache()

def main():
    """
    Main execution function for the drama rating prediction system.
    
    Orchestrates the complete workflow from argument parsing through model
    training, prediction generation, and result analysis. Saves final
    predictions to CSV and caches for future performance improvements.
    
    Notes
    -----
    This function represents the main pipeline:
    1. Parse command line arguments and create feature configuration
    2. Initialize all system components
    3. Load and prepare training/prediction data
    4. Create features for both datasets
    5. Train models and generate predictions
    6. Save results and display comprehensive analysis
    7. Save caches for future runs
    
    The function handles all coordination between components and provides
    user feedback throughout the process.
    """
    args = parse_arguments()
    feature_config = create_feature_config(args)

    # Initialize all components
    (data_loader, text_processor, feature_engineer, 
     model_trainer, predictor, evaluator) = initialize_components(feature_config)
    
    # Load and prepare data
    watched_dramas, unwatched_dramas = load_and_prepare_data(data_loader, args.user_id)
    
    # Create features for training and prediction
    (X_traditional, X_hybrid, y_train, 
     X_predict_traditional, X_predict_hybrid, 
     feature_names_dict) = create_features(feature_engineer, text_processor, 
                                          watched_dramas, unwatched_dramas)
    
    # Train models and generate predictions
    (trained_models, evaluation_results, 
     loocv_predictions, predictions_unwatched) = train_and_predict(
        model_trainer, predictor, X_traditional, X_hybrid, y_train,
        X_predict_traditional, X_predict_hybrid, unwatched_dramas
    )
    
    # Save predictions to file
    print("Saving results...")
    predictions_unwatched.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")
    
    # Display and analyze results
    display_and_analyze_results(evaluator, evaluation_results, loocv_predictions,
                               watched_dramas, feature_names_dict, trained_models,
                               X_traditional, X_hybrid, y_train, predictions_unwatched)
    
    # Display top predictions
    display_top_predictions(predictions_unwatched)
    
    # Save caches for future runs
    save_caches(feature_config, feature_engineer)

if __name__ == "__main__":
    main()