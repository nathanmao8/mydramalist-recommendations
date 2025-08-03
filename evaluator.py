# evaluator.py
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from sklearn.inspection import permutation_importance
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConstants:
    """Constants used throughout the evaluation process."""
    
    # Display formatting
    SEPARATOR_WIDTH: int = 80
    SEPARATOR_CHAR: str = "="
    SUB_SEPARATOR_CHAR: str = "-"
    
    # Metrics
    DEFAULT_TOP_N_FEATURES: int = 20
    DEFAULT_TOP_N_PERMUTATION: int = 15
    DEFAULT_HIGHLIGHTS: int = 5
    
    # Column formatting
    MODEL_COL_WIDTH: int = 25
    METRIC_COL_WIDTH: int = 8
    PRECISION_COL_WIDTH: int = 15
    TIME_COL_WIDTH: int = 10
    
    # File settings
    DEFAULT_PREDICTIONS_FILE: str = 'loocv_predictions_watched.csv'
    DECIMAL_PLACES: int = 2
    DISPLAY_DECIMAL_PLACES: int = 3
    
    # Permutation importance settings
    PERMUTATION_REPEATS: int = 10
    RANDOM_STATE: int = 42


@dataclass
class ModelConfig:
    """Configuration for model types and their display names."""
    
    MODELS: List[Tuple[str, str]] = None
    MODEL_KEYS: List[str] = None
    
    def __post_init__(self):
        """
        Initialize default model configurations if not provided.
        
        Sets up default model types and keys for evaluation if they
        weren't specified during instantiation.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        if self.MODELS is None:
            self.MODELS = [
                ('RF_Traditional', 'Random Forest (Trad)'),
                ('SVM_Traditional', 'SVM (Traditional)'),
                ('RF_BERT', 'Random Forest (BERT)'),
                ('SVM_BERT', 'SVM (BERT)')
            ]
        
        if self.MODEL_KEYS is None:
            self.MODEL_KEYS = ['RF_Traditional', 'SVM_Traditional', 'RF_BERT', 'SVM_BERT']


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class PredictionSaver:
    """Handles saving and displaying LOOCV predictions."""
    
    def __init__(self, constants: EvaluationConstants = None):
        """
        Initialize PredictionSaver with evaluation constants.
        
        Parameters
        ----------
        constants : EvaluationConstants, optional
            Configuration constants for evaluation. If None, uses default
            EvaluationConstants.
        """
        self.constants = constants or EvaluationConstants()
    
    def save_loocv_predictions(
        self, 
        dramas: List[Dict], 
        loocv_predictions: Dict, 
        filename: Optional[str] = None
    ) -> bool:
        """
        Create and save a DataFrame of LOOCV predictions for watched dramas.
        
        Validates input data, creates a structured DataFrame with predictions
        and errors, saves to CSV, and displays prediction highlights.
        
        Parameters
        ----------
        dramas : List[Dict]
            The original list of watched drama data containing titles and metadata.
        loocv_predictions : Dict
            Dictionary containing true ratings and model predictions with keys
            like 'true_ratings', 'rf_traditional_preds', etc.
        filename : str, optional
            Output CSV filename. If None, uses default from constants.
            
        Returns
        -------
        bool
            True if successful, False if an error occurred.
            
        Raises
        ------
        ValidationError
            If input data validation fails.
        """
        filename = filename or self.constants.DEFAULT_PREDICTIONS_FILE
        
        try:
            self._validate_prediction_data(dramas, loocv_predictions)
            df = self._create_predictions_dataframe(dramas, loocv_predictions)
            self._save_dataframe(df, filename)
            self._display_prediction_highlights(df)
            return True
            
        except (ValidationError, Exception) as e:
            logger.error(f"Error saving LOOCV predictions: {e}")
            return False
    
    def _validate_prediction_data(self, dramas: List[Dict], loocv_predictions: Dict) -> None:
        """
        Validate input data for predictions.
        
        Checks that dramas list is not empty, loocv_predictions has required
        structure, and that the lengths match between dramas and predictions.
        
        Parameters
        ----------
        dramas : List[Dict]
            List of drama dictionaries to validate.
        loocv_predictions : Dict
            Dictionary of predictions to validate.
            
        Returns
        -------
        None
        
        Raises
        ------
        ValidationError
            If validation fails for any reason.
        """
        if not dramas:
            raise ValidationError("Dramas list is empty")
        
        if not loocv_predictions or 'true_ratings' not in loocv_predictions:
            raise ValidationError("Invalid loocv_predictions structure")
        
        if len(dramas) != len(loocv_predictions['true_ratings']):
            raise ValidationError(
                f"Mismatch between dramas ({len(dramas)}) and predictions "
                f"({len(loocv_predictions['true_ratings'])})"
            )
    
    def _create_predictions_dataframe(self, dramas: List[Dict], loocv_predictions: Dict) -> pd.DataFrame:
        """
        Create DataFrame with predictions and calculated errors.
        
        Constructs a structured DataFrame containing drama titles, actual ratings,
        model predictions, and calculated prediction errors.
        
        Parameters
        ----------
        dramas : List[Dict]
            List of drama dictionaries containing title information.
        loocv_predictions : Dict
            Dictionary containing predictions from different models.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns for titles, ratings, predictions, and errors.
        """
        df_data = {
            'Drama_Title': [d.get('title', 'Unknown') for d in dramas],
            'Actual_Rating': loocv_predictions['true_ratings'],
            'RF_Traditional_Pred': loocv_predictions.get('rf_traditional_preds', []),
            'SVM_Traditional_Pred': loocv_predictions.get('svm_traditional_preds', []),
            'RF_BERT_Pred': loocv_predictions.get('rf_bert_preds', []),
            'SVM_BERT_Pred': loocv_predictions.get('svm_bert_preds', [])
        }
        
        df = pd.DataFrame(df_data)
        
        # Calculate prediction errors
        prediction_cols = [col for col in df.columns if col.endswith('_Pred')]
        for col in prediction_cols:
            error_col = col.replace('_Pred', '_Error')
            df[error_col] = df[col] - df['Actual_Rating']
        
        # Round float columns
        float_cols = [col for col in df.columns if any(keyword in col for keyword in ['Pred', 'Error', 'Rating'])]
        df[float_cols] = df[float_cols].round(self.constants.DECIMAL_PLACES)
        
        return df
    
    def _save_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save DataFrame to CSV file.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save.
        filename : str
            Target filename for the CSV file.
            
        Returns
        -------
        None
        """
        logger.info(f"Saving LOOCV predictions to {filename}...")
        df.to_csv(filename, index=False)
        logger.info(f"Successfully saved {len(df)} predictions.")
    
    def _display_prediction_highlights(self, df: pd.DataFrame) -> None:
        """
        Display prediction highlights for the best performing model.
        
        Shows highest and lowest predictions, as well as largest over-predictions
        and under-predictions to help understand model behavior.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing predictions and errors.
            
        Returns
        -------
        None
        """
        best_model_col = 'RF_BERT_Pred'  # Could be made configurable
        best_error_col = 'RF_BERT_Error'
        
        if best_model_col not in df.columns:
            logger.warning(f"Column {best_model_col} not found in predictions")
            return
        
        print(f"\n--- LOOCV Prediction Highlights ({best_model_col.replace('_Pred', '')} Model) ---")
        
        highlights = self.constants.DEFAULT_HIGHLIGHTS
        display_cols = ['Drama_Title', 'Actual_Rating', best_model_col]
        error_display_cols = display_cols + [best_error_col]
        
        print(f"\nHighest Predictions:")
        print(df.nlargest(highlights, best_model_col)[display_cols])
        
        print(f"\nLowest Predictions:")
        print(df.nsmallest(highlights, best_model_col)[display_cols])
        
        print(f"\nLargest Over-predictions (Model was too optimistic):")
        print(df.nlargest(highlights, best_error_col)[error_display_cols])
        
        print(f"\nLargest Under-predictions (Model was too pessimistic):")
        print(df.nsmallest(highlights, best_error_col)[error_display_cols])


class ResultsFormatter:
    """Handles formatting and displaying evaluation results."""
    
    def __init__(self, constants: EvaluationConstants = None, model_config: ModelConfig = None):
        """
        Initialize ResultsFormatter with constants and model configuration.
        
        Parameters
        ----------
        constants : EvaluationConstants, optional
            Configuration constants for evaluation. If None, uses default
            EvaluationConstants.
        model_config : ModelConfig, optional
            Model configuration settings. If None, uses default ModelConfig.
        """
        self.constants = constants or EvaluationConstants()
        self.model_config = model_config or ModelConfig()
    
    def display_comprehensive_results(self, evaluation_results: Dict) -> None:
        """
        Display comprehensive evaluation results for all models.
        
        Shows header, comparison table, feature analysis, and recommendations
        in a structured format.
        
        Parameters
        ----------
        evaluation_results : Dict
            Dictionary containing evaluation metrics for all models.
            
        Returns
        -------
        None
        """
        self._print_header("COMPREHENSIVE MODEL EVALUATION RESULTS")
        self.create_comparison_table(evaluation_results)
        self.display_feature_analysis()
        self.display_recommendations(evaluation_results)
    
    def create_comparison_table(self, results: Dict) -> None:
        """
        Create and display model performance comparison table.
        
        Formats and displays a table comparing MAP, Precision@K metrics,
        and training time across all models.
        
        Parameters
        ----------
        results : Dict
            Dictionary containing evaluation metrics for all models.
            
        Returns
        -------
        None
        """
        print("\nModel Performance Comparison:")
        print(self.constants.SUB_SEPARATOR_CHAR * self.constants.SEPARATOR_WIDTH)
        
        # Header
        headers = ['Model', 'MAP@10%', 'MAP@25%', 'MAP@33%', 'Precision@10%', 'Precision@25%', 'Precision@33%']
        widths = [
            self.constants.MODEL_COL_WIDTH,
            self.constants.METRIC_COL_WIDTH,
            self.constants.METRIC_COL_WIDTH,
            self.constants.METRIC_COL_WIDTH,
            self.constants.METRIC_COL_WIDTH,
            self.constants.METRIC_COL_WIDTH,
            self.constants.METRIC_COL_WIDTH
        ]
        
        header_line = "".join(f"{header:<{width}}" for header, width in zip(headers, widths))
        print(header_line)
        print(self.constants.SUB_SEPARATOR_CHAR * self.constants.SEPARATOR_WIDTH)
        
        # Model rows
        for model_key, model_name in self.model_config.MODELS:
            metrics = self._extract_model_metrics(results, model_key)
            row = self._format_model_row(model_name, metrics, widths)
            print(row)
    
    def _extract_model_metrics(self, results: Dict, model_key: str) -> Dict[str, float]:
        """
        Extract performance metrics for a specific model.
        
        Parameters
        ----------
        results : Dict
            Dictionary containing all model evaluation results.
        model_key : str
            Key identifying the specific model (e.g., 'RF_Traditional').
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing extracted metrics (map, precision_10, precision_25, precision_33, time).
        """
        return {
            'map_10': results.get(f'{model_key}_MAP_10%', 0),
            'map_25': results.get(f'{model_key}_MAP_25%', 0),
            'map_33': results.get(f'{model_key}_MAP_33%', 0),
            'precision_10': results.get(f'{model_key}_Precision_at_10%', 0),
            'precision_25': results.get(f'{model_key}_Precision_at_25%', 0),
            'precision_33': results.get(f'{model_key}_Precision_at_33%', 0),
            'time': results.get(f'{model_key}_TrainingTime', 0)
        }
    
    def _format_model_row(self, model_name: str, metrics: Dict[str, float], widths: List[int]) -> str:
        """
        Format a single model row for the comparison table.
        
        Parameters
        ----------
        model_name : str
            Display name of the model.
        metrics : Dict[str, float]
            Dictionary containing model metrics.
        widths : List[int]
            List of column widths for formatting.
            
        Returns
        -------
        str
            Formatted string representing the model row.
        """
        values = [
            model_name,
            f"{metrics['map_10']:.{self.constants.DISPLAY_DECIMAL_PLACES}f}",
            f"{metrics['map_25']:.{self.constants.DISPLAY_DECIMAL_PLACES}f}",
            f"{metrics['map_33']:.{self.constants.DISPLAY_DECIMAL_PLACES}f}",
            f"{metrics['precision_10']:.{self.constants.DISPLAY_DECIMAL_PLACES}f}",
            f"{metrics['precision_25']:.{self.constants.DISPLAY_DECIMAL_PLACES}f}",
            f"{metrics['precision_33']:.{self.constants.DISPLAY_DECIMAL_PLACES}f}"
        ]
        
        return "".join(f"{value:<{width}}" for value, width in zip(values, widths))
    
    def display_feature_analysis(self) -> None:
        """
        Display general feature importance analysis information.
        
        Shows information about different feature types and their characteristics
        in a structured format.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        """
        self._print_header("FEATURE ANALYSIS")
        
        print("\nFeature Types:")
        print("• Traditional Features: TF-IDF + Categorical + Sentiment")
        print("• BERT Features: 768-dimensional contextual embeddings")
        print("• Hybrid Features: Traditional + BERT combined")
        
        print("\nKey Insights:")
        print("• Traditional models excel at explicit preferences (cast, genre)")
        print("• BERT models capture nuanced textual patterns")
        print("• Hybrid approach leverages both strengths")
    
    def display_recommendations(self, results: Dict) -> None:
        """
        Display model recommendations based on performance metrics.
        
        Identifies and displays the best performing models for different
        ranking metrics (MAP and Precision@K).
        
        Parameters
        ----------
        results : Dict
            Dictionary containing evaluation metrics for all models.
            
        Returns
        -------
        None
        """
        self._print_header("RECOMMENDATIONS")
        
        best_models = self._find_best_models(results)
        
        print(f"• Best MAP@10%: {best_models['map_10']['name']} "
              f"({best_models['map_10']['score']:.{self.constants.DISPLAY_DECIMAL_PLACES}f})")
        print(f"• Best MAP@25%: {best_models['map_25']['name']} "
              f"({best_models['map_25']['score']:.{self.constants.DISPLAY_DECIMAL_PLACES}f})")
        print(f"• Best MAP@33%: {best_models['map_33']['name']} "
              f"({best_models['map_33']['score']:.{self.constants.DISPLAY_DECIMAL_PLACES}f})")
        print(f"• Best Precision@10%: {best_models['precision_10']['name']} "
              f"({best_models['precision_10']['score']:.{self.constants.DISPLAY_DECIMAL_PLACES}f})")
        print(f"• Best Precision@25%: {best_models['precision_25']['name']} "
              f"({best_models['precision_25']['score']:.{self.constants.DISPLAY_DECIMAL_PLACES}f})")
        print(f"• Best Precision@33%: {best_models['precision_33']['name']} "
              f"({best_models['precision_33']['score']:.{self.constants.DISPLAY_DECIMAL_PLACES}f})")
    
    def _find_best_models(self, results: Dict) -> Dict[str, Dict[str, Any]]:
        """
        Find the best performing models for different metrics.
        
        Iterates through all models to find the highest performing ones
        for MAP and Precision@K metrics.
        
        Parameters
        ----------
        results : Dict
            Dictionary containing evaluation metrics for all models.
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with 'map', 'precision_10', 'precision_25', 'precision_33' keys, 
            each containing the best model's name and score.
        """
        best_map_10 = {'score': 0, 'name': ''}
        best_map_25 = {'score': 0, 'name': ''}
        best_map_33 = {'score': 0, 'name': ''}
        best_precision_10 = {'score': 0, 'name': ''}
        best_precision_25 = {'score': 0, 'name': ''}
        best_precision_33 = {'score': 0, 'name': ''}
        
        for model in self.model_config.MODEL_KEYS:
            map_10 = results.get(f'{model}_MAP_10%', 0)
            map_25 = results.get(f'{model}_MAP_25%', 0)
            map_33 = results.get(f'{model}_MAP_33%', 0)
            precision_10 = results.get(f'{model}_Precision_at_10%', 0)
            precision_25 = results.get(f'{model}_Precision_at_25%', 0)
            precision_33 = results.get(f'{model}_Precision_at_33%', 0)
            
            if map_10 > best_map_10['score']:
                best_map_10 = {'score': map_10, 'name': model}
            
            if map_25 > best_map_25['score']:
                best_map_25 = {'score': map_25, 'name': model}
            
            if map_33 > best_map_33['score']:
                best_map_33 = {'score': map_33, 'name': model}
            
            if precision_10 > best_precision_10['score']:
                best_precision_10 = {'score': precision_10, 'name': model}
            
            if precision_25 > best_precision_25['score']:
                best_precision_25 = {'score': precision_25, 'name': model}
            
            if precision_33 > best_precision_33['score']:
                best_precision_33 = {'score': precision_33, 'name': model}
        
        return {
            'map_10': best_map_10,
            'map_25': best_map_25,
            'map_33': best_map_33,
            'precision_10': best_precision_10, 
            'precision_25': best_precision_25,
            'precision_33': best_precision_33
        }
    
    def _print_header(self, title: str) -> None:
        """
        Print a formatted header with title and separators.
        
        Parameters
        ----------
        title : str
            Title text to display in the header.
            
        Returns
        -------
        None
        """
        print(f"\n{self.constants.SEPARATOR_CHAR * self.constants.SEPARATOR_WIDTH}")
        print(title)
        print(self.constants.SEPARATOR_CHAR * self.constants.SEPARATOR_WIDTH)


class FeatureAnalyzer:
    """Handles feature importance analysis and display."""
    
    def __init__(self, constants: EvaluationConstants = None):
        """
        Initialize FeatureAnalyzer with evaluation constants.
        
        Parameters
        ----------
        constants : EvaluationConstants, optional
            Configuration constants for evaluation. If None, uses default
            EvaluationConstants.
        """
        self.constants = constants or EvaluationConstants()
    
    def display_feature_importances(
        self, 
        results: Dict, 
        feature_names: Dict, 
        top_n: Optional[int] = None
    ) -> None:
        """
        Display the top N most important features for Random Forest models.
        
        Extracts feature importance data from results and displays them
        in a ranked format for each Random Forest model.
        
        Parameters
        ----------
        results : Dict
            Dictionary containing evaluation results including feature importances.
        feature_names : Dict
            Dictionary mapping model types to their feature name lists.
        top_n : int, optional
            Number of top features to display. If None, uses default from constants.
            
        Returns
        -------
        None
        """
        top_n = top_n or self.constants.DEFAULT_TOP_N_FEATURES
        
        print(f"\n{self.constants.SEPARATOR_CHAR * self.constants.SEPARATOR_WIDTH}")
        print("FEATURE IMPORTANCE ANALYSIS (from Random Forest)")
        print(self.constants.SEPARATOR_CHAR * self.constants.SEPARATOR_WIDTH)
        
        importances_data = results.get('feature_importances', {})
        if not importances_data:
            logger.warning("Feature importances not found in results.")
            return
        
        for model_name, importances in importances_data.items():
            self._display_model_feature_importance(model_name, importances, feature_names, top_n)
    
    def display_permutation_importance(
        self, 
        models: Dict, 
        X: Dict, 
        y: np.ndarray, 
        feature_names: Dict, 
        top_n: Optional[int] = None
    ) -> None:
        """
        Calculate and display permutation importance for all models.
        
        Computes model-agnostic permutation importance by shuffling features
        and measuring the decrease in model performance.
        
        Parameters
        ----------
        models : Dict
            Dictionary containing trained models keyed by model name.
        X : Dict
            Dictionary containing feature matrices for different model types.
        y : np.ndarray
            Target values (true ratings).
        feature_names : Dict
            Dictionary mapping model types to their feature name lists.
        top_n : int, optional
            Number of top features to display. If None, uses default from constants.
            
        Returns
        -------
        None
        """
        top_n = top_n or self.constants.DEFAULT_TOP_N_PERMUTATION
        
        print(f"\n{self.constants.SEPARATOR_CHAR * self.constants.SEPARATOR_WIDTH}")
        print("PERMUTATION IMPORTANCE ANALYSIS (Model-Agnostic)")
        print(self.constants.SEPARATOR_CHAR * self.constants.SEPARATOR_WIDTH)
        
        for model_name, model in models.items():
            self._calculate_and_display_permutation_importance(
                model_name, model, X, y, feature_names, top_n
            )
    
    def _display_model_feature_importance(
        self, 
        model_name: str, 
        importances: List[float], 
        feature_names: Dict, 
        top_n: int
    ) -> None:
        """
        Display feature importance for a specific model.
        
        Creates a DataFrame with feature names and importance scores,
        sorts by importance, and displays the top N features.
        
        Parameters
        ----------
        model_name : str
            Name of the model for which to display feature importance.
        importances : List[float]
            List of feature importance scores.
        feature_names : Dict
            Dictionary mapping model types to their feature name lists.
        top_n : int
            Number of top features to display.
            
        Returns
        -------
        None
        """
        print(f"\n--- Top {top_n} Features for {model_name} ---")
        
        names = self._get_feature_names(model_name, feature_names)
        if not self._validate_feature_data(names, importances, model_name):
            return
        
        df_importances = pd.DataFrame({
            'feature': names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        print(df_importances.head(top_n))
    
    def _calculate_and_display_permutation_importance(
        self, 
        model_name: str, 
        model: Any, 
        X: Dict, 
        y: np.ndarray, 
        feature_names: Dict, 
        top_n: int
    ) -> None:
        """
        Calculate and display permutation importance for a specific model.
        
        Uses sklearn's permutation_importance to calculate feature importance
        by measuring the decrease in model score when feature values are randomly shuffled.
        
        Parameters
        ----------
        model_name : str
            Name of the model for analysis.
        model : Any
            Trained model object with predict method.
        X : Dict
            Dictionary containing feature matrices for different model types.
        y : np.ndarray
            Target values for evaluation.
        feature_names : Dict
            Dictionary mapping model types to their feature name lists.
        top_n : int
            Number of top features to display.
            
        Returns
        -------
        None
        """
        print(f"\n--- Top {top_n} Permutation Features for {model_name} ---")
        
        X_data, names = self._get_model_data(model_name, X, feature_names)
        if not self._validate_feature_data(names, X_data, model_name, is_matrix=True):
            return
        
        try:
            result = permutation_importance(
                model, X_data, y, 
                n_repeats=self.constants.PERMUTATION_REPEATS, 
                random_state=self.constants.RANDOM_STATE, 
                n_jobs=-1
            )
            
            df_perm = pd.DataFrame({
                'feature': names,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values(by='importance_mean', ascending=False)
            
            print(df_perm.head(top_n))
            
        except Exception as e:
            logger.error(f"Error calculating permutation importance for {model_name}: {e}")
    
    def _get_feature_names(self, model_name: str, feature_names: Dict) -> List[str]:
        """
        Get appropriate feature names based on model type.
        
        Determines whether to use traditional or hybrid feature names
        based on the model name.
        
        Parameters
        ----------
        model_name : str
            Name of the model to get feature names for.
        feature_names : Dict
            Dictionary mapping model types to their feature name lists.
            
        Returns
        -------
        List[str]
            List of feature names appropriate for the model type.
        """
        return (feature_names['traditional'] if 'traditional' in model_name.lower() 
                else feature_names.get('hybrid', []))
    
    def _get_model_data(self, model_name: str, X: Dict, feature_names: Dict) -> Tuple[np.ndarray, List[str]]:
        """
        Get appropriate data and feature names for a specific model.
        
        Returns the correct feature matrix and names based on whether
        the model uses traditional, BERT, or hybrid features.
        
        Parameters
        ----------
        model_name : str
            Name of the model to get data for.
        X : Dict
            Dictionary containing feature matrices for different model types.
        feature_names : Dict
            Dictionary mapping model types to their feature name lists.
            
        Returns
        -------
        Tuple[np.ndarray, List[str]]
            Tuple containing the feature matrix and corresponding feature names.
        """
        if 'traditional' in model_name.lower():
            return X['traditional'], feature_names['traditional']
        else:
            return X.get('hybrid', X.get('bert', np.array([]))), feature_names.get('hybrid', [])
    
    def _validate_feature_data(
        self, 
        names: List[str], 
        data: Any, 
        model_name: str, 
        is_matrix: bool = False
    ) -> bool:
        """
        Validate feature names and data consistency.
        
        Checks that the number of feature names matches the number of features
        in the data to ensure proper alignment.
        
        Parameters
        ----------
        names : List[str]
            List of feature names.
        data : Any
            Feature data (list of importances or numpy array).
        model_name : str
            Name of the model for error reporting.
        is_matrix : bool, optional
            Whether data is a matrix (True) or list (False). Default is False.
            
        Returns
        -------
        bool
            True if validation passes, False otherwise.
        """
        data_length = data.shape[1] if is_matrix else len(data)
        
        if len(names) != data_length:
            logger.warning(
                f"Feature names ({len(names)}) and data ({data_length}) have different lengths "
                f"for {model_name}. Skipping analysis."
            )
            return False
        return True


class ModelEvaluator:
    """Main evaluator class that orchestrates the evaluation process."""
    
    def __init__(
        self, 
        constants: EvaluationConstants = None,
        model_config: ModelConfig = None
    ):
        """
        Initialize ModelEvaluator with configuration objects.
        
        Sets up the main evaluator with constants, model configuration,
        and component classes for predictions, results formatting, and feature analysis.
        
        Parameters
        ----------
        constants : EvaluationConstants, optional
            Configuration constants for evaluation. If None, uses default
            EvaluationConstants.
        model_config : ModelConfig, optional
            Model configuration settings. If None, uses default ModelConfig.
        """
        self.constants = constants or EvaluationConstants()
        self.model_config = model_config or ModelConfig()
        
        self.prediction_saver = PredictionSaver(self.constants)
        self.results_formatter = ResultsFormatter(self.constants, self.model_config)
        self.feature_analyzer = FeatureAnalyzer(self.constants)
    
    def display_results(self, evaluation_results: Dict) -> None:
        """
        Display comprehensive evaluation results for all models.
        
        Delegates to the results formatter to show a complete evaluation
        summary including performance metrics, feature analysis, and recommendations.
        
        Parameters
        ----------
        evaluation_results : Dict
            Dictionary containing evaluation metrics for all models.
            
        Returns
        -------
        None
        """
        self.results_formatter.display_comprehensive_results(evaluation_results)
    
    def save_loocv_predictions(
        self, 
        dramas: List[Dict], 
        loocv_predictions: Dict, 
        filename: Optional[str] = None
    ) -> bool:
        """
        Save Leave-One-Out Cross-Validation predictions to file.
        
        Creates a structured DataFrame of predictions and saves it as CSV,
        with additional display of prediction highlights.
        
        Parameters
        ----------
        dramas : List[Dict]
            List of drama dictionaries containing metadata.
        loocv_predictions : Dict
            Dictionary containing true ratings and model predictions.
        filename : str, optional
            Output filename. If None, uses default filename.
            
        Returns
        -------
        bool
            True if save operation was successful, False otherwise.
        """
        return self.prediction_saver.save_loocv_predictions(dramas, loocv_predictions, filename)
    
    def display_feature_importances(
        self, 
        results: Dict, 
        feature_names: Dict, 
        top_n: Optional[int] = None
    ) -> None:
        """
        Display feature importance analysis for Random Forest models.
        
        Shows the most important features identified by Random Forest
        models, helping understand which features drive predictions.
        
        Parameters
        ----------
        results : Dict
            Dictionary containing evaluation results including feature importances.
        feature_names : Dict
            Dictionary mapping model types to their feature name lists.
        top_n : int, optional
            Number of top features to display. If None, uses default.
            
        Returns
        -------
        None
        """
        self.feature_analyzer.display_feature_importances(results, feature_names, top_n)
    
    def display_permutation_importance(
        self, 
        models: Dict, 
        X: Dict, 
        y: np.ndarray, 
        feature_names: Dict, 
        top_n: Optional[int] = None
    ) -> None:
        """
        Display permutation importance analysis for all models.
        
        Calculates and shows model-agnostic feature importance using
        permutation testing across all trained models.
        
        Parameters
        ----------
        models : Dict
            Dictionary containing trained models keyed by model name.
        X : Dict
            Dictionary containing feature matrices for different model types.
        y : np.ndarray
            Target values (true ratings).
        feature_names : Dict
            Dictionary mapping model types to their feature name lists.
        top_n : int, optional
            Number of top features to display. If None, uses default.
            
        Returns
        -------
        None
        """
        self.feature_analyzer.display_permutation_importance(models, X, y, feature_names, top_n)


# Backward compatibility - maintain the original Evaluator class interface
class Evaluator(ModelEvaluator):
    """
    Backward compatible evaluator class.
    
    Extends ModelEvaluator to maintain compatibility with existing code
    that expects the original Evaluator interface.
    """
    
    def create_comparison_table(self, results: Dict) -> None:
        """
        Create and display model performance comparison table.
        
        Backward compatible method that delegates to the results formatter
        to display performance metrics across all models.
        
        Parameters
        ----------
        results : Dict
            Dictionary containing evaluation metrics for all models.
            
        Returns
        -------
        None
        """
        self.results_formatter.create_comparison_table(results)
    
    def display_feature_analysis(self) -> None:
        """
        Display general feature importance analysis information.
        
        Backward compatible method that shows information about different
        feature types and their characteristics.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        """
        self.results_formatter.display_feature_analysis()
    
    def display_recommendations(self, results: Dict) -> None:
        """
        Display model recommendations based on performance metrics.
        
        Backward compatible method that identifies and displays the best
        performing models for different evaluation metrics.
        
        Parameters
        ----------
        results : Dict
            Dictionary containing evaluation metrics for all models.
            
        Returns
        -------
        None
        """
        self.results_formatter.display_recommendations(results)
