import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class PredictionConfig:
    """Configuration for prediction parameters and ensemble weights."""
    
    def __init__(self, ensemble_weights: Optional[Dict[str, float]] = None):
        """
        Initialize prediction configuration.

        Parameters
        ----------
        ensemble_weights : Dict[str, float], optional
            Weights for ensemble combination. Defaults to equal weights.
            
        Raises
        ------
        ValueError
            If ensemble_weights do not sum to 1.0.
        """
        self.ensemble_weights = ensemble_weights or {
            'traditional': 0.5,
            'bert': 0.5
        }
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate that ensemble weights sum to 1.0."""
        if abs(sum(self.ensemble_weights.values()) - 1.0) > 1e-6:
            raise ValueError("Ensemble weights must sum to 1.0")


class ModelPredictor:
    """Handles predictions from individual trained models."""
    
    MODEL_CONFIGS = [
        ('rf_traditional', 'traditional'),
        ('svm_traditional', 'traditional'),
        ('rf_bert', 'bert'),
        ('svm_bert', 'bert')
    ]
    
    @staticmethod
    def get_model_predictions(
        drama_features: Tuple[np.ndarray, np.ndarray],
        trained_models: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Generate predictions from all trained models.

        Parameters
        ----------
        drama_features : Tuple[np.ndarray, np.ndarray]
            (traditional_features, hybrid_features) arrays.
        trained_models : Dict[str, Any]
            Dictionary of trained model objects.
            
        Returns
        -------
        Dict[str, float]
            Predictions from each model with fallback values for missing models.
        """
        traditional_features, hybrid_features = drama_features
        feature_map = {'traditional': traditional_features, 'bert': hybrid_features}
        
        predictions = {}
        
        for model_name, feature_type in ModelPredictor.MODEL_CONFIGS:
            predictions[model_name] = ModelPredictor._get_single_prediction(
                model_name, trained_models.get(model_name), feature_map[feature_type]
            )
        
        return predictions
    
    @staticmethod
    def _get_single_prediction(
        model_name: str, 
        model: Any, 
        features: np.ndarray
    ) -> float:
        """
        Get prediction from a single model with error handling.
        
        Parameters
        ----------
        model_name : str
            Name of the model for logging.
        model : Any
            Trained model object.
        features : np.ndarray
            Feature array for prediction.
            
        Returns
        -------
        float
            Model prediction or 0.0 as fallback.
        """
        if model is None or not hasattr(model, 'predict'):
            logger.warning(f"Model {model_name} not available for prediction")
            return 0.0
        
        try:
            return model.predict(features)[0]
        except Exception as e:
            logger.warning(f"Failed to get prediction from {model_name}: {e}")
            return 0.0


class PredictionFormatter:
    """Handles formatting and standardization of prediction results."""
    
    @staticmethod
    def format_prediction_result(
        drama: Dict, 
        model_predictions: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Format prediction result with standardized structure.

        Parameters
        ----------
        drama : Dict
            Drama metadata dictionary.
        model_predictions : Dict[str, float]
            Individual model prediction values.
            
        Returns
        -------
        Dict[str, Any]
            Formatted prediction result with rounded values.
        """
        return {
            'Drama_Title': drama.get('title', ''),
            'Drama_ID': drama.get('slug', ''),
            'RF_Traditional': round(model_predictions['rf_traditional'], 2),
            'SVM_Traditional': round(model_predictions['svm_traditional'], 2),
            'RF_BERT': round(model_predictions['rf_bert'], 2),
            'SVM_BERT': round(model_predictions['svm_bert'], 2)
        }
    
    @staticmethod
    def create_fallback_prediction(drama: Dict) -> Dict[str, Any]:
        """
        Create fallback prediction when normal prediction fails.

        Parameters
        ----------
        drama : Dict
            Drama metadata dictionary.
            
        Returns
        -------
        Dict[str, Any]
            Fallback prediction with zero values.
        """
        return {
            'Drama_Title': drama.get('title', ''),
            'Drama_ID': drama.get('slug', ''),
            'RF_Traditional': 0.0,
            'SVM_Traditional': 0.0,
            'RF_BERT': 0.0,
            'SVM_BERT': 0.0
        }


class Predictor:
    """Main predictor class for drama recommendation system."""
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        """
        Initialize predictor with configuration.

        Parameters
        ----------
        config : PredictionConfig, optional
            Prediction configuration object.
        """
        self.config = config or PredictionConfig()
        self.model_predictor = ModelPredictor()
        self.formatter = PredictionFormatter()
    
    def predict_all_dramas(
        self, 
        dramas: List[Dict], 
        trained_models: Dict[str, Any], 
        X_traditional: np.ndarray, 
        X_hybrid: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate predictions for all dramas with individual model scores.

        Parameters
        ----------
        dramas : List[Dict]
            List of drama dictionaries.
        trained_models : Dict[str, Any]
            Dictionary of trained model objects.
        X_traditional : np.ndarray
            Traditional feature matrix.
        X_hybrid : np.ndarray
            Hybrid feature matrix.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with prediction results.
            
        Raises
        ------
        ValueError
            If drama count doesn't match feature matrix dimensions.
        """
        self._validate_input_dimensions(dramas, X_traditional, X_hybrid)
        
        predictions = []
        for i, drama in enumerate(dramas):
            prediction_result = self._predict_single_drama_safe(
                drama, trained_models, X_traditional[i:i+1], X_hybrid[i:i+1]
            )
            predictions.append(prediction_result)
        
        return pd.DataFrame(predictions)
    
    def _validate_input_dimensions(
        self, 
        dramas: List[Dict], 
        X_traditional: np.ndarray, 
        X_hybrid: np.ndarray
    ) -> None:
        """Validate that input dimensions match."""
        if len(dramas) != len(X_traditional) or len(dramas) != len(X_hybrid):
            raise ValueError("Number of dramas must match feature matrix dimensions")
    
    def _predict_single_drama_safe(
        self, 
        drama: Dict, 
        trained_models: Dict[str, Any],
        traditional_features: np.ndarray, 
        hybrid_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate prediction for a single drama with error handling.

        Parameters
        ----------
        drama : Dict
            Drama metadata dictionary.
        trained_models : Dict[str, Any]
            Dictionary of trained models.
        traditional_features : np.ndarray
            Traditional features for this drama.
        hybrid_features : np.ndarray
            Hybrid features for this drama.
            
        Returns
        -------
        Dict[str, Any]
            Prediction result or fallback prediction.
        """
        try:
            return self._predict_single_drama(
                drama, trained_models, traditional_features, hybrid_features
            )
        except Exception as e:
            logger.warning(f"Failed to predict for drama {drama.get('title', 'Unknown')}: {e}")
            return self.formatter.create_fallback_prediction(drama)
    
    def _predict_single_drama(
        self, 
        drama: Dict, 
        trained_models: Dict[str, Any],
        traditional_features: np.ndarray, 
        hybrid_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate prediction for a single drama.

        Parameters
        ----------
        drama : Dict
            Drama metadata dictionary.
        trained_models : Dict[str, Any]
            Dictionary of trained models.
        traditional_features : np.ndarray
            Traditional features for this drama.
        hybrid_features : np.ndarray
            Hybrid features for this drama.
            
        Returns
        -------
        Dict[str, Any]
            Formatted prediction result.
        """
        drama_features = (traditional_features, hybrid_features)
        model_predictions = self.model_predictor.get_model_predictions(
            drama_features, trained_models
        )
        
        return self.formatter.format_prediction_result(drama, model_predictions)
