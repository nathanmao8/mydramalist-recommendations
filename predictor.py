# predictor.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class PredictionConfig:
    """
    Configuration class for prediction parameters.
    
    This class manages configuration settings for the prediction system,
    including confidence levels and ensemble weights.
    
    Attributes
    ----------
    confidence_level : float
        The confidence level threshold for predictions (0 < value < 1).
    ensemble_weights : Dict[str, float]
        Weights for combining traditional and BERT model predictions.
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 ensemble_weights: Optional[Dict[str, float]] = None):
        """
        Initialize prediction configuration.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level threshold for predictions, by default 0.95.
            Must be between 0 and 1.
        ensemble_weights : Dict[str, float], optional
            Weights for ensemble combination, by default None.
            If None, uses equal weights of 0.5 for 'traditional' and 'bert'.
            
        Raises
        ------
        ValueError
            If confidence_level is not between 0 and 1, or if ensemble_weights
            do not sum to 1.0.
        """
        self.confidence_level = confidence_level
        self.ensemble_weights = ensemble_weights or {
            'traditional': 0.5,
            'bert': 0.5
        }
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Ensures that confidence_level is within valid range and ensemble_weights
        sum to 1.0.
        
        Raises
        ------
        ValueError
            If confidence_level is not between 0 and 1, or if ensemble_weights
            do not sum to 1.0 (within tolerance of 1e-6).
        """
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        
        if abs(sum(self.ensemble_weights.values()) - 1.0) > 1e-6:
            raise ValueError("Ensemble weights must sum to 1.0")

class ConfidenceCalculator:
    """
    Handles confidence score calculations.
    
    This class provides static methods for calculating confidence scores
    based on prediction variance and ensemble combinations.
    """
    
    @staticmethod
    def calculate_prediction_confidence(predictions: List[float]) -> float:
        """
        Calculate confidence score based on prediction variance.
        
        Uses inverse logistic function to map prediction variance to a confidence
        score. Lower variance indicates higher confidence in the predictions.

        Parameters
        ----------
        predictions : List[float]
            List of prediction values from different models.
            
        Returns
        -------
        float
            Confidence score between 0 and 1, where higher values indicate
            more confident predictions. Returns 0.5 for single predictions.
            
        Notes
        -----
        The confidence calculation uses the formula: 1 / (1 + variance)
        This provides an intuitive mapping where low variance yields high confidence.
        """
        if len(predictions) <= 1:
            return 0.5
        
        # Calculate variance of predictions
        variance = np.var(predictions)
        
        # Lower variance = higher confidence
        # Using inverse logistic function to map variance to confidence
        confidence_score = 1 / (1 + variance)
        
        return min(confidence_score, 1.0)
    
    @staticmethod
    def calculate_ensemble_confidence(traditional_conf: float, 
                                    bert_conf: float,
                                    weights: Dict[str, float]) -> float:
        """
        Calculate weighted ensemble confidence.
        
        Combines confidence scores from traditional and BERT models using
        the provided weights to produce a final ensemble confidence score.

        Parameters
        ----------
        traditional_conf : float
            Confidence score from traditional models (0 to 1).
        bert_conf : float
            Confidence score from BERT models (0 to 1).
        weights : Dict[str, float]
            Dictionary containing 'traditional' and 'bert' weights for
            ensemble combination.
            
        Returns
        -------
        float
            Weighted ensemble confidence score between 0 and 1.
            
        Examples
        --------
        >>> weights = {'traditional': 0.6, 'bert': 0.4}
        >>> calculate_ensemble_confidence(0.8, 0.9, weights)
        0.84
        """
        return (traditional_conf * weights['traditional'] + 
                bert_conf * weights['bert'])

class ModelPredictor:
    """
    Handles individual model predictions.
    
    This class provides methods for generating predictions from trained models
    using both traditional and hybrid (BERT-enhanced) features.
    """
    
    @staticmethod
    def get_model_predictions(drama_features: Tuple[np.ndarray, np.ndarray],
                            trained_models: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate predictions from all trained models.
        
        Uses both traditional and hybrid features to generate predictions from
        Random Forest and SVM models for both feature types.

        Parameters
        ----------
        drama_features : Tuple[np.ndarray, np.ndarray]
            Tuple containing (traditional_features, hybrid_features) arrays.
            Each array should have shape (1, n_features) for single drama prediction.
        trained_models : Dict[str, Any]
            Dictionary containing trained model objects with keys:
            'rf_traditional', 'svm_traditional', 'rf_bert', 'svm_bert'.
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing predictions from each model:
            - 'rf_traditional': Random Forest prediction on traditional features
            - 'svm_traditional': SVM prediction on traditional features  
            - 'rf_bert': Random Forest prediction on hybrid features
            - 'svm_bert': SVM prediction on hybrid features
            
        Raises
        ------
        ValueError
            If model keys are missing, features have wrong shape, or models
            cannot generate predictions.
            
        Examples
        --------
        >>> features = (np.array([[1, 2, 3]]), np.array([[1, 2, 3, 4, 5]]))
        >>> models = {'rf_traditional': rf_model, 'svm_traditional': svm_model, ...}
        >>> predictions = get_model_predictions(features, models)
        >>> print(predictions.keys())
        dict_keys(['rf_traditional', 'svm_traditional', 'rf_bert', 'svm_bert'])
        """
        traditional_features, hybrid_features = drama_features
        
        try:
            predictions = {
                'rf_traditional': trained_models['rf_traditional'].predict(traditional_features)[0],
                'svm_traditional': trained_models['svm_traditional'].predict(traditional_features)[0],
                'rf_bert': trained_models['rf_bert'].predict(hybrid_features)[0],
                'svm_bert': trained_models['svm_bert'].predict(hybrid_features)[0]
            }
            return predictions
            
        except (KeyError, IndexError, AttributeError) as e:
            logger.error(f"Error generating model predictions: {e}")
            raise ValueError(f"Invalid model or feature data: {e}")

class EnsembleCalculator:
    """
    Handles ensemble prediction calculations.
    
    This class provides methods for combining individual model predictions
    into ensemble predictions using various weighting strategies.
    """
    
    @staticmethod
    def calculate_ensemble_predictions(model_predictions: Dict[str, float],
                                     config: PredictionConfig) -> Dict[str, float]:
        """
        Calculate ensemble predictions from individual model outputs.
        
        Combines predictions from traditional and BERT models to create
        sub-ensemble predictions, then combines these using configuration weights
        to produce a final ensemble prediction.

        Parameters
        ----------
        model_predictions : Dict[str, float]
            Dictionary containing individual model predictions with keys:
            'rf_traditional', 'svm_traditional', 'rf_bert', 'svm_bert'.
        config : PredictionConfig
            Configuration object containing ensemble weights for combining
            traditional and BERT predictions.
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing ensemble predictions:
            - 'traditional_ensemble': Average of traditional model predictions
            - 'bert_ensemble': Average of BERT model predictions
            - 'final_prediction': Weighted combination of sub-ensembles
            
        Examples
        --------
        >>> model_preds = {'rf_traditional': 0.8, 'svm_traditional': 0.7, 
        ...                'rf_bert': 0.9, 'svm_bert': 0.85}
        >>> config = PredictionConfig()
        >>> ensembles = calculate_ensemble_predictions(model_preds, config)
        >>> print(ensembles['final_prediction'])
        0.8125
        """
        # Calculate sub-ensemble predictions
        traditional_ensemble = (
            model_predictions['rf_traditional'] + 
            model_predictions['svm_traditional']
        ) / 2
        
        bert_ensemble = (
            model_predictions['rf_bert'] + 
            model_predictions['svm_bert']
        ) / 2
        
        # Calculate weighted final ensemble
        final_prediction = (
            traditional_ensemble * config.ensemble_weights['traditional'] +
            bert_ensemble * config.ensemble_weights['bert']
        )
        
        return {
            'traditional_ensemble': traditional_ensemble,
            'bert_ensemble': bert_ensemble,
            'final_prediction': final_prediction
        }

class Predictor:
    """
    Main predictor class for drama recommendation system.
    
    This class orchestrates the prediction process by combining feature extraction,
    model prediction, ensemble calculation, and confidence scoring to generate
    comprehensive drama recommendations.
    
    Attributes
    ----------
    config : PredictionConfig
        Configuration settings for predictions.
    confidence_calculator : ConfidenceCalculator
        Calculator for confidence scores.
    model_predictor : ModelPredictor
        Handler for individual model predictions.
    ensemble_calculator : EnsembleCalculator
        Calculator for ensemble predictions.
    """
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        """
        Initialize predictor with configuration.

        Parameters
        ----------
        config : PredictionConfig, optional
            Prediction configuration object, by default None.
            If None, uses default PredictionConfig settings.
        """
        self.config = config or PredictionConfig()
        self.confidence_calculator = ConfidenceCalculator()
        self.model_predictor = ModelPredictor()
        self.ensemble_calculator = EnsembleCalculator()
    
    def predict_all_dramas(self, 
                          dramas: List[Dict], 
                          trained_models: Dict[str, Any], 
                          X_traditional: np.ndarray, 
                          X_hybrid: np.ndarray) -> pd.DataFrame:
        """
        Generate predictions for all unwatched dramas with both model types.
        
        Processes a list of dramas to generate comprehensive predictions including
        individual model scores, ensemble predictions, and confidence metrics.

        Parameters
        ----------
        dramas : List[Dict]
            List of drama dictionaries containing metadata (title, slug, etc.).
        trained_models : Dict[str, Any]
            Dictionary of trained model objects with keys for different model types.
        X_traditional : np.ndarray
            Traditional feature matrix with shape (n_dramas, n_traditional_features).
        X_hybrid : np.ndarray
            Hybrid feature matrix with shape (n_dramas, n_hybrid_features).
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing prediction results with columns for:
            - Drama metadata (title, ID)
            - Individual model predictions
            - Ensemble predictions  
            - Confidence scores
            
        Raises
        ------
        ValueError
            If the number of dramas doesn't match feature matrix dimensions.
            
        Notes
        -----
        Failed predictions for individual dramas are handled gracefully by
        creating fallback predictions with zero values and logging warnings.
        """
        if len(dramas) != len(X_traditional) or len(dramas) != len(X_hybrid):
            raise ValueError("Number of dramas must match feature matrix dimensions")
        
        predictions = []
        
        for i, drama in enumerate(dramas):
            try:
                prediction_result = self._predict_single_drama(
                    drama, trained_models, X_traditional[i:i+1], X_hybrid[i:i+1]
                )
                predictions.append(prediction_result)
                
            except Exception as e:
                logger.warning(f"Failed to predict for drama {drama.get('title', 'Unknown')}: {e}")
                predictions.append(self._create_fallback_prediction(drama))
        
        return pd.DataFrame(predictions)
    
    def _predict_single_drama(self, 
                            drama: Dict, 
                            trained_models: Dict[str, Any],
                            traditional_features: np.ndarray, 
                            hybrid_features: np.ndarray) -> Dict[str, Any]:
        """
        Generate prediction for a single drama.
        
        Coordinates the prediction process for a single drama by getting
        individual model predictions, calculating ensembles, and computing
        confidence scores.

        Parameters
        ----------
        drama : Dict
            Drama metadata dictionary containing at minimum 'title' and 'slug' keys.
        trained_models : Dict[str, Any]
            Dictionary of trained models for prediction generation.
        traditional_features : np.ndarray
            Traditional features for this drama with shape (1, n_features).
        hybrid_features : np.ndarray
            Hybrid features for this drama with shape (1, n_features).
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing comprehensive prediction results including:
            - Individual model predictions
            - Ensemble predictions
            - Confidence scores
            - Drama metadata
        """
        # Get predictions from all models
        drama_features = (traditional_features, hybrid_features)
        model_predictions = self.model_predictor.get_model_predictions(
            drama_features, trained_models
        )
        
        # Calculate ensemble predictions
        ensemble_predictions = self.ensemble_calculator.calculate_ensemble_predictions(
            model_predictions, self.config
        )
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(model_predictions)
        
        # Format final result
        return self._format_prediction_result(
            drama, model_predictions, ensemble_predictions, confidence_scores
        )
    
    def _calculate_confidence_scores(self, model_predictions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate confidence scores for different prediction types.
        
        Computes confidence scores for traditional models, BERT models, and
        overall ensemble based on prediction variance within each group.

        Parameters
        ----------
        model_predictions : Dict[str, float]
            Dictionary containing predictions from all individual models.
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing confidence scores:
            - 'traditional_confidence': Confidence for traditional models
            - 'bert_confidence': Confidence for BERT models  
            - 'overall_confidence': Weighted ensemble confidence
        """
        traditional_preds = [
            model_predictions['rf_traditional'], 
            model_predictions['svm_traditional']
        ]
        bert_preds = [
            model_predictions['rf_bert'], 
            model_predictions['svm_bert']
        ]
        
        traditional_confidence = self.confidence_calculator.calculate_prediction_confidence(traditional_preds)
        bert_confidence = self.confidence_calculator.calculate_prediction_confidence(bert_preds)
        overall_confidence = self.confidence_calculator.calculate_ensemble_confidence(
            traditional_confidence, bert_confidence, self.config.ensemble_weights
        )
        
        return {
            'traditional_confidence': traditional_confidence,
            'bert_confidence': bert_confidence,
            'overall_confidence': overall_confidence
        }
    
    def _format_prediction_result(self, 
                                drama: Dict, 
                                model_predictions: Dict[str, float],
                                ensemble_predictions: Dict[str, float],
                                confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Format the final prediction result.
        
        Combines all prediction components into a standardized output format
        with rounded values for display.

        Parameters
        ----------
        drama : Dict
            Drama metadata dictionary.
        model_predictions : Dict[str, float]
            Individual model prediction values.
        ensemble_predictions : Dict[str, float]
            Ensemble prediction values.
        confidence_scores : Dict[str, float]
            Confidence score values.
            
        Returns
        -------
        Dict[str, Any]
            Formatted prediction result with standardized column names and
            rounded numeric values (2 decimal places for predictions,
            3 decimal places for confidence scores).
        """
        return {
            'Drama_Title': drama.get('title', ''),
            'Drama_ID': drama.get('slug', ''),
            'RF_Traditional': round(model_predictions['rf_traditional'], 2),
            'SVM_Traditional': round(model_predictions['svm_traditional'], 2),
            'RF_BERT': round(model_predictions['rf_bert'], 2),
            'SVM_BERT': round(model_predictions['svm_bert'], 2),
            'Traditional_Ensemble': round(ensemble_predictions['traditional_ensemble'], 2),
            'BERT_Ensemble': round(ensemble_predictions['bert_ensemble'], 2),
            'Final_Prediction': round(ensemble_predictions['final_prediction'], 2),
            'Confidence_Score': round(confidence_scores['overall_confidence'], 3),
            'Traditional_Confidence': round(confidence_scores['traditional_confidence'], 3),
            'BERT_Confidence': round(confidence_scores['bert_confidence'], 3)
        }
    
    def _create_fallback_prediction(self, drama: Dict) -> Dict[str, Any]:
        """
        Create a fallback prediction when normal prediction fails.
        
        Generates a safe default prediction with zero values when the normal
        prediction process encounters errors. Preserves drama metadata while
        setting all numeric predictions to zero.

        Parameters
        ----------
        drama : Dict
            Drama metadata dictionary containing at minimum title and slug.
            
        Returns
        -------
        Dict[str, Any]
            Fallback prediction result with same structure as normal predictions
            but with all prediction values set to 0.0.
            
        Notes
        -----
        This method ensures the prediction pipeline continues to function even
        when individual drama predictions fail due to data issues or model errors.
        """
        return {
            'Drama_Title': drama.get('title', ''),
            'Drama_ID': drama.get('slug', ''),
            'RF_Traditional': 0.0,
            'SVM_Traditional': 0.0,
            'RF_BERT': 0.0,
            'SVM_BERT': 0.0,
            'Traditional_Ensemble': 0.0,
            'BERT_Ensemble': 0.0,
            'Final_Prediction': 0.0,
            'Confidence_Score': 0.0,
            'Traditional_Confidence': 0.0,
            'BERT_Confidence': 0.0
        }
    
    # Legacy method for backward compatibility
    def calculate_confidence_score(self, predictions: List[float]) -> float:
        """
        Legacy method for backward compatibility.
        
        Wrapper around the confidence calculator for existing code that uses
        the old interface.

        Parameters
        ----------
        predictions : List[float]
            List of prediction values from different models.
            
        Returns
        -------
        float
            Confidence score between 0 and 1.
            
        Notes
        -----
        This method is deprecated and maintained only for backward compatibility.
        New code should use ConfidenceCalculator.calculate_prediction_confidence
        directly.
        """
        return self.confidence_calculator.calculate_prediction_confidence(predictions)
