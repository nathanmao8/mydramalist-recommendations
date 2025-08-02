# model_trainer.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score
from typing import Dict, List, Tuple
import time

class ModelTrainer:
    """
    ModelTrainer handles model training with Leave-One-Out Cross-Validation and bootstrap sampling.
    
    Features:
    - Traditional and BERT-enhanced model training
    - Leave-One-Out Cross-Validation for robust evaluation
    - Bootstrap sampling for prediction stability
    - Comprehensive evaluation metrics
    - Configurable model parameters and training options
    
    Models:
    - Random Forest Regressor
    - Support Vector Machine Regressor
    """
    
    # Model configuration constants
    DEFAULT_RF_PARAMS = {
        'n_estimators': 100,
        'max_features': 'sqrt',
        'random_state': 42
    }
    
    DEFAULT_SVM_PARAMS = {
        'kernel': 'rbf',
        'gamma': 'scale'
    }
    
    # Training configuration
    DEFAULT_BOOTSTRAP_ITERATIONS = 100
    DEFAULT_HIGH_RATING_THRESHOLD = 7.5
    PROGRESS_UPDATE_INTERVAL = 10  # Update progress every N folds
    
    # Metric names
    REGRESSION_METRICS = ['RMSE', 'MAE', 'R2']
    CLASSIFICATION_METRICS = ['Precision_7.5+', 'Recall_7.5+', 'MAE_High_Rated']
    
    def __init__(self, rf_params: Dict = None, svm_params: Dict = None, 
                 bootstrap_iterations: int = None, threshold: float = None):
        """
        Initialize ModelTrainer with configurable parameters.
        
        Parameters
        ----------
        rf_params : Dict, optional
            Parameters for Random Forest models. Will be merged with DEFAULT_RF_PARAMS.
            Default is None.
        svm_params : Dict, optional
            Parameters for SVM models. Will be merged with DEFAULT_SVM_PARAMS.
            Default is None.
        bootstrap_iterations : int, optional
            Number of bootstrap iterations for prediction stability.
            Default is DEFAULT_BOOTSTRAP_ITERATIONS (100).
        threshold : float, optional
            Rating threshold for high-rating classification tasks.
            Default is DEFAULT_HIGH_RATING_THRESHOLD (7.5).
        """
        
        # Configuration
        self.rf_params = {**self.DEFAULT_RF_PARAMS, **(rf_params or {})}
        self.svm_params = {**self.DEFAULT_SVM_PARAMS, **(svm_params or {})}
        self.bootstrap_iterations = bootstrap_iterations or self.DEFAULT_BOOTSTRAP_ITERATIONS
        self.threshold = threshold or self.DEFAULT_HIGH_RATING_THRESHOLD
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Initialize all model instances with configured parameters.
        
        Creates Random Forest and SVM model instances for both traditional
        and BERT-enhanced feature sets using the configured parameters.
        
        Notes
        -----
        This method is called during initialization and when parameters are updated.
        """
        # Traditional models
        self.rf_traditional = RandomForestRegressor(**self.rf_params)
        self.svm_traditional = SVR(**self.svm_params)
        
        # BERT-enhanced models  
        self.rf_bert = RandomForestRegressor(**self.rf_params)
        self.svm_bert = SVR(**self.svm_params)
    
    def train_models(self, X_traditional: np.ndarray, X_hybrid: np.ndarray, y: np.ndarray) -> Tuple[Dict, Dict, Dict]:
        """
        Train both traditional and hybrid models with LOOCV and return LOOCV predictions.
        
        Trains Random Forest and SVM models on both traditional features and 
        BERT-enhanced features using Leave-One-Out Cross-Validation for robust
        evaluation.
        
        Parameters
        ----------
        X_traditional : np.ndarray, shape (n_samples, n_traditional_features)
            Traditional feature matrix.
        X_hybrid : np.ndarray, shape (n_samples, n_hybrid_features)
            BERT-enhanced feature matrix combining traditional and BERT features.
        y : np.ndarray, shape (n_samples,)
            Target rating values.
            
        Returns
        -------
        trained_models : Dict
            Dictionary containing all trained model instances with keys:
            'rf_traditional', 'svm_traditional', 'rf_bert', 'svm_bert'.
        all_results : Dict
            Comprehensive evaluation metrics for all models including
            feature importances.
        loocv_predictions : Dict
            LOOCV predictions for all models with keys: 'true_ratings',
            'rf_traditional_preds', 'svm_traditional_preds', 'rf_bert_preds',
            'svm_bert_preds'.
            
        Notes
        -----
        Models are trained using bootstrap sampling within LOOCV for
        improved prediction stability.
        """
        
        try:
            print("Training traditional models...")
            traditional_results, rf_trad_preds, svm_trad_preds, true_values = self.train_model_set(
                X_traditional, y, 
                self.rf_traditional, self.svm_traditional,
                model_prefix="Traditional"
            )
            
            print("Training BERT-enhanced models...")
            bert_results, rf_bert_preds, svm_bert_preds, _ = self.train_model_set(
                X_hybrid, y,
                self.rf_bert, self.svm_bert,
                model_prefix="BERT"
            )
            
            # Combine results
            all_results = {**traditional_results, **bert_results}
            
            print("Capturing feature importances from Random Forest models...")
            feature_importances = {
                'rf_traditional': self.rf_traditional.feature_importances_,
                'rf_bert': self.rf_bert.feature_importances_
            }
            
            # Add the captured importances to the results dictionary
            all_results['feature_importances'] = feature_importances

            # Create model dictionary for predictions
            trained_models = {
                'rf_traditional': self.rf_traditional,
                'svm_traditional': self.svm_traditional,
                'rf_bert': self.rf_bert,
                'svm_bert': self.svm_bert
            }
            
            # Create a dictionary to hold all LOOCV predictions
            loocv_predictions = {
                'true_ratings': true_values,
                'rf_traditional_preds': rf_trad_preds,
                'svm_traditional_preds': svm_trad_preds,
                'rf_bert_preds': rf_bert_preds,
                'svm_bert_preds': svm_bert_preds
            }

            return trained_models, all_results, loocv_predictions
            
        except Exception as e:
            print(f"Error during model training: {e}")
            # Return empty results in case of failure
            return {}, {}, {}
    
    def _perform_bootstrap_prediction(self, X_train_fold: np.ndarray, y_train_fold: np.ndarray, 
                                     X_test_fold: np.ndarray) -> Tuple[float, float]:
        """
        Perform bootstrap sampling and prediction for a single fold.
        
        Creates multiple bootstrap samples from the training data, trains models
        on each sample, and returns averaged predictions for improved stability.
        
        Parameters
        ----------
        X_train_fold : np.ndarray, shape (n_train_samples, n_features)
            Training feature matrix for the current fold.
        y_train_fold : np.ndarray, shape (n_train_samples,)
            Training target values for the current fold.
        X_test_fold : np.ndarray, shape (1, n_features)
            Test feature vector for the current fold (single sample in LOOCV).
            
        Returns
        -------
        rf_pred : float
            Averaged Random Forest prediction across bootstrap samples.
        svm_pred : float
            Averaged SVM prediction across bootstrap samples.
            
        Notes
        -----
        Uses self.bootstrap_iterations to determine number of bootstrap samples.
        Returns (0.0, 0.0) if an error occurs during prediction.
        """
        try:
            bootstrap_rf_preds = []
            bootstrap_svm_preds = []
            
            for _ in range(self.bootstrap_iterations):
                # Bootstrap sample
                n_samples = len(X_train_fold)
                bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap = X_train_fold[bootstrap_idx]
                y_bootstrap = y_train_fold[bootstrap_idx]
                
                # Train models on bootstrap sample
                rf_temp = RandomForestRegressor(**self.rf_params)
                svm_temp = SVR(**self.svm_params)
                
                rf_temp.fit(X_bootstrap, y_bootstrap)
                svm_temp.fit(X_bootstrap, y_bootstrap)
                
                # Predict on test sample
                bootstrap_rf_preds.append(rf_temp.predict(X_test_fold)[0])
                bootstrap_svm_preds.append(svm_temp.predict(X_test_fold)[0])
            
            # Return averaged bootstrap predictions
            return np.mean(bootstrap_rf_preds), np.mean(bootstrap_svm_preds)
            
        except Exception as e:
            print(f"Error in bootstrap prediction: {e}")
            # Return fallback predictions
            return 0.0, 0.0
    
    def _perform_loocv(self, X: np.ndarray, y: np.ndarray, model_prefix: str) -> Tuple[List, List, List]:
        """
        Perform Leave-One-Out Cross-Validation with bootstrap sampling.
        
        Iterates through all samples, using each as a test case while training
        on the remaining samples with bootstrap sampling for stability.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray, shape (n_samples,)
            Target rating values.
        model_prefix : str
            Prefix for progress messages (e.g., "Traditional", "BERT").
            
        Returns
        -------
        rf_predictions : List[float]
            Random Forest predictions for each test sample.
        svm_predictions : List[float]
            SVM predictions for each test sample.
        true_values : List[float]
            True rating values for each test sample.
            
        Notes
        -----
        Progress is printed every PROGRESS_UPDATE_INTERVAL folds.
        Returns empty lists if an error occurs.
        """
        try:
            loo = LeaveOneOut()
            
            rf_predictions = []
            svm_predictions = []
            true_values = []
            
            print(f"Performing LOOCV for {model_prefix} models...")
            
            for i, (train_idx, test_idx) in enumerate(loo.split(X)):
                if i % self.PROGRESS_UPDATE_INTERVAL == 0:
                    print(f"  Processing fold {i+1}/{len(X)}")
                
                X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]
                
                # Perform bootstrap prediction for this fold
                rf_pred, svm_pred = self._perform_bootstrap_prediction(
                    X_train_fold, y_train_fold, X_test_fold
                )
                
                rf_predictions.append(rf_pred)
                svm_predictions.append(svm_pred)
                true_values.append(y_test_fold[0])
            
            return rf_predictions, svm_predictions, true_values
            
        except Exception as e:
            print(f"Error in LOOCV: {e}")
            # Return empty lists in case of failure
            return [], [], []
    
    def _train_final_models(self, X: np.ndarray, y: np.ndarray, rf_model, svm_model, model_prefix: str):
        """
        Train final models on the complete dataset.
        
        After LOOCV evaluation, trains the final models on the entire dataset
        for future predictions.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Complete feature matrix.
        y : np.ndarray, shape (n_samples,)
            Complete target rating values.
        rf_model : RandomForestRegressor
            Random Forest model instance to train.
        svm_model : SVR
            SVM model instance to train.
        model_prefix : str
            Prefix for progress messages (e.g., "Traditional", "BERT").
            
        Notes
        -----
        Models are trained in-place. Error messages are printed if training fails.
        """
        try:
            print(f"Training final {model_prefix} models on full dataset...")
            rf_model.fit(X, y)
            svm_model.fit(X, y)
        except Exception as e:
            print(f"Error training final {model_prefix} models: {e}")

    def train_model_set(self, X: np.ndarray, y: np.ndarray, rf_model, svm_model, model_prefix: str) -> Tuple[Dict, list, list, list]:
        """
        Train a set of models using LOOCV, return evaluation results and predictions.
        
        Orchestrates the complete training process for a pair of models (RF and SVM)
        including LOOCV evaluation, metric calculation, and final model training.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray, shape (n_samples,)
            Target rating values.
        rf_model : RandomForestRegressor
            Random Forest model instance to train.
        svm_model : SVR
            SVM model instance to train.
        model_prefix : str
            Prefix for model identification (e.g., "Traditional", "BERT").
            
        Returns
        -------
        results : Dict
            Comprehensive evaluation metrics for both models.
        rf_predictions : List[float]
            Random Forest LOOCV predictions.
        svm_predictions : List[float]
            SVM LOOCV predictions.
        true_values : List[float]
            True rating values corresponding to predictions.
            
        Notes
        -----
        Returns empty results if an error occurs during training.
        """
        
        try:
            start_time = time.time()
            
            # Perform Leave-One-Out Cross-Validation
            rf_predictions, svm_predictions, true_values = self._perform_loocv(X, y, model_prefix)
            
            training_time = time.time() - start_time
            
            # Calculate evaluation metrics
            results = self.calculate_evaluation_metrics(
                true_values, rf_predictions, svm_predictions, model_prefix, training_time
            )
            
            # Train final models on full dataset
            self._train_final_models(X, y, rf_model, svm_model, model_prefix)
            
            return results, rf_predictions, svm_predictions, true_values
            
        except Exception as e:
            print(f"Error in train_model_set for {model_prefix}: {e}")
            # Return empty results in case of failure
            return {}, [], [], []
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics (RMSE, MAE, R2).
        
        Computes standard regression evaluation metrics for predicted vs true values.
        
        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples,)
            True target values.
        y_pred : np.ndarray, shape (n_samples,)
            Predicted target values.
            
        Returns
        -------
        metrics : Dict[str, float]
            Dictionary containing 'RMSE', 'MAE', and 'R2' metrics.
            Returns infinite/negative infinite values if calculation fails.
            
        Notes
        -----
        RMSE: Root Mean Squared Error
        MAE: Mean Absolute Error  
        R2: Coefficient of Determination
        """
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            return {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
        except Exception as e:
            print(f"Error calculating regression metrics: {e}")
            return {'RMSE': float('inf'), 'MAE': float('inf'), 'R2': -float('inf')}
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate classification metrics for high ratings.
        
        Converts the regression problem to binary classification using the
        configured threshold and computes precision and recall for high ratings.
        
        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples,)
            True target values.
        y_pred : np.ndarray, shape (n_samples,)
            Predicted target values.
            
        Returns
        -------
        metrics : Dict[str, float]
            Dictionary containing 'Precision_7.5+' and 'Recall_7.5+' metrics.
            Returns 0.0 values if calculation fails.
            
        Notes
        -----
        Uses self.threshold to determine high vs low ratings.
        Zero division is handled by returning 0.0 for precision/recall.
        """
        try:
            # Convert to binary classification problem
            y_true_binary = (y_true >= self.threshold).astype(int)
            y_pred_binary = (y_pred >= self.threshold).astype(int)
            
            # Calculate precision and recall
            precision_high = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall_high = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            
            return {
                'Precision_7.5+': precision_high,
                'Recall_7.5+': recall_high
            }
        except Exception as e:
            print(f"Error calculating classification metrics: {e}")
            return {'Precision_7.5+': 0.0, 'Recall_7.5+': 0.0}
    
    def _calculate_high_rated_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate MAE specifically for high-rated items.
        
        Filters predictions to include only items with true ratings above
        the threshold and computes MAE on this subset.
        
        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples,)
            True target values.
        y_pred : np.ndarray, shape (n_samples,)
            Predicted target values.
            
        Returns
        -------
        mae_high : float
            Mean Absolute Error for high-rated items only.
            Returns 0.0 if no high-rated items exist or if calculation fails.
            
        Notes
        -----
        Uses self.threshold to determine high-rated items.
        This metric helps evaluate performance specifically on highly-rated content.
        """
        try:
            # Create mask for high-rated items
            high_rated_mask = (y_true >= self.threshold)
            
            # Apply mask to get high-rated subset
            y_true_high_rated = y_true[high_rated_mask]
            y_pred_high_rated = y_pred[high_rated_mask]
            
            # Calculate MAE on filtered subset
            if len(y_true_high_rated) > 0:
                return mean_absolute_error(y_true_high_rated, y_pred_high_rated)
            else:
                return 0.0
        except Exception as e:
            print(f"Error calculating high-rated MAE: {e}")
            return 0.0
    
    def _calculate_single_model_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        """
        Calculate all metrics for a single model.
        
        Computes comprehensive evaluation metrics including regression metrics,
        classification metrics, and high-rated MAE for a single model.
        
        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples,)
            True target values.
        y_pred : np.ndarray, shape (n_samples,)
            Predicted target values.
        model_name : str
            Name prefix for the model (e.g., "RF_Traditional").
            
        Returns
        -------
        metrics : Dict[str, float]
            Dictionary containing all metrics with model_name prefixed keys:
            - {model_name}_RMSE, _MAE, _R2
            - {model_name}_Precision_7.5+, _Recall_7.5+
            - {model_name}_MAE_High_Rated
            
        Notes
        -----
        Combines results from regression, classification, and high-rated metrics.
        """
        metrics = {}
        
        # Regression metrics
        regression_metrics = self._calculate_regression_metrics(y_true, y_pred)
        for metric_name, value in regression_metrics.items():
            metrics[f'{model_name}_{metric_name}'] = value
        
        # Classification metrics
        classification_metrics = self._calculate_classification_metrics(y_true, y_pred)
        for metric_name, value in classification_metrics.items():
            metrics[f'{model_name}_{metric_name}'] = value
        
        # High-rated MAE
        mae_high = self._calculate_high_rated_mae(y_true, y_pred)
        metrics[f'{model_name}_MAE_High_Rated'] = mae_high
        
        return metrics

    def calculate_evaluation_metrics(self, true_values: List, rf_preds: List, svm_preds: List, 
                                   model_prefix: str, training_time: float) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Computes all evaluation metrics for both Random Forest and SVM models
        including training time information.
        
        Parameters
        ----------
        true_values : List[float]
            True target rating values.
        rf_preds : List[float]
            Random Forest predictions.
        svm_preds : List[float]
            SVM predictions.
        model_prefix : str
            Prefix for model identification (e.g., "Traditional", "BERT").
        training_time : float
            Total training time in seconds.
            
        Returns
        -------
        metrics : Dict[str, float]
            Combined metrics dictionary for both models including:
            - All regression and classification metrics for both models
            - Training time split between models
            Returns empty dict if calculation fails.
            
        Notes
        -----
        Training time is divided equally between RF and SVM models.
        """
        
        try:
            # Convert lists to NumPy arrays for easier manipulation
            true_values_np = np.array(true_values)
            rf_preds_np = np.array(rf_preds)
            svm_preds_np = np.array(svm_preds)

            # Calculate metrics for both models
            rf_metrics = self._calculate_single_model_metrics(
                true_values_np, rf_preds_np, f'RF_{model_prefix}'
            )
            svm_metrics = self._calculate_single_model_metrics(
                true_values_np, svm_preds_np, f'SVM_{model_prefix}'
            )

            # Add training time
            rf_metrics[f'RF_{model_prefix}_TrainingTime'] = training_time / 2
            svm_metrics[f'SVM_{model_prefix}_TrainingTime'] = training_time / 2
            
            return {**rf_metrics, **svm_metrics}
        
        except Exception as e:
            print(f"Error calculating evaluation metrics: {e}")
            return {}
    
    def get_configuration(self) -> Dict[str, any]:
        """
        Get current configuration settings for debugging and monitoring.
        
        Returns
        -------
        config : Dict[str, any]
            Dictionary containing all current configuration parameters:
            - rf_params: Random Forest parameters
            - svm_params: SVM parameters  
            - bootstrap_iterations: Number of bootstrap samples
            - threshold: High rating threshold
            - progress_update_interval: Progress reporting frequency
            - regression_metrics: List of regression metric names
            - classification_metrics: List of classification metric names
            
        Notes
        -----
        Returns copies of mutable parameters to prevent accidental modification.
        """
        return {
            'rf_params': self.rf_params.copy(),
            'svm_params': self.svm_params.copy(),
            'bootstrap_iterations': self.bootstrap_iterations,
            'threshold': self.threshold,
            'progress_update_interval': self.PROGRESS_UPDATE_INTERVAL,
            'regression_metrics': self.REGRESSION_METRICS,
            'classification_metrics': self.CLASSIFICATION_METRICS
        }
    
    def update_configuration(self, **kwargs):
        """
        Update configuration parameters and reinitialize models if needed.
        
        Updates the specified configuration parameters and reinitializes
        model instances if model parameters were changed.
        
        Parameters
        ----------
        **kwargs : dict
            Configuration parameters to update. Supported keys:
            - rf_params (Dict): Random Forest parameters
            - svm_params (Dict): SVM parameters
            - bootstrap_iterations (int): Number of bootstrap samples
            - threshold (float): High rating threshold
            
        Notes
        -----
        Model instances are automatically reinitialized if rf_params or 
        svm_params are updated. Parameters are merged with defaults.
        """
        model_params_changed = False
        
        if 'rf_params' in kwargs:
            self.rf_params = {**self.DEFAULT_RF_PARAMS, **kwargs['rf_params']}
            model_params_changed = True
            
        if 'svm_params' in kwargs:
            self.svm_params = {**self.DEFAULT_SVM_PARAMS, **kwargs['svm_params']}
            model_params_changed = True
            
        if 'bootstrap_iterations' in kwargs:
            self.bootstrap_iterations = kwargs['bootstrap_iterations']
            
        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']
        
        # Reinitialize models if parameters changed
        if model_params_changed:
            self._initialize_models()
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the initialized models.
        
        Returns
        -------
        model_info : Dict[str, any]
            Nested dictionary containing information about all model instances:
            - traditional_models: Info about RF and SVM traditional models
            - bert_models: Info about RF and SVM BERT-enhanced models
            Each model entry contains 'type' and 'params' keys.
            
        Notes
        -----
        Useful for debugging model configurations and verifying proper initialization.
        """
        return {
            'traditional_models': {
                'random_forest': {
                    'type': type(self.rf_traditional).__name__,
                    'params': self.rf_traditional.get_params()
                },
                'svm': {
                    'type': type(self.svm_traditional).__name__,
                    'params': self.svm_traditional.get_params()
                }
            },
            'bert_models': {
                'random_forest': {
                    'type': type(self.rf_bert).__name__,
                    'params': self.rf_bert.get_params()
                },
                'svm': {
                    'type': type(self.svm_bert).__name__,
                    'params': self.svm_bert.get_params()
                }
            }
        }
    
    def validate_configuration(self) -> Dict[str, any]:
        """
        Validate the current configuration and report any issues.
        
        Checks configuration parameters for common issues and provides
        warnings, errors, and recommendations for optimal performance.
        
        Returns
        -------
        validation_results : Dict[str, any]
            Dictionary containing validation results with keys:
            - is_valid (bool): Whether configuration is valid
            - warnings (List[str]): Non-critical issues
            - errors (List[str]): Critical configuration errors
            - recommendations (List[str]): Suggested improvements
            
        Notes
        -----
        Checks bootstrap iterations, threshold values, RF parameters,
        and reproducibility settings. Critical errors set is_valid to False.
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check bootstrap iterations
        if self.bootstrap_iterations < 10:
            validation_results['warnings'].append(
                f"Bootstrap iterations ({self.bootstrap_iterations}) is quite low - consider increasing for better stability"
            )
        elif self.bootstrap_iterations > 500:
            validation_results['warnings'].append(
                f"Bootstrap iterations ({self.bootstrap_iterations}) is very high - this may lead to long training times"
            )
        
        # Check threshold value
        if not 1.0 <= self.threshold <= 10.0:
            validation_results['errors'].append(
                f"Threshold ({self.threshold}) should be between 1.0 and 10.0 for rating prediction"
            )
            validation_results['is_valid'] = False
        
        # Check Random Forest parameters
        if self.rf_params.get('n_estimators', 0) < 10:
            validation_results['warnings'].append(
                "Random Forest n_estimators is quite low - consider increasing for better performance"
            )
        
        # Check if random_state is set for reproducibility
        if 'random_state' not in self.rf_params:
            validation_results['recommendations'].append(
                "Consider setting random_state in RF parameters for reproducible results"
            )
        
        return validation_results
    
    def get_training_statistics(self, evaluation_results: Dict = None) -> Dict[str, any]:
        """
        Get comprehensive training statistics if evaluation results are provided.
        
        Analyzes evaluation results to provide statistical summaries and
        performance comparisons between model types.
        
        Parameters
        ----------
        evaluation_results : Dict, optional
            Results dictionary from model training containing metrics for all models.
            Default is None.
            
        Returns
        -------
        statistics : Dict[str, any]
            Dictionary containing training statistics with keys:
            - models_trained (List[str]): List of trained model identifiers
            - metric_summary (Dict): Statistical summary for each metric type
            - performance_comparison (Dict): Comparison between traditional and BERT models
            Returns error dict if evaluation_results is None or processing fails.
            
        Notes
        -----
        Calculates mean, std, min, max for each metric across models.
        Computes BERT improvement percentage over traditional models for RMSE.
        """
        if not evaluation_results:
            return {'error': 'No evaluation results provided'}
        
        try:
            stats = {
                'models_trained': [],
                'metric_summary': {},
                'performance_comparison': {}
            }
            
            # Extract model names and metrics
            for key in evaluation_results.keys():
                if not key.endswith('_TrainingTime') and key != 'feature_importances':
                    model_name = key.split('_')[0] + '_' + key.split('_')[1]
                    if model_name not in stats['models_trained']:
                        stats['models_trained'].append(model_name)
            
            # Calculate metric summaries
            for metric in self.REGRESSION_METRICS + self.CLASSIFICATION_METRICS + ['MAE_High_Rated']:
                metric_values = []
                for key, value in evaluation_results.items():
                    if metric in key and not key.endswith('_TrainingTime'):
                        metric_values.append(value)
                
                if metric_values:
                    stats['metric_summary'][metric] = {
                        'mean': np.mean(metric_values),
                        'std': np.std(metric_values),
                        'min': np.min(metric_values),
                        'max': np.max(metric_values)
                    }
            
            # Performance comparison between traditional and BERT models
            traditional_rmse = []
            bert_rmse = []
            
            for key, value in evaluation_results.items():
                if 'RMSE' in key:
                    if 'Traditional' in key:
                        traditional_rmse.append(value)
                    elif 'BERT' in key:
                        bert_rmse.append(value)
            
            if traditional_rmse and bert_rmse:
                stats['performance_comparison'] = {
                    'traditional_avg_rmse': np.mean(traditional_rmse),
                    'bert_avg_rmse': np.mean(bert_rmse),
                    'bert_improvement': (np.mean(traditional_rmse) - np.mean(bert_rmse)) / np.mean(traditional_rmse) * 100
                }
            
            return stats
            
        except Exception as e:
            return {'error': f'Error generating training statistics: {e}'}
