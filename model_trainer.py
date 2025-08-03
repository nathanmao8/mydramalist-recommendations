# model_trainer.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut, KFold

from typing import Dict, List, Tuple
import time
import pandas as pd

class ModelTrainer:
    """
    ModelTrainer handles model training with cross-validation for ranking evaluation.
    
    Features:
    - Traditional and BERT-enhanced model training
    - K-Fold and Leave-One-Out Cross-Validation
    - MAP and Precision@K ranking evaluation metrics
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
    PROGRESS_UPDATE_INTERVAL = 10  # Update progress every N folds
    
    # Cross-validation configuration
    DEFAULT_VALIDATION_METHOD = 'kfold'
    DEFAULT_N_FOLDS = 10
    RANDOM_STATE = 42
    
    # Ranking evaluation thresholds
    RANKING_THRESHOLDS = ['top_10%', 'top_25%', 'top_33%']
    
    # MAP thresholds
    MAP_THRESHOLDS = [0.1, 0.25, 0.33]  # 10%, 25%, 33%
    
    # Metric names
    RANKING_METRICS = ['MAP_10%', 'MAP_25%', 'MAP_33%', 'Precision_at_10%', 'Precision_at_25%', 'Precision_at_33%']
    
    def __init__(self, rf_params: Dict = None, svm_params: Dict = None, 
                 validation_method: str = None, n_folds: int = None):
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

        validation_method : str, optional
            Cross-validation method: 'loocv' or 'kfold'.
            Default is DEFAULT_VALIDATION_METHOD ('kfold').
        n_folds : int, optional
            Number of folds for k-fold cross-validation.
            Default is DEFAULT_N_FOLDS (10).
        """
        
        # Configuration
        self.rf_params = {**self.DEFAULT_RF_PARAMS, **(rf_params or {})}
        self.svm_params = {**self.DEFAULT_SVM_PARAMS, **(svm_params or {})}
        self.validation_method = validation_method or self.DEFAULT_VALIDATION_METHOD
        self.n_folds = n_folds or self.DEFAULT_N_FOLDS
        
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
        Train both traditional and hybrid models using specified validation method.
        
        Dispatches to either LOOCV or K-Fold cross-validation based on 
        the validation_method setting. Both methods focus on ranking evaluation
        using Mean Average Precision (MAP) and Precision@K metrics.
        
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
            MAP, Precision@K (10%, 25%, 33%), and feature importances.
        cv_predictions : Dict
            Cross-validation predictions for all models. Format depends
            on validation method used.
            
        Notes
        -----
        Uses ranking-focused evaluation metrics optimized for recommendation systems:
        - Mean Average Precision (MAP): Overall ranking quality
        - Precision@K: Accuracy of top recommendations at 10%, 25%, and 33% thresholds
        
        Example
        -------
        >>> trainer = ModelTrainer(validation_method='kfold', n_folds=5)
        >>> models, results, predictions = trainer.train_models(X_trad, X_bert, y)
        >>> print(f"RF Traditional MAP: {results['RF_TRADITIONAL_MAP']:.3f}")
        """
        if self.validation_method == 'loocv':
            return self.train_models_loocv(X_traditional, X_hybrid, y)
        elif self.validation_method == 'kfold':
            return self.train_models_kfold(X_traditional, X_hybrid, y)
        else:
            raise ValueError(f"Unknown validation method: {self.validation_method}")
    
    def train_models_loocv(self, X_traditional: np.ndarray, X_hybrid: np.ndarray, y: np.ndarray) -> Tuple[Dict, Dict, Dict]:
        
        try:
            print("Training traditional models with LOOCV...")
            print("Focusing on MAP and Precision@K metrics for recommendation ranking evaluation.")
            traditional_results, rf_trad_preds, svm_trad_preds, true_values = self.train_model_set(
                X_traditional, y, 
                self.rf_traditional, self.svm_traditional,
                model_prefix="Traditional"
            )
            
            print("Training BERT-enhanced models with LOOCV...")
            bert_results, rf_bert_preds, svm_bert_preds, _ = self.train_model_set(
                X_hybrid, y,
                self.rf_bert, self.svm_bert,
                model_prefix="BERT"
            )
            
            # Combine traditional results
            all_results = {**traditional_results, **bert_results}
            
            print("Calculating MAP and Precision@K evaluation metrics...")
            
            # Add ranking metrics for each model
            model_predictions = {
                'rf_traditional': rf_trad_preds,
                'svm_traditional': svm_trad_preds,
                'rf_bert': rf_bert_preds,
                'svm_bert': svm_bert_preds
            }
            
            for model_name, predictions in model_predictions.items():
                actual = np.array(true_values)
                predicted = np.array(predictions)
                
                # Calculate MAP and Precision@K metrics
                ranking_metrics = self._calculate_map_and_precision_at_k(actual, predicted)
                
                # Store results with consistent naming
                # Fix the key naming to match evaluator expectations
                if model_name == 'rf_traditional':
                    model_key = 'RF_Traditional'
                elif model_name == 'svm_traditional':
                    model_key = 'SVM_Traditional'
                elif model_name == 'rf_bert':
                    model_key = 'RF_BERT'
                elif model_name == 'svm_bert':
                    model_key = 'SVM_BERT'
                else:
                    model_key = model_name.upper().replace('_', '_')
                
                # Store MAP scores for different thresholds
                for map_key, map_value in ranking_metrics['map_scores'].items():
                    all_results[f'{model_key}_{map_key}'] = map_value
                
                # Store Precision@K scores
                all_results[f'{model_key}_Precision_at_10%'] = ranking_metrics['precision_at_k']['top_10%']
                all_results[f'{model_key}_Precision_at_25%'] = ranking_metrics['precision_at_k']['top_25%']
                all_results[f'{model_key}_Precision_at_33%'] = ranking_metrics['precision_at_k']['top_33%']
            
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
                'validation_method': 'loocv',
                'true_ratings': true_values,
                'rf_traditional_preds': rf_trad_preds,
                'svm_traditional_preds': svm_trad_preds,
                'rf_bert_preds': rf_bert_preds,
                'svm_bert_preds': svm_bert_preds,
                'ranking_focus': True
            }
            
            print("LOOCV evaluation completed. Mean Average Precision (MAP) scores:")
            for model_name in ['RF_TRADITIONAL', 'SVM_TRADITIONAL', 'RF_BERT', 'SVM_BERT']:
                map_10 = all_results.get(f'{model_name}_MAP_10%', 0)
                map_25 = all_results.get(f'{model_name}_MAP_25%', 0)
                map_33 = all_results.get(f'{model_name}_MAP_33%', 0)
                print(f"  {model_name}: MAP@10%={map_10:.3f}, MAP@25%={map_25:.3f}, MAP@33%={map_33:.3f}")

            return trained_models, all_results, loocv_predictions
            
        except Exception as e:
            print(f"Error during model training: {e}")
            # Return empty results in case of failure
            return {}, {}, {}
    
    def train_models_kfold(self, X_traditional: np.ndarray, X_hybrid: np.ndarray, y: np.ndarray) -> Tuple[Dict, Dict, Dict]:
        """
        Train both traditional and hybrid models using K-Fold cross-validation with ranking evaluation.
        
        Focuses on Mean Average Precision (MAP) and Precision@K metrics rather than 
        absolute rating accuracy, optimized for recommendation system evaluation.
        
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
            Dictionary containing all trained model instances trained on full dataset.
        all_results : Dict
            Comprehensive evaluation metrics focused on ranking performance.
        kfold_predictions : Dict
            K-fold predictions and ranking metrics for all models.
            
        Notes
        -----
        Uses ranking-focused evaluation metrics optimized for recommendation systems:
        - Mean Average Precision (MAP): Overall ranking quality
        - Precision@K: Accuracy of top recommendations at 10%, 25%, and 33% thresholds
        """
        print(f"Training models with {self.n_folds}-Fold Cross-Validation...")
        
        try:
            # Initialize k-fold splitter
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.RANDOM_STATE)
            
            # Initialize result collectors
            model_results = {
                'rf_traditional': {'actual': [], 'predicted': []},
                'svm_traditional': {'actual': [], 'predicted': []},
                'rf_bert': {'actual': [], 'predicted': []},
                'svm_bert': {'actual': [], 'predicted': []}
            }
            
            fold_count = 0
            for train_idx, test_idx in kf.split(X_traditional):
                fold_count += 1
                if fold_count % self.PROGRESS_UPDATE_INTERVAL == 0:
                    print(f"Processing fold {fold_count}/{self.n_folds}...")
                
                # Split data for traditional models
                X_train_trad, X_test_trad = X_traditional[train_idx], X_traditional[test_idx]
                X_train_bert, X_test_bert = X_hybrid[train_idx], X_hybrid[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train and predict with traditional models
                rf_trad = RandomForestRegressor(**self.rf_params)
                svm_trad = SVR(**self.svm_params)
                
                rf_trad.fit(X_train_trad, y_train)
                svm_trad.fit(X_train_trad, y_train)
                
                rf_trad_pred = rf_trad.predict(X_test_trad)
                svm_trad_pred = svm_trad.predict(X_test_trad)
                
                # Train and predict with BERT models
                rf_bert = RandomForestRegressor(**self.rf_params)
                svm_bert = SVR(**self.svm_params)
                
                rf_bert.fit(X_train_bert, y_train)
                svm_bert.fit(X_train_bert, y_train)
                
                rf_bert_pred = rf_bert.predict(X_test_bert)
                svm_bert_pred = svm_bert.predict(X_test_bert)
                
                # Store results
                model_results['rf_traditional']['actual'].extend(y_test)
                model_results['rf_traditional']['predicted'].extend(rf_trad_pred)
                model_results['svm_traditional']['actual'].extend(y_test)
                model_results['svm_traditional']['predicted'].extend(svm_trad_pred)
                model_results['rf_bert']['actual'].extend(y_test)
                model_results['rf_bert']['predicted'].extend(rf_bert_pred)
                model_results['svm_bert']['actual'].extend(y_test)
                model_results['svm_bert']['predicted'].extend(svm_bert_pred)
            
            print("Calculating MAP and Precision@K evaluation metrics...")
            
            # Calculate ranking metrics for each model
            all_results = {}
            for model_name, results in model_results.items():
                actual = np.array(results['actual'])
                predicted = np.array(results['predicted'])
                
                # Calculate MAP and Precision@K metrics
                ranking_metrics = self._calculate_map_and_precision_at_k(actual, predicted)
                
                # Store results with consistent naming
                # Fix the key naming to match evaluator expectations
                if model_name == 'rf_traditional':
                    model_key = 'RF_Traditional'
                elif model_name == 'svm_traditional':
                    model_key = 'SVM_Traditional'
                elif model_name == 'rf_bert':
                    model_key = 'RF_BERT'
                elif model_name == 'svm_bert':
                    model_key = 'SVM_BERT'
                else:
                    model_key = model_name.upper().replace('_', '_')
                
                # Store MAP scores for different thresholds
                for map_key, map_value in ranking_metrics['map_scores'].items():
                    all_results[f'{model_key}_{map_key}'] = map_value
                
                # Store Precision@K scores
                all_results[f'{model_key}_Precision_at_10%'] = ranking_metrics['precision_at_k']['top_10%']
                all_results[f'{model_key}_Precision_at_25%'] = ranking_metrics['precision_at_k']['top_25%']
                all_results[f'{model_key}_Precision_at_33%'] = ranking_metrics['precision_at_k']['top_33%']
                

            
            # Train final models on full dataset
            print("Training final models on full dataset...")
            self._initialize_models()
            
            self.rf_traditional.fit(X_traditional, y)
            self.svm_traditional.fit(X_traditional, y)
            self.rf_bert.fit(X_hybrid, y)
            self.svm_bert.fit(X_hybrid, y)
            
            # Capture feature importances
            feature_importances = {
                'rf_traditional': self.rf_traditional.feature_importances_,
                'rf_bert': self.rf_bert.feature_importances_
            }
            all_results['feature_importances'] = feature_importances
            
            # Create model dictionary
            trained_models = {
                'rf_traditional': self.rf_traditional,
                'svm_traditional': self.svm_traditional,
                'rf_bert': self.rf_bert,
                'svm_bert': self.svm_bert
            }
            
            # Create k-fold predictions dictionary
            kfold_predictions = {
                'validation_method': 'kfold',
                'n_folds': self.n_folds,
                'model_results': model_results,
                'ranking_focus': True
            }
            
            return trained_models, all_results, kfold_predictions
            
        except Exception as e:
            print(f"Error during K-Fold training: {e}")
            return {}, {}, {}
    
    def _calculate_map_and_precision_at_k(self, actual_ratings: np.ndarray, predicted_ratings: np.ndarray) -> Dict:
        """
        Calculate Mean Average Precision (MAP) and Precision@K for recommendation ranking evaluation.
        
        Parameters
        ----------
        actual_ratings : np.ndarray
            Actual rating values.
        predicted_ratings : np.ndarray
            Predicted rating values.
            
        Returns
        -------
        Dict
            Dictionary with MAP scores at different thresholds and Precision@K metrics.
        """
        # Create dataframe with indices to track original positions
        df = pd.DataFrame({
            'actual_rating': actual_ratings,
            'predicted_rating': predicted_ratings,
            'index': range(len(actual_ratings))
        })
        
        # Sort by predicted ratings (descending - highest predicted first)
        df_predicted_rank = df.sort_values('predicted_rating', ascending=False).reset_index(drop=True)
        
        # Sort by actual ratings to identify truly good items (descending - highest actual first)
        df_actual_rank = df.sort_values('actual_rating', ascending=False).reset_index(drop=True)
        
        # Calculate MAP at different thresholds
        map_scores = {}
        for threshold in self.MAP_THRESHOLDS:
            map_score = self._calculate_mean_average_precision(df_predicted_rank, df_actual_rank, threshold)
            map_scores[f'MAP_{int(threshold*100)}%'] = map_score
        
        # Calculate Precision@K for different thresholds
        precision_at_k = {}
        
        for threshold in self.RANKING_THRESHOLDS:
            if threshold == 'top_10%':
                k = max(1, len(df) // 10)  # 10%
            elif threshold == 'top_25%':
                k = max(1, len(df) // 4)   # 25%
            else:  # top_33%
                k = max(1, len(df) // 3)   # 33%
            
            # Get top-k predicted items
            top_k_predicted_indices = set(df_predicted_rank.head(k)['index'])
            
            # Get top-k actual items (ground truth)
            top_k_actual_indices = set(df_actual_rank.head(k)['index'])
            
            # Calculate precision@k
            intersection = top_k_predicted_indices.intersection(top_k_actual_indices)
            precision_k = len(intersection) / k if k > 0 else 0
            
            precision_at_k[threshold] = precision_k
        
        return {
            'map_scores': map_scores,
            'precision_at_k': precision_at_k
        }
    
    def _calculate_mean_average_precision(self, df_predicted_rank: pd.DataFrame, df_actual_rank: pd.DataFrame, threshold: float = 0.5) -> float:
        """
        Calculate Mean Average Precision (MAP) for ranking evaluation.
        
        Parameters
        ----------
        df_predicted_rank : pd.DataFrame
            DataFrame sorted by predicted ratings (descending).
        df_actual_rank : pd.DataFrame
            DataFrame sorted by actual ratings (descending).
        threshold : float, optional
            Fraction of top items to consider as relevant (default: 0.5 for 50%).
            
        Returns
        -------
        float
            Mean Average Precision score.
        """
        # Get the indices of top items according to actual ratings
        # Consider top threshold% as relevant items for MAP calculation
        n_relevant = max(1, int(len(df_actual_rank) * threshold))
        relevant_indices = set(df_actual_rank.head(n_relevant)['index'])
        
        if len(relevant_indices) == 0:
            return 0.0
        
        # Calculate average precision
        relevant_found = 0
        precision_sum = 0.0
        
        for rank, row in df_predicted_rank.iterrows():
            item_index = row['index']
            
            if item_index in relevant_indices:
                relevant_found += 1
                precision_at_rank = relevant_found / (rank + 1)  # rank is 0-indexed
                precision_sum += precision_at_rank
        
        # Average precision for this ranking
        if relevant_found > 0:
            return precision_sum / len(relevant_indices)
        else:
            return 0.0
    

    
    def _perform_loocv(self, X: np.ndarray, y: np.ndarray, model_prefix: str) -> Tuple[List, List, List]:
        """
        Perform Leave-One-Out Cross-Validation.
        
        Iterates through all samples, using each as a test case while training
        on the remaining samples for ranking evaluation.
        
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
                
                # Train models on training fold and predict on test fold
                rf_temp = RandomForestRegressor(**self.rf_params)
                svm_temp = SVR(**self.svm_params)
                
                rf_temp.fit(X_train_fold, y_train_fold)
                svm_temp.fit(X_train_fold, y_train_fold)
                
                rf_pred = rf_temp.predict(X_test_fold)[0]
                svm_pred = svm_temp.predict(X_test_fold)[0]
                
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
    
    def _calculate_single_model_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        """
        Calculate all metrics for a single model.
        
        Computes ranking evaluation metrics for a single model.
        
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
            - {model_name}_MAP, _Precision_at_10%, _Precision_at_25%, _Precision_at_33%
            
        Notes
        -----
        Focuses on ranking metrics optimized for recommendation systems.
        """
        metrics = {}
        
        # Ranking metrics
        ranking_metrics = self._calculate_map_and_precision_at_k(y_true, y_pred)
        for metric_name, value in ranking_metrics.items():
            if metric_name == 'map':
                metrics[f'{model_name}_MAP'] = value
            elif metric_name == 'precision_at_k':
                for threshold, precision in value.items():
                    if threshold == 'top_10%':
                        metrics[f'{model_name}_Precision_at_10%'] = precision
                    elif threshold == 'top_25%':
                        metrics[f'{model_name}_Precision_at_25%'] = precision
                    elif threshold == 'top_33%':
                        metrics[f'{model_name}_Precision_at_33%'] = precision
        
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
            - All regression and ranking metrics for both models
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
            - validation_method: Cross-validation method
            - n_folds: Number of folds for k-fold validation
            - progress_update_interval: Progress reporting frequency
            - ranking_metrics: List of ranking metric names
            
        Notes
        -----
        Returns copies of mutable parameters to prevent accidental modification.
        """
        return {
            'rf_params': self.rf_params.copy(),
            'svm_params': self.svm_params.copy(),
            'validation_method': self.validation_method,
            'n_folds': self.n_folds,
            'progress_update_interval': self.PROGRESS_UPDATE_INTERVAL,
            'ranking_metrics': self.RANKING_METRICS
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
            - validation_method (str): Cross-validation method
            - n_folds (int): Number of folds for k-fold validation
            
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
            
        if 'validation_method' in kwargs:
            self.validation_method = kwargs['validation_method']
            
        if 'n_folds' in kwargs:
            self.n_folds = kwargs['n_folds']
            

        
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
        Checks validation method, n_folds, threshold values, RF parameters,
        and reproducibility settings. Critical errors set is_valid to False.
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check validation method
        if self.validation_method not in ['loocv', 'kfold']:
            validation_results['errors'].append(
                f"Invalid validation method: {self.validation_method}. Must be 'loocv' or 'kfold'"
            )
            validation_results['is_valid'] = False
        
        # Check n_folds for k-fold validation
        if self.validation_method == 'kfold' and (self.n_folds < 2 or self.n_folds > 50):
            validation_results['warnings'].append(
                f"Number of folds ({self.n_folds}) is outside recommended range (2-50)"
            )
        

        
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
            for metric in self.RANKING_METRICS:
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
            traditional_map = []
            bert_map = []
            
            for key, value in evaluation_results.items():
                if 'MAP' in key:
                    if 'Traditional' in key:
                        traditional_map.append(value)
                    elif 'BERT' in key:
                        bert_map.append(value)
            
            if traditional_map and bert_map:
                stats['performance_comparison'] = {
                    'traditional_avg_map': np.mean(traditional_map),
                    'bert_avg_map': np.mean(bert_map),
                    'bert_improvement': (np.mean(bert_map) - np.mean(traditional_map)) / np.mean(traditional_map) * 100
                }
            
            return stats
            
        except Exception as e:
            return {'error': f'Error generating training statistics: {e}'}
