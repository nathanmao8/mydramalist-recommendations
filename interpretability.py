import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ExplainabilityConfig:
    """Configuration class for SHAP explanations."""
    output_dir: str = "shap_outputs"
    figure_dpi: int = 300
    figure_size: tuple = (12, 8)
    save_plots: bool = True
    show_plots: bool = False
    max_display_features: int = 20


class ModelExplainer(ABC):
    """Abstract base class for model explainers."""
    
    @abstractmethod
    def create_explainer(self, model: Any, X: Optional[pd.DataFrame] = None) -> Any:
        """
        Create an explainer for the given model.
        
        Parameters
        ----------
        model : Any
            The trained model object for which to create an explainer.
        X : Optional[pd.DataFrame], default=None
            Training data used to create the explainer. May be required for some explainer types.
            
        Returns
        -------
        Any
            The created SHAP explainer object.
            
        Raises
        ------
        ValueError
            If the explainer cannot be created for the given model.
        """
        pass
    
    @abstractmethod
    def calculate_shap_values(self, explainer: Any, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for the given data.
        
        Parameters
        ----------
        explainer : Any
            The SHAP explainer object.
        X : pd.DataFrame
            Feature data for which to calculate SHAP values.
            
        Returns
        -------
        np.ndarray
            Array of SHAP values with shape (n_samples, n_features).
            
        Raises
        ------
        RuntimeError
            If SHAP values cannot be calculated.
        """
        pass


class TreeExplainer(ModelExplainer):
    """Explainer for tree-based models using SHAP TreeExplainer."""
    
    def create_explainer(self, model: Any, X: Optional[pd.DataFrame] = None) -> shap.TreeExplainer:
        """
        Create a SHAP TreeExplainer for tree-based models.
        
        Parameters
        ----------
        model : Any
            The trained tree-based model (e.g., RandomForest, XGBoost, LightGBM).
        X : Optional[pd.DataFrame], default=None
            Training data. Not required for TreeExplainer but included for interface consistency.
            
        Returns
        -------
        shap.TreeExplainer
            Configured TreeExplainer for the model.
            
        Raises
        ------
        ValueError
            If the TreeExplainer cannot be created for the given model.
        """
        try:
            return shap.TreeExplainer(model)
        except Exception as e:
            raise ValueError(f"Failed to create TreeExplainer: {e}")
    
    def calculate_shap_values(self, explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values using TreeExplainer.
        
        Parameters
        ----------
        explainer : shap.TreeExplainer
            The TreeExplainer object.
        X : pd.DataFrame
            Feature data for which to calculate SHAP values.
            
        Returns
        -------
        np.ndarray
            Array of SHAP values with shape (n_samples, n_features).
            
        Raises
        ------
        RuntimeError
            If SHAP values cannot be calculated.
        """
        try:
            return explainer.shap_values(X)
        except Exception as e:
            raise RuntimeError(f"Failed to calculate SHAP values: {e}")


class LinearExplainer(ModelExplainer):
    """Explainer for linear models using SHAP LinearExplainer."""
    
    def create_explainer(self, model: Any, X: Optional[pd.DataFrame] = None) -> shap.LinearExplainer:
        """
        Create a SHAP LinearExplainer for linear models.
        
        Parameters
        ----------
        model : Any
            The trained linear model (e.g., LinearRegression, LogisticRegression).
        X : Optional[pd.DataFrame], default=None
            Training data used to create background data for the explainer.
            
        Returns
        -------
        shap.LinearExplainer
            Configured LinearExplainer for the model.
            
        Raises
        ------
        ValueError
            If the LinearExplainer cannot be created for the given model.
        """
        try:
            if X is not None:
                return shap.LinearExplainer(model, shap.sample(X, min(100, len(X))))
            else:
                return shap.LinearExplainer(model)
        except Exception as e:
            raise ValueError(f"Failed to create LinearExplainer: {e}")
    
    def calculate_shap_values(self, explainer: shap.LinearExplainer, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values using LinearExplainer.
        
        Parameters
        ----------
        explainer : shap.LinearExplainer
            The LinearExplainer object.
        X : pd.DataFrame
            Feature data for which to calculate SHAP values.
            
        Returns
        -------
        np.ndarray
            Array of SHAP values with shape (n_samples, n_features).
            
        Raises
        ------
        RuntimeError
            If SHAP values cannot be calculated.
        """
        try:
            return explainer.shap_values(X)
        except Exception as e:
            raise RuntimeError(f"Failed to calculate SHAP values: {e}")


class PlotManager:
    """Handles SHAP visualization and plotting."""
    
    def __init__(self, config: ExplainabilityConfig):
        """
        Initialize the PlotManager with configuration settings.
        
        Parameters
        ----------
        config : ExplainabilityConfig
            Configuration object containing plotting settings and output directory.
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure matplotlib
        plt.rcParams['figure.dpi'] = config.figure_dpi
        plt.rcParams['figure.figsize'] = config.figure_size
    
    def create_summary_plot(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        feature_names: List[str],
        model_name: str
    ) -> Optional[str]:
        """
        Create and optionally save a SHAP summary plot.
        
        Parameters
        ----------
        shap_values : np.ndarray
            Array of SHAP values with shape (n_samples, n_features).
        X : pd.DataFrame
            Feature data corresponding to the SHAP values.
        feature_names : List[str]
            Names of the features.
        model_name : str
            Name of the model for plot title and filename.
            
        Returns
        -------
        Optional[str]
            Path to saved plot if saved, None otherwise.
        """
        try:
            plt.figure(figsize=self.config.figure_size)
            
            shap.summary_plot(
                shap_values, 
                X, 
                feature_names=feature_names,
                max_display=self.config.max_display_features,
                show=False
            )
            
            plt.title(f'SHAP Global Feature Importance - {model_name}', fontsize=14, pad=20)
            plt.tight_layout()
            
            if self.config.save_plots:
                filename = self.output_dir / f'shap_summary_{model_name}.png'
                plt.savefig(filename, dpi=self.config.figure_dpi, bbox_inches='tight')
                logging.info(f"Saved SHAP summary plot to {filename}")
                
                if not self.config.show_plots:
                    plt.close()
                return str(filename)
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logging.error(f"Failed to create summary plot for {model_name}: {e}")
            plt.close()
        
        return None
    
    def create_force_plot(
        self,
        explainer: Any,
        shap_values: np.ndarray,
        instance_index: int,
        feature_names: List[str],
        model_name: str
    ) -> Optional[str]:
        """
        Create and optionally save a SHAP force plot for a single instance.
        
        Parameters
        ----------
        explainer : Any
            The SHAP explainer object.
        shap_values : np.ndarray
            Array of SHAP values with shape (n_samples, n_features).
        instance_index : int
            Index of the instance to explain.
        feature_names : List[str]
            Names of the features.
        model_name : str
            Name of the model for plot title and filename.
            
        Returns
        -------
        Optional[str]
            Path to saved plot if saved, None otherwise.
        """
        try:
            plt.figure(figsize=self.config.figure_size)
            
            shap.force_plot(
                explainer.expected_value,
                shap_values[instance_index, :],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            
            plt.title(f'SHAP Force Plot - {model_name} (Instance {instance_index})', 
                     fontsize=14, pad=20)
            
            if self.config.save_plots:
                filename = self.output_dir / f'shap_force_{model_name}_instance_{instance_index}.png'
                plt.savefig(filename, dpi=self.config.figure_dpi, bbox_inches='tight')
                logging.info(f"Saved SHAP force plot to {filename}")
                
                if not self.config.show_plots:
                    plt.close()
                return str(filename)
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logging.error(f"Failed to create force plot for {model_name}, instance {instance_index}: {e}")
            plt.close()
        
        return None
    
    def create_waterfall_plot(
        self,
        explainer: Any,
        shap_values: np.ndarray,
        instance_index: int,
        feature_names: List[str],
        model_name: str,
        X: pd.DataFrame
    ) -> Optional[str]:
        """
        Create and optionally save a SHAP waterfall plot for a single instance.
        
        Parameters
        ----------
        explainer : Any
            The SHAP explainer object.
        shap_values : np.ndarray
            Array of SHAP values with shape (n_samples, n_features).
        instance_index : int
            Index of the instance to explain.
        feature_names : List[str]
            Names of the features.
        model_name : str
            Name of the model for plot title and filename.
        X : pd.DataFrame
            Feature data corresponding to the SHAP values.
            
        Returns
        -------
        Optional[str]
            Path to saved plot if saved, None otherwise.
        """
        try:
            plt.figure(figsize=self.config.figure_size)
            
            # Create explanation object for waterfall plot
            explanation = shap.Explanation(
                values=shap_values[instance_index],
                base_values=explainer.expected_value,
                data=X.iloc[instance_index].values,
                feature_names=feature_names
            )
            
            shap.waterfall_plot(explanation, show=False)
            plt.title(f'SHAP Waterfall Plot - {model_name} (Instance {instance_index})', 
                     fontsize=14, pad=20)
            
            if self.config.save_plots:
                filename = self.output_dir / f'shap_waterfall_{model_name}_instance_{instance_index}.png'
                plt.savefig(filename, dpi=self.config.figure_dpi, bbox_inches='tight')
                logging.info(f"Saved SHAP waterfall plot to {filename}")
                
                if not self.config.show_plots:
                    plt.close()
                return str(filename)
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logging.error(f"Failed to create waterfall plot for {model_name}, instance {instance_index}: {e}")
            plt.close()
        
        return None


class ShapExplainer:
    """
    Enhanced SHAP explainer with support for multiple model types and improved error handling.
    """
    
    def __init__(
        self,
        models: Dict[str, Any],
        X_data: Dict[str, pd.DataFrame],
        feature_names: Dict[str, List[str]],
        config: Optional[ExplainabilityConfig] = None
    ):
        """
        Initialize the SHAP explainer.
        
        Parameters
        ----------
        models : Dict[str, Any]
            Dictionary of trained model objects where keys are model names.
        X_data : Dict[str, pd.DataFrame]
            Dictionary containing feature DataFrames for each model type.
        feature_names : Dict[str, List[str]]
            Dictionary containing feature name lists for each model type.
        config : Optional[ExplainabilityConfig], default=None
            Configuration for explainability settings. If None, uses default configuration.
        """
        self.models = self._validate_models(models)
        self.X_data = self._validate_data(X_data)
        self.feature_names = self._validate_feature_names(feature_names)
        self.config = config or ExplainabilityConfig()
        
        self.explainers: Dict[str, Any] = {}
        self.shap_values: Dict[str, np.ndarray] = {}
        self.plot_manager = PlotManager(self.config)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        logging.info("Initializing SHAP explainers...")
        self._initialize_explainers()
    
    def _validate_models(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the models dictionary.
        
        Parameters
        ----------
        models : Dict[str, Any]
            Dictionary of model objects to validate.
            
        Returns
        -------
        Dict[str, Any]
            Validated models dictionary.
            
        Raises
        ------
        ValueError
            If models dictionary is empty, contains None values, or models lack predict method.
        """
        if not models:
            raise ValueError("Models dictionary cannot be empty")
        
        for name, model in models.items():
            if model is None:
                raise ValueError(f"Model '{name}' is None")
            if not hasattr(model, 'predict'):
                raise ValueError(f"Model '{name}' does not have a predict method")
        
        return models
    
    def _validate_data(self, X_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Validate the feature data dictionary.
        
        Parameters
        ----------
        X_data : Dict[str, pd.DataFrame]
            Dictionary of feature DataFrames to validate.
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Validated data dictionary.
            
        Raises
        ------
        ValueError
            If data dictionary is empty or contains empty DataFrames.
        TypeError
            If data values are not pandas DataFrames.
        """
        if not X_data:
            raise ValueError("X_data dictionary cannot be empty")
        
        for name, data in X_data.items():
            if not isinstance(data, pd.DataFrame):
                raise TypeError(f"Data for '{name}' must be a pandas DataFrame")
            if data.empty:
                raise ValueError(f"Data for '{name}' cannot be empty")
        
        return X_data
    
    def _validate_feature_names(self, feature_names: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Validate the feature names dictionary.
        
        Parameters
        ----------
        feature_names : Dict[str, List[str]]
            Dictionary of feature name lists to validate.
            
        Returns
        -------
        Dict[str, List[str]]
            Validated feature names dictionary.
            
        Raises
        ------
        ValueError
            If feature names dictionary is empty or contains empty lists.
        TypeError
            If feature names values are not lists.
        """
        if not feature_names:
            raise ValueError("Feature names dictionary cannot be empty")
        
        for name, names in feature_names.items():
            if not isinstance(names, list):
                raise TypeError(f"Feature names for '{name}' must be a list")
            if not names:
                raise ValueError(f"Feature names for '{name}' cannot be empty")
        
        return feature_names
    
    def _get_explainer_type(self, model_name: str, model: Any) -> ModelExplainer:
        """
        Determine the appropriate explainer type for the model.
        
        Parameters
        ----------
        model_name : str
            Name of the model.
        model : Any
            The model object.
            
        Returns
        -------
        ModelExplainer
            Appropriate explainer instance for the model type.
        """
        model_name_lower = model_name.lower()
        
        # Check for tree-based models
        tree_indicators = ['rf', 'random_forest', 'randomforest', 'xgb', 'xgboost', 
                          'lgb', 'lightgbm', 'catboost', 'tree']
        
        if any(indicator in model_name_lower for indicator in tree_indicators):
            return TreeExplainer()
        
        # Check for linear models
        linear_indicators = ['linear', 'logistic', 'ridge', 'lasso', 'elastic']
        
        if any(indicator in model_name_lower for indicator in linear_indicators):
            return LinearExplainer()
        
        # Default to TreeExplainer for unknown models (with warning)
        logging.warning(f"Unknown model type for {model_name}, defaulting to TreeExplainer")
        return TreeExplainer()
    
    def _initialize_explainers(self) -> None:
        """
        Initialize SHAP explainers for all models.
        
        Creates appropriate SHAP explainers for each model based on model type detection.
        Logs success and failure information for each model.
        """
        for model_name, model in self.models.items():
            try:
                explainer_type = self._get_explainer_type(model_name, model)
                # Get appropriate data for the model
                data_key = self._get_data_key(model_name)
                X_set = self.X_data[data_key]
                
                self.explainers[model_name] = explainer_type.create_explainer(model, X_set)
                logging.info(f"Created explainer for: {model_name}")
                
            except Exception as e:
                logging.error(f"Failed to create explainer for {model_name}: {e}")
    
    def calculate_shap_values(self, model_names: Optional[List[str]] = None) -> None:
        """
        Calculate SHAP values for specified models.
        
        Parameters
        ----------
        model_names : Optional[List[str]], default=None
            List of model names to calculate SHAP values for.
            If None, calculates for all available models.
        """
        target_models = model_names or list(self.explainers.keys())
        
        for model_name in target_models:
            if model_name not in self.explainers:
                logging.warning(f"No explainer available for {model_name}, skipping...")
                continue
            
            try:
                # Get the appropriate data and feature names
                data_key = self._get_data_key(model_name)
                X_set = self.X_data[data_key]
                
                # Calculate SHAP values
                explainer_type = self._get_explainer_type(model_name, self.models[model_name])
                shap_values = explainer_type.calculate_shap_values(
                    self.explainers[model_name], X_set
                )
                
                self.shap_values[model_name] = shap_values
                logging.info(f"Calculated SHAP values for: {model_name}")
                
            except Exception as e:
                logging.error(f"Failed to calculate SHAP values for {model_name}: {e}")
    
    def _get_data_key(self, model_name: str) -> str:
        """
        Determine the appropriate data key for a model.
        
        Parameters
        ----------
        model_name : str
            Name of the model.
            
        Returns
        -------
        str
            Data key corresponding to the model type.
        """
        if 'traditional' in model_name.lower():
            return 'traditional'
        elif 'hybrid' in model_name.lower():
            return 'hybrid'
        else:
            # Default to the first available key
            return list(self.X_data.keys())[0]
    
    def explain_models(
        self,
        model_names: Optional[List[str]] = None,
        instance_indices: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate comprehensive SHAP explanations for specified models.
        
        Parameters
        ----------
        model_names : Optional[List[str]], default=None
            List of model names to explain. If None, explains all models.
        instance_indices : Optional[List[int]], default=None
            List of instance indices for local explanations.
            If None, uses [0] as default.
            
        Returns
        -------
        Dict[str, Dict[str, str]]
            Nested dictionary mapping model names to plot types and their file paths.
            Structure: {model_name: {plot_type: file_path}}
        """
        if not self.shap_values:
            self.calculate_shap_values(model_names)
        
        target_models = model_names or list(self.shap_values.keys())
        target_indices = instance_indices or [0]
        
        results = {}
        
        logging.info("="*80)
        logging.info("SHAP (SHapley Additive exPlanations) ANALYSIS")
        logging.info("Explaining how each feature contributes to model predictions")
        logging.info("="*80)
        
        for model_name in target_models:
            if model_name not in self.shap_values:
                logging.warning(f"No SHAP values available for {model_name}, skipping...")
                continue
            
            results[model_name] = {}
            
            try:
                # Get data and feature names
                data_key = self._get_data_key(model_name)
                X_set = self.X_data[data_key]
                feature_names = self.feature_names[data_key]
                shap_values = self.shap_values[model_name]
                explainer = self.explainers[model_name]
                
                logging.info(f"\nGenerating explanations for: {model_name.upper()}")
                
                # Create summary plot
                summary_path = self.plot_manager.create_summary_plot(
                    shap_values, X_set, feature_names, model_name
                )
                if summary_path:
                    results[model_name]['summary'] = summary_path
                
                # Create local explanations
                for idx in target_indices:
                    if idx >= len(X_set):
                        logging.warning(f"Instance index {idx} is out of bounds for {model_name}")
                        continue
                    
                    # Force plot
                    force_path = self.plot_manager.create_force_plot(
                        explainer, shap_values, idx, feature_names, model_name
                    )
                    if force_path:
                        results[model_name][f'force_{idx}'] = force_path
                    
                    # Waterfall plot
                    waterfall_path = self.plot_manager.create_waterfall_plot(
                        explainer, shap_values, idx, feature_names, model_name, X_set
                    )
                    if waterfall_path:
                        results[model_name][f'waterfall_{idx}'] = waterfall_path
                
            except Exception as e:
                logging.error(f"Failed to generate explanations for {model_name}: {e}")
        
        return results
    
    def get_feature_importance(self, model_name: str, top_k: int = 10) -> pd.DataFrame:
        """
        Get feature importance based on mean absolute SHAP values.
        
        Parameters
        ----------
        model_name : str
            Name of the model to get feature importance for.
        top_k : int, default=10
            Number of top features to return.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['feature', 'importance'] sorted by importance in descending order.
            Contains the top_k most important features.
            
        Raises
        ------
        ValueError
            If no SHAP values are available for the specified model.
        """
        if model_name not in self.shap_values:
            raise ValueError(f"No SHAP values available for {model_name}")
        
        shap_values = self.shap_values[model_name]
        data_key = self._get_data_key(model_name)
        feature_names = self.feature_names[data_key]
        
        # Calculate mean absolute SHAP values
        importance_scores = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_k)
    
    def export_shap_values(self, output_path: str) -> None:
        """
        Export SHAP values to a file for later analysis.
        
        Parameters
        ----------
        output_path : str
            Path where to save the SHAP values. The file will be saved in compressed numpy format.
            
        Raises
        ------
        Exception
            If the export operation fails due to file system or serialization errors.
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                'shap_values': self.shap_values,
                'feature_names': self.feature_names,
                'model_names': list(self.models.keys())
            }
            
            np.savez_compressed(output_path, **export_data)
            logging.info(f"Exported SHAP values to {output_path}")
            
        except Exception as e:
            logging.error(f"Failed to export SHAP values: {e}")
