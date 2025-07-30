import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

class DynamicThresholding:
    """
    Implements dynamic thresholding at the 75th percentile for taste reasoning.
    Automatically adjusts thresholds based on the distribution of similarity scores.
    """
    
    def __init__(self, percentile: float = 75.0):
        """
        Initialize dynamic thresholding.
        
        Args:
            percentile: Percentile to use for thresholding (default 75th)
        """
        self.percentile = percentile
        self.thresholds = {}
        self.is_fitted = False
        
    def fit_thresholds(self, similarities_df: pd.DataFrame) -> Dict:
        """
        Fit dynamic thresholds based on the distribution of similarity scores.
        
        Args:
            similarities_df: DataFrame with similarity scores
            
        Returns:
            Dictionary with calculated thresholds for each component
        """
        print(f"    🎯 Fitting dynamic thresholds at {self.percentile}th percentile...")
        
        thresholds = {}
        
        # Define components to threshold
        components = {
            'categorical': ['Genres_Similarity', 'Tags_Similarity', 'Cast_Similarity', 'Crew_Similarity'],
            'text': ['Synopsis_Similarity', 'Reviews_Similarity', 'Synopsis_Topics', 'Review_Topics'],
            'semantic': ['Semantic_Similarity']
        }
        
        for component_name, column_names in components.items():
            component_thresholds = {}
            
            for column in column_names:
                if column in similarities_df.columns:
                    values = similarities_df[column].values
                    
                    # Calculate threshold at specified percentile
                    threshold = np.percentile(values, self.percentile)
                    
                    # Store threshold and statistics
                    component_thresholds[column] = {
                        'threshold': threshold,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'percentile': self.percentile
                    }
                    
                    print(f"      ✅ {column}: {threshold:.3f} (mean: {np.mean(values):.3f}, std: {np.std(values):.3f})")
            
            thresholds[component_name] = component_thresholds
        
        self.thresholds = thresholds
        self.is_fitted = True
        
        return thresholds
    
    def get_threshold(self, component: str, feature: str) -> float:
        """
        Get the threshold for a specific component and feature.
        
        Args:
            component: Component name ('categorical', 'text', 'semantic')
            feature: Feature name (e.g., 'Genres_Similarity')
            
        Returns:
            Threshold value
        """
        if not self.is_fitted or component not in self.thresholds:
            return 0.5  # Default threshold
        
        if feature in self.thresholds[component]:
            return self.thresholds[component][feature]['threshold']
        
        return 0.5  # Default threshold
    
    def generate_taste_reasoning(self, similarities: Dict) -> str:
        """
        Generate taste reasoning using dynamic thresholds.
        
        Args:
            similarities: Dictionary with similarity scores
            
        Returns:
            String with taste reasoning
        """
        if not self.is_fitted:
            return "Dynamic thresholds not fitted yet."
        
        reasons = []
        
        # Check categorical similarities
        if 'categorical' in similarities:
            for feature, score in similarities['categorical'].items():
                threshold = self.get_threshold('categorical', f'{feature.capitalize()}_Similarity')
                if score >= threshold:
                    reasons.append(f"high {feature} similarity")
        
        # Check text similarities
        if 'text' in similarities:
            for feature, score in similarities['text'].items():
                threshold = self.get_threshold('text', f'{feature.capitalize()}_Similarity')
                if score >= threshold:
                    reasons.append(f"high {feature} similarity")
        
        # Check semantic similarity
        if 'semantic' in similarities:
            semantic_score = similarities['semantic']
            threshold = self.get_threshold('semantic', 'Semantic_Similarity')
            if semantic_score >= threshold:
                reasons.append("high semantic similarity")
        
        if not reasons:
            return "moderate similarity across components"
        
        return ", ".join(reasons)
    
    def get_threshold_summary(self) -> str:
        """Get a summary of the calculated thresholds."""
        if not self.is_fitted:
            return "No thresholds fitted yet."
        
        summary = f"🎯 DYNAMIC THRESHOLDING SUMMARY ({self.percentile}th percentile)\n"
        summary += "=" * 60 + "\n\n"
        
        for component, features in self.thresholds.items():
            summary += f"🔧 {component.upper()} COMPONENT:\n"
            for feature, stats in features.items():
                summary += f"   {feature}: {stats['threshold']:.3f}\n"
                summary += f"      Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}\n"
                summary += f"      Range: {stats['min']:.3f} - {stats['max']:.3f}\n"
            summary += "\n"
        
        return summary
    
    def analyze_threshold_impact(self, similarities_df: pd.DataFrame) -> str:
        """
        Analyze how many dramas would pass each threshold.
        
        Args:
            similarities_df: DataFrame with similarity scores
            
        Returns:
            Analysis string
        """
        if not self.is_fitted:
            return "No thresholds fitted yet."
        
        analysis = "📊 THRESHOLD IMPACT ANALYSIS\n"
        analysis += "=" * 40 + "\n\n"
        
        total_dramas = len(similarities_df)
        
        for component, features in self.thresholds.items():
            analysis += f"🔧 {component.upper()} COMPONENT:\n"
            
            for feature, stats in features.items():
                if feature in similarities_df.columns:
                    values = similarities_df[feature].values
                    threshold = stats['threshold']
                    
                    # Count dramas above threshold
                    above_threshold = np.sum(values >= threshold)
                    percentage = (above_threshold / total_dramas) * 100
                    
                    analysis += f"   {feature}: {above_threshold}/{total_dramas} ({percentage:.1f}%)\n"
                    analysis += f"      Threshold: {threshold:.3f}\n"
                    analysis += f"      Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}\n"
            
            analysis += "\n"
        
        return analysis
    
    def get_adaptive_thresholds(self, similarities_df: pd.DataFrame, target_percentage: float = 25.0) -> Dict:
        """
        Get adaptive thresholds that ensure a target percentage of dramas pass each threshold.
        
        Args:
            similarities_df: DataFrame with similarity scores
            target_percentage: Target percentage of dramas that should pass each threshold
            
        Returns:
            Dictionary with adaptive thresholds
        """
        print(f"    🎯 Calculating adaptive thresholds for {target_percentage}% pass rate...")
        
        adaptive_thresholds = {}
        
        for component, features in self.thresholds.items():
            component_thresholds = {}
            
            for feature, stats in features.items():
                if feature in similarities_df.columns:
                    values = similarities_df[feature].values
                    
                    # Calculate threshold for target percentage
                    target_percentile = 100 - target_percentage
                    adaptive_threshold = np.percentile(values, target_percentile)
                    
                    component_thresholds[feature] = {
                        'threshold': adaptive_threshold,
                        'target_percentage': target_percentage,
                        'original_threshold': stats['threshold']
                    }
                    
                    print(f"      ✅ {feature}: {adaptive_threshold:.3f} (target: {target_percentage}%)")
            
            adaptive_thresholds[component] = component_thresholds
        
        return adaptive_thresholds 