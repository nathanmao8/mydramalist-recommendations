import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PerformanceBasedWeighting:
    """
    Determines optimal component weights based on performance correlation with user ratings.
    """
    
    def __init__(self, min_weight: float = 0.1, max_weight: float = 0.6):
        """
        Initialize the performance-based weighting system.
        
        Args:
            min_weight: Minimum weight for any component
            max_weight: Maximum weight for any component
        """
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.optimal_weights = None
        self.performance_metrics = {}
        
    def calculate_component_performance(self, similarities_df: pd.DataFrame, ratings: List[float]) -> Dict:
        """
        Calculate performance metrics for each component.
        
        Args:
            similarities_df: DataFrame with similarity scores for each component
            ratings: List of user ratings corresponding to the dramas
            
        Returns:
            Dictionary with performance metrics for each component
        """
        if len(similarities_df) != len(ratings):
            raise ValueError("Number of dramas must match number of ratings")
        
        performance_metrics = {}
        
        # Calculate performance for each component
        components = {
            'categorical': 'Categorical_Similarity',
            'text': 'Text_Similarity', 
            'semantic': 'Semantic_Similarity'
        }
        
        for component_name, column_name in components.items():
            if column_name in similarities_df.columns:
                similarities = similarities_df[column_name].values
                
                # Calculate correlation metrics
                spearman_corr, spearman_p = spearmanr(similarities, ratings)
                kendall_corr, kendall_p = kendalltau(similarities, ratings)
                
                # Calculate ranking accuracy (how well it ranks high-rated dramas)
                ranking_score = self._calculate_ranking_accuracy(similarities, ratings)
                
                # Calculate linear regression R²
                r_squared = self._calculate_r_squared(similarities, ratings)
                
                performance_metrics[component_name] = {
                    'spearman_correlation': spearman_corr,
                    'spearman_p_value': spearman_p,
                    'kendall_correlation': kendall_corr,
                    'kendall_p_value': kendall_p,
                    'ranking_accuracy': ranking_score,
                    'r_squared': r_squared,
                    'mean_similarity': np.mean(similarities),
                    'std_similarity': np.std(similarities)
                }
        
        self.performance_metrics = performance_metrics
        return performance_metrics
    
    def _calculate_ranking_accuracy(self, similarities: np.ndarray, ratings: List[float]) -> float:
        """
        Calculate how well the similarity scores rank high-rated dramas.
        
        Args:
            similarities: Array of similarity scores
            ratings: List of user ratings
            
        Returns:
            Ranking accuracy score (0-1)
        """
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'similarity': similarities,
            'rating': ratings
        })
        
        # Sort by similarity (descending)
        df_sorted = df.sort_values('similarity', ascending=False)
        
        # Calculate top 1/3 ranking accuracy
        top_threshold = len(df) // 3
        top_dramas = df_sorted.head(top_threshold)
        
        # Calculate how many high-rated dramas (rating >= 7) are in top 1/3
        high_rated_threshold = 7.0
        high_rated_in_top = len(top_dramas[top_dramas['rating'] >= high_rated_threshold])
        total_high_rated = len(df[df['rating'] >= high_rated_threshold])
        
        if total_high_rated == 0:
            return 0.0
        
        ranking_accuracy = high_rated_in_top / total_high_rated
        return ranking_accuracy
    
    def _calculate_r_squared(self, similarities: np.ndarray, ratings: List[float]) -> float:
        """
        Calculate R² from linear regression.
        
        Args:
            similarities: Array of similarity scores
            ratings: List of user ratings
            
        Returns:
            R² score
        """
        try:
            # Reshape for sklearn
            X = similarities.reshape(-1, 1)
            y = np.array(ratings)
            
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(X, y)
            
            # Calculate R²
            r_squared = reg.score(X, y)
            return max(0.0, r_squared)  # Ensure non-negative
        except:
            return 0.0
    
    def determine_optimal_weights(self, performance_metrics: Dict) -> Dict:
        """
        Determine optimal weights based on performance metrics.
        
        Args:
            performance_metrics: Dictionary with performance metrics for each component
            
        Returns:
            Dictionary with optimal weights for each component
        """
        if not performance_metrics:
            return {'categorical': 0.45, 'text': 0.20, 'semantic': 0.35}
        
        # Calculate composite performance score for each component
        component_scores = {}
        
        for component, metrics in performance_metrics.items():
            # Combine multiple metrics into a single score
            spearman_score = abs(metrics['spearman_correlation'])
            kendall_score = abs(metrics['kendall_correlation'])
            ranking_score = metrics['ranking_accuracy']
            r_squared_score = metrics['r_squared']
            
            # Weighted combination (can be adjusted)
            composite_score = (
                0.3 * spearman_score +
                0.3 * kendall_score +
                0.2 * ranking_score +
                0.2 * r_squared_score
            )
            
            component_scores[component] = composite_score
        
        # Normalize scores to weights
        total_score = sum(component_scores.values())
        
        if total_score == 0:
            # Fallback to equal weights
            weights = {component: 1.0/len(component_scores) for component in component_scores}
        else:
            # Calculate proportional weights
            weights = {}
            for component, score in component_scores.items():
                weight = score / total_score
                # Apply min/max constraints
                weight = np.clip(weight, self.min_weight, self.max_weight)
                weights[component] = weight
            
            # Renormalize to sum to 1.0
            total_weight = sum(weights.values())
            weights = {component: weight/total_weight for component, weight in weights.items()}
        
        self.optimal_weights = weights
        return weights
    
    def analyze_component_performance(self, performance_metrics: Dict) -> str:
        """
        Generate a detailed analysis of component performance.
        
        Args:
            performance_metrics: Dictionary with performance metrics
            
        Returns:
            Formatted analysis string
        """
        analysis = "🎯 COMPONENT PERFORMANCE ANALYSIS\n"
        analysis += "=" * 50 + "\n\n"
        
        for component, metrics in performance_metrics.items():
            analysis += f"📊 {component.upper()} COMPONENT:\n"
            analysis += f"   Spearman Correlation: {metrics['spearman_correlation']:.3f} (p={metrics['spearman_p_value']:.3f})\n"
            analysis += f"   Kendall Correlation: {metrics['kendall_correlation']:.3f} (p={metrics['kendall_p_value']:.3f})\n"
            analysis += f"   Ranking Accuracy: {metrics['ranking_accuracy']:.3f}\n"
            analysis += f"   R² Score: {metrics['r_squared']:.3f}\n"
            analysis += f"   Mean Similarity: {metrics['mean_similarity']:.3f}\n"
            analysis += f"   Std Similarity: {metrics['std_similarity']:.3f}\n\n"
        
        if self.optimal_weights:
            analysis += "⚖️ OPTIMAL WEIGHTS:\n"
            analysis += "-" * 30 + "\n"
            for component, weight in self.optimal_weights.items():
                analysis += f"   {component.capitalize()}: {weight:.3f} ({weight*100:.1f}%)\n"
        
        return analysis
    
    def get_weighting_recommendations(self, performance_metrics: Dict) -> List[str]:
        """
        Get specific recommendations for weight adjustments.
        
        Args:
            performance_metrics: Dictionary with performance metrics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        for component, metrics in performance_metrics.items():
            spearman_corr = metrics['spearman_correlation']
            ranking_acc = metrics['ranking_accuracy']
            
            if abs(spearman_corr) < 0.1:
                recommendations.append(f"⚠️ {component.capitalize()} component shows weak correlation ({spearman_corr:.3f})")
            elif spearman_corr > 0.3:
                recommendations.append(f"✅ {component.capitalize()} component shows strong positive correlation ({spearman_corr:.3f})")
            elif spearman_corr < -0.3:
                recommendations.append(f"🔄 {component.capitalize()} component shows strong negative correlation ({spearman_corr:.3f}) - consider inverting")
            
            if ranking_acc < 0.2:
                recommendations.append(f"⚠️ {component.capitalize()} component has low ranking accuracy ({ranking_acc:.3f})")
            elif ranking_acc > 0.5:
                recommendations.append(f"✅ {component.capitalize()} component has high ranking accuracy ({ranking_acc:.3f})")
        
        return recommendations 