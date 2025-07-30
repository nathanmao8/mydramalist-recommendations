#!/usr/bin/env python3
"""
Stratified Holdout Evaluator for Taste Analysis
Faster alternative to LOOCV for iterative testing
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from taste_analyzer import TasteAnalyzer
from semantic_similarity import SemanticSimilarityExtractor
import time
from datetime import datetime
from sklearn.model_selection import train_test_split

class TasteStratifiedEvaluator:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.semantic_extractor = SemanticSimilarityExtractor()
        self.taste_analyzer = TasteAnalyzer(self.semantic_extractor)
        self.test_size = test_size
        self.random_state = random_state
        self.results = {}
        
    def run_stratified_evaluation(self, watched_dramas: List[Dict], ratings: List[float]) -> Dict:
        """
        Run stratified holdout evaluation on taste analysis system.
        
        Args:
            watched_dramas: List of drama dictionaries
            ratings: List of corresponding ratings
            test_size: Fraction of data to use for testing (default: 0.2)
            
        Returns:
            Dictionary with evaluation results and metrics
        """
        print("🎯 STRATIFIED HOLDOUT TASTE ANALYSIS EVALUATION")
        print("=" * 60)
        print(f"Testing with {self.test_size*100:.0f}% holdout...")
        
        start_time = time.time()
        
        # Create stratified split based on rating bins
        rating_bins = self._create_rating_bins(ratings)
                
        # Split data
        train_indices, test_indices = train_test_split(
            range(len(watched_dramas)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=rating_bins
        )
        
        # Create train/test sets
        train_dramas = [watched_dramas[i] for i in train_indices]
        train_ratings = [ratings[i] for i in train_indices]
        test_dramas = [watched_dramas[i] for i in test_indices]
        test_ratings = [ratings[i] for i in test_indices]
        
        print(f"Training set: {len(train_dramas)} dramas")
        print(f"Test set: {len(test_dramas)} dramas")
        
        # Build taste profile on training data
        print("\n🔄 Building taste profile on training data...")
        self.taste_analyzer.analyze_user_taste(train_dramas, train_ratings)
        
        # Calculate similarities for test dramas
        print("🔄 Calculating similarities for test dramas...")
        test_similarities = []
        raw_semantic_similarities = []
        for i, test_drama in enumerate(test_dramas):
            print(f"  Processing {test_drama['title']} ({i+1}/{len(test_dramas)})")
            try:
                similarity = self.taste_analyzer.calculate_taste_similarity(test_drama)
                # Get raw (inverted, pre-normalized) semantic similarity
                if hasattr(self.taste_analyzer, 'get_raw_inverted_semantic'):
                    raw_sem = self.taste_analyzer.get_raw_inverted_semantic(test_drama)
                else:
                    raw_sem = similarity['semantic']
                raw_semantic_similarities.append(raw_sem)
                
                # Extract single values from nested dictionaries
                categorical_avg = np.mean(list(similarity['categorical'].values()))
                text_avg = np.mean(list(similarity['text'].values()))
                
                test_similarities.append({
                    'drama_title': test_drama['title'],
                    'actual_rating': test_ratings[i],
                    'overall_similarity': similarity['overall'],
                    'categorical_similarity': categorical_avg,
                    'text_similarity': text_avg,
                    'semantic_similarity': raw_sem  # Temporarily store raw value
                })
            except Exception as e:
                print(f"    ❌ Error processing {test_drama['title']}: {str(e)}")
        # Dynamic normalization for semantic similarity
        if raw_semantic_similarities:
            min_sem = min(raw_semantic_similarities)
            max_sem = max(raw_semantic_similarities)
            for i, row in enumerate(test_similarities):
                raw_val = row['semantic_similarity']
                if max_sem > min_sem:
                    norm_val = (raw_val - min_sem) / (max_sem - min_sem)
                else:
                    norm_val = 0.0
                test_similarities[i]['semantic_similarity'] = norm_val
        
        # Calculate metrics
        self.calculate_metrics(test_similarities)
        
        # Store results
        self.results = {
            'train_size': len(train_dramas),
            'test_size': len(test_dramas),
            'train_avg_rating': np.mean(train_ratings),
            'test_avg_rating': np.mean(test_ratings),
            'test_similarities': test_similarities
        }
        
        # Generate report
        report = self.generate_report()
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Stratified evaluation completed in {elapsed_time:.1f} seconds")
        
        return report
        
    def _create_rating_bins(self, ratings: List[float]) -> List[str]:
        """Create rating bins for stratification."""
        bins = []
        for rating in ratings:
            if rating >= 9.0:
                bins.append('high')
            elif rating >= 7.5:
                bins.append('medium')
            else:
                bins.append('low')
        return bins
    
    def _create_simple_bins(self, ratings: List[float]) -> List[str]:
        """Create simple binary bins for small datasets."""
        median_rating = np.median(ratings)
        bins = []
        for rating in ratings:
            if rating >= median_rating:
                bins.append('high')
            else:
                bins.append('low')
        return bins
    
    def calculate_metrics(self, test_similarities: List[Dict]):
        """Calculate evaluation metrics."""
        if not test_similarities:
            print("❌ No valid test results to analyze")
            return
        
        # Extract data
        actual_ratings = [s['actual_rating'] for s in test_similarities]
        similarities = [s['overall_similarity'] for s in test_similarities]
        
        # Calculate ranking metrics
        self.calculate_ranking_metrics(actual_ratings, similarities)
        
        # Calculate similarity distribution metrics
        self.calculate_similarity_metrics(similarities)
        
        # Calculate component analysis
        self.calculate_component_analysis(test_similarities)
    
    def calculate_ranking_metrics(self, actual_ratings: List[float], similarities: List[float]):
        """Calculate ranking-based evaluation metrics."""
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'actual_rating': actual_ratings,
            'similarity': similarities
        })
        
        # Sort by actual rating and similarity
        df_actual_rank = df.sort_values('actual_rating', ascending=False).reset_index()
        df_similarity_rank = df.sort_values('similarity', ascending=False).reset_index()
        
        # Calculate ranking correlation
        from scipy.stats import spearmanr, kendalltau
        
        spearman_corr, spearman_p = spearmanr(actual_ratings, similarities)
        kendall_corr, kendall_p = kendalltau(actual_ratings, similarities)
        
        # Calculate ranking accuracy for different thresholds
        thresholds = [0.25, 0.33, 0.5]  # Top 25%, 33%, 50%
        ranking_accuracy = {}
        
        for threshold in thresholds:
            top_count = max(1, int(len(df) * threshold))
            
            top_actual = set(df_actual_rank.head(top_count)['index'])
            top_similarity = set(df_similarity_rank.head(top_count)['index'])
            
            precision = len(top_actual & top_similarity) / len(top_similarity) if top_similarity else 0
            recall = len(top_actual & top_similarity) / len(top_actual) if top_actual else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            ranking_accuracy[f'top_{int(threshold*100)}%'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
        
        self.ranking_metrics = {
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'kendall_correlation': kendall_corr,
            'kendall_p_value': kendall_p,
            'ranking_accuracy': ranking_accuracy
        }
    
    def calculate_similarity_metrics(self, similarities: List[float]):
        """Calculate similarity distribution metrics."""
        similarities = np.array(similarities)
        
        self.similarity_metrics = {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'similarity_range': np.max(similarities) - np.min(similarities),
            'high_similarity_count': np.sum(similarities > 0.5),
            'low_similarity_count': np.sum(similarities < 0.2),
            'similarity_variance': np.var(similarities),
            'similarity_skewness': self._calculate_skewness(similarities)
        }
    
    def calculate_component_analysis(self, test_similarities: List[Dict]):
        """Analyze performance of different similarity components."""
        categorical_sims = [s['categorical_similarity'] for s in test_similarities]
        text_sims = [s['text_similarity'] for s in test_similarities]
        semantic_sims = [s['semantic_similarity'] for s in test_similarities]
        actual_ratings = [s['actual_rating'] for s in test_similarities]
        
        # Calculate correlations for each component
        from scipy.stats import spearmanr
        
        components = {
            'categorical': categorical_sims,
            'text': text_sims,
            'semantic': semantic_sims
        }
        
        component_correlations = {}
        for component_name, component_sims in components.items():
            corr, p_value = spearmanr(actual_ratings, component_sims)
            component_correlations[component_name] = {
                'correlation': corr,
                'p_value': p_value
            }
        
        self.component_metrics = {
            'correlations': component_correlations,
            'component_stats': {
                'categorical': {
                    'mean': np.mean(categorical_sims),
                    'std': np.std(categorical_sims)
                },
                'text': {
                    'mean': np.mean(text_sims),
                    'std': np.std(text_sims)
                },
                'semantic': {
                    'mean': np.mean(semantic_sims),
                    'std': np.std(semantic_sims)
                }
            }
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def generate_report(self) -> Dict:
        """Generate comprehensive evaluation report."""
        report = {
            'summary': self.generate_summary(),
            'detailed_results': self.results,
            'ranking_metrics': getattr(self, 'ranking_metrics', {}),
            'similarity_metrics': getattr(self, 'similarity_metrics', {}),
            'component_metrics': getattr(self, 'component_metrics', {})
        }
        
        return report
    
    def generate_summary(self) -> str:
        """Generate text summary of evaluation results."""
        summary = []
        summary.append("🎯 STRATIFIED HOLDOUT EVALUATION RESULTS")
        summary.append("=" * 60)
        
        # Basic statistics
        summary.append(f"\n📊 DATA SPLIT:")
        summary.append(f"   Training Set: {self.results['train_size']} dramas")
        summary.append(f"   Test Set: {self.results['test_size']} dramas")
        summary.append(f"   Train Avg Rating: {self.results['train_avg_rating']:.2f}")
        summary.append(f"   Test Avg Rating: {self.results['test_avg_rating']:.2f}")
        
        # Ranking metrics
        if hasattr(self, 'ranking_metrics'):
            summary.append(f"\n📈 RANKING METRICS:")
            summary.append(f"   Spearman Correlation: {self.ranking_metrics['spearman_correlation']:.3f}")
            summary.append(f"   Kendall Correlation: {self.ranking_metrics['kendall_correlation']:.3f}")
            
            summary.append(f"\n🎯 RANKING ACCURACY:")
            for threshold, metrics in self.ranking_metrics['ranking_accuracy'].items():
                summary.append(f"   {threshold}: Precision {metrics['precision']:.3f}, "
                             f"Recall {metrics['recall']:.3f}, F1 {metrics['f1_score']:.3f}")
        
        # Similarity metrics
        if hasattr(self, 'similarity_metrics'):
            summary.append(f"\n📊 SIMILARITY DISTRIBUTION:")
            summary.append(f"   Mean Similarity: {self.similarity_metrics['mean_similarity']:.3f}")
            summary.append(f"   Std Similarity: {self.similarity_metrics['std_similarity']:.3f}")
            summary.append(f"   Similarity Range: {self.similarity_metrics['min_similarity']:.3f} - {self.similarity_metrics['max_similarity']:.3f}")
            summary.append(f"   High Similarity (>0.5): {self.similarity_metrics['high_similarity_count']} dramas")
            summary.append(f"   Low Similarity (<0.2): {self.similarity_metrics['low_similarity_count']} dramas")
            summary.append(f"   Skewness: {self.similarity_metrics['similarity_skewness']:.3f}")
        
        # Component analysis
        if hasattr(self, 'component_metrics'):
            summary.append(f"\n🔧 COMPONENT ANALYSIS:")
            for component, metrics in self.component_metrics['correlations'].items():
                summary.append(f"   {component.capitalize()}: Correlation {metrics['correlation']:.3f} (p={metrics['p_value']:.3f})")
        
        return "\n".join(summary)
    
    def save_results(self, filename: str = None):
        """Save evaluation results to files."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stratified_results_{timestamp}"
        
        # Save detailed results
        df = pd.DataFrame(self.results['test_similarities'])
        df.to_csv(f"{filename}.csv", index=False)
        
        # Save summary report
        report = self.generate_report()
        with open(f"{filename}_report.txt", 'w') as f:
            f.write(report['summary'])
        
        print(f"\n✅ Stratified evaluation results saved:")
        print(f"   • {filename}.csv - Detailed test results")
        print(f"   • {filename}_report.txt - Summary report")
    
    def plot_results(self, save_path: str = None):
        """Create visualizations of evaluation results."""
        test_similarities = self.results['test_similarities']
        
        if not test_similarities:
            print("❌ No test results to plot")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Similarity scatter plot
        actual_ratings = [s['actual_rating'] for s in test_similarities]
        similarities = [s['overall_similarity'] for s in test_similarities]
        
        axes[0, 0].scatter(actual_ratings, similarities, alpha=0.7)
        axes[0, 0].set_xlabel('Actual Rating')
        axes[0, 0].set_ylabel('Similarity Score')
        axes[0, 0].set_title('Actual Rating vs Similarity Score')
        
        # Add trend line
        z = np.polyfit(actual_ratings, similarities, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(actual_ratings, p(actual_ratings), "r--", alpha=0.8)
        
        # 2. Similarity distribution
        axes[0, 1].hist(similarities, bins=10, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Similarity Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Similarity Score Distribution')
        
        # 3. Component similarities
        categorical_sims = [s['categorical_similarity'] for s in test_similarities]
        text_sims = [s['text_similarity'] for s in test_similarities]
        semantic_sims = [s['semantic_similarity'] for s in test_similarities]
        
        component_data = [categorical_sims, text_sims, semantic_sims]
        component_labels = ['Categorical', 'Text', 'Semantic']
        
        axes[1, 0].boxplot(component_data, labels=component_labels)
        axes[1, 0].set_ylabel('Similarity Score')
        axes[1, 0].set_title('Similarity by Component')
        
        # 4. Rating distribution
        axes[1, 1].hist(actual_ratings, bins=8, alpha=0.7, edgecolor='black', color='green')
        axes[1, 1].set_xlabel('Actual Rating')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Actual Rating Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Example usage of stratified evaluator."""
    from taste_profile_example import load_sample_data
    
    print("🎯 STRATIFIED HOLDOUT EVALUATION")
    print("=" * 50)
    
    # Load sample data
    watched_dramas, unwatched_dramas, ratings = load_sample_data()
    
    # Initialize evaluator
    evaluator = TasteStratifiedEvaluator(test_size=0.3, random_state=42)
    
    # Run evaluation
    results = evaluator.run_stratified_evaluation(watched_dramas, ratings)
    
    # Print summary
    print("\n" + results['summary'])
    
    # Save results
    evaluator.save_results()
    
    # Create plots
    evaluator.plot_results('stratified_visualization.png')
    
    print("\n✅ Stratified evaluation complete!")

if __name__ == "__main__":
    main() 