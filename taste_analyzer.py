import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from taste_profile import UserTasteProfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from semantic_similarity import SemanticSimilarityExtractor
from dynamic_thresholding import DynamicThresholding

class TasteAnalyzer:
    """
    Analyzes user taste profiles and provides recommendations based on taste similarity.
    Integrates with the existing prediction system to enhance recommendations.
    """
    
    def __init__(self, semantic_extractor=None, topic_extractor=None):
        """
        Initialize the taste analyzer.
        
        Args:
            semantic_extractor: SemanticSimilarityExtractor instance
            topic_extractor: TopicModelingExtractor instance
        """
        self.semantic_extractor = semantic_extractor
        self.topic_extractor = topic_extractor
        self.taste_profile = UserTasteProfile(semantic_extractor, topic_extractor)
        self.analysis_results = {}
        self.dynamic_thresholding = DynamicThresholding(percentile=75.0)
        self.use_dynamic_thresholding = False
        
    def analyze_user_taste(self, watched_dramas: List[Dict], ratings: List[float]) -> Dict:
        """
        Build and analyze the user's taste profile.
        
        Args:
            watched_dramas: List of watched drama dictionaries
            ratings: List of user ratings
            
        Returns:
            Dictionary containing taste analysis results
        """
        print("\n" + "="*80)
        print("USER TASTE PROFILE ANALYSIS")
        print("="*80)
        
        # Build taste profile
        profile = self.taste_profile.build_taste_profile(watched_dramas, ratings)
        
        # Get insights
        insights = self.taste_profile.get_taste_insights()
        
        # Display analysis
        self._display_taste_analysis(insights, profile)
        
        # Store results
        self.analysis_results = {
            'profile': profile,
            'insights': insights
        }
        
        return self.analysis_results
    
    def perform_performance_weighting(self, watched_dramas: List[Dict], ratings: List[float]) -> Dict:
        """
        Perform performance-based weighting analysis and update component weights.
        
        Args:
            watched_dramas: List of watched drama dictionaries
            ratings: List of user ratings
            
        Returns:
            Dictionary with updated weights and analysis
        """
        print("\n" + "="*80)
        print("PERFORMANCE-BASED WEIGHTING ANALYSIS")
        print("="*80)
        
        # Calculate similarities for watched dramas using current weights
        similarities_data = []
        for drama in watched_dramas:
            similarities = self.taste_profile.calculate_taste_similarity(drama)
            drama_data = {
                'Drama_Title': drama.get('title', ''),
                'Drama_ID': drama.get('slug', ''),
                'Categorical_Similarity': np.mean(list(similarities['categorical'].values())),
                'Text_Similarity': np.mean(list(similarities['text'].values())),
                'Semantic_Similarity': similarities['semantic'],
                'Overall_Taste_Similarity': similarities['overall']
            }
            similarities_data.append(drama_data)
        
        similarities_df = pd.DataFrame(similarities_data)
        
        # Perform performance-based weighting
        optimal_weights = self.taste_profile.update_component_weights(similarities_df, ratings)
        
        # Recalculate similarities with new weights
        print("\n🔄 Recalculating similarities with optimized weights...")
        updated_similarities_data = []
        for drama in watched_dramas:
            similarities = self.taste_profile.calculate_taste_similarity(drama)
            drama_data = {
                'Drama_Title': drama.get('title', ''),
                'Drama_ID': drama.get('slug', ''),
                'Categorical_Similarity': np.mean(list(similarities['categorical'].values())),
                'Text_Similarity': np.mean(list(similarities['text'].values())),
                'Semantic_Similarity': similarities['semantic'],
                'Overall_Taste_Similarity': similarities['overall']
            }
            updated_similarities_data.append(drama_data)
        
        updated_similarities_df = pd.DataFrame(updated_similarities_data)
        
        # Compare performance before and after
        print("\n�� PERFORMANCE COMPARISON:")
        print("-" * 40)
        
        # Calculate correlation improvements
        for component in ['Categorical_Similarity', 'Text_Similarity', 'Semantic_Similarity', 'Overall_Taste_Similarity']:
            before_corr = np.corrcoef(similarities_df[component], ratings)[0, 1]
            after_corr = np.corrcoef(updated_similarities_df[component], ratings)[0, 1]
            improvement = after_corr - before_corr
            print(f"   {component}: {before_corr:.3f} → {after_corr:.3f} (Δ: {improvement:+.3f})")
        
        return {
            'optimal_weights': optimal_weights,
            'before_similarities': similarities_df,
            'after_similarities': updated_similarities_df,
            'ratings': ratings
        }
    
    def calculate_taste_similarities(self, unwatched_dramas: List[Dict]) -> pd.DataFrame:
        """
        Calculate taste similarity scores for all unwatched dramas.
        
        Args:
            unwatched_dramas: List of unwatched drama dictionaries
            
        Returns:
            DataFrame with similarity scores for each drama
        """
        print(f"\nCalculating taste similarities for {len(unwatched_dramas)} unwatched dramas...")
        
        similarity_data = []
        raw_semantic_similarities = []  # Store raw (inverted, pre-normalized) semantic similarities
        
        # First pass: collect all similarities and raw semantic similarities
        for i, drama in enumerate(unwatched_dramas):
            if i % 100 == 0:
                print(f"  Processing drama {i+1}/{len(unwatched_dramas)}")
            
            similarities = self.taste_profile.calculate_taste_similarity(drama)
            # Extract the raw (inverted, pre-normalized) semantic similarity
            raw_sem = self.taste_profile.get_raw_inverted_semantic(drama)
            raw_semantic_similarities.append(raw_sem)
            
            drama_data = {
                'Drama_Title': drama.get('title', ''),
                'Drama_ID': drama.get('slug', ''),
                'Overall_Taste_Similarity': similarities['overall'],
                'Categorical_Similarity': np.mean(list(similarities['categorical'].values())),
                'Text_Similarity': np.mean(list(similarities['text'].values())),
                'Semantic_Similarity': raw_sem,  # Temporarily store raw value
            }
            # Extract individual categorical similarities
            for category, score in similarities['categorical'].items():
                drama_data[f'{category}_similarity'] = score
            # Extract individual text similarities
            for text_type, score in similarities['text'].items():
                drama_data[f'{text_type}_similarity'] = score
            # Calculate crew similarity (average of directors, screenwriters, composers)
            crew_scores = [
                similarities['categorical'].get('directors', 0),
                similarities['categorical'].get('screenwriters', 0),
                similarities['categorical'].get('composers', 0)
            ]
            drama_data['crew_similarity'] = np.mean(crew_scores)
            drama_data['Taste_Reasoning'] = self._generate_taste_reasoning(similarities)
            similarity_data.append(drama_data)
        # Compute min/max for normalization
        min_sem = min(raw_semantic_similarities)
        max_sem = max(raw_semantic_similarities)
        # Second pass: normalize
        for i, row in enumerate(similarity_data):
            raw_val = row['Semantic_Similarity']
            if max_sem > min_sem:
                norm_val = (raw_val - min_sem) / (max_sem - min_sem)
            else:
                norm_val = 0.0
            similarity_data[i]['Semantic_Similarity'] = norm_val
        df = pd.DataFrame(similarity_data)
        df = df.sort_values('Overall_Taste_Similarity', ascending=False)
        print(f"Taste similarity calculation complete")
        return df

    # Helper to get raw (inverted, pre-normalized) semantic similarity for a drama
    def get_raw_inverted_semantic(self, drama: Dict) -> float:
        """
        Returns the raw (inverted, pre-normalized) semantic similarity for a drama.
        """
        if not hasattr(self, 'taste_profile') or not self.taste_profile.taste_profile:
            return 0.0
        semantic_profile = self.taste_profile.taste_profile.get('semantic', {})
        if not self.semantic_extractor or not semantic_profile:
            return 0.0
        training_dramas = getattr(self.taste_profile, 'training_dramas', [])
        if not training_dramas:
            return 0.0
        drama_semantic_features = self.semantic_extractor.extract_single_drama_semantic_features(drama, training_dramas)
        similarities = []
        feature_names = [
            'avg_synopsis_similarity', 'max_synopsis_similarity', 'max_review_similarity'
        ]
        for i, feature in enumerate(feature_names):
            if feature in semantic_profile:
                drama_val = drama_semantic_features[i]
                profile_val = semantic_profile[feature]
                max_val = max(abs(drama_val), abs(profile_val))
                if max_val > 0:
                    similarity = 1 - abs(drama_val - profile_val) / max_val
                else:
                    similarity = 1.0
                similarities.append(similarity)
        semantic_similarity = np.mean(similarities) if similarities else 0.0
        inverted_semantic = 1.0 - semantic_similarity
        return inverted_semantic
    
    def enhance_predictions_with_taste(self, predictions_df: pd.DataFrame, 
                                     taste_similarities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance existing predictions with taste similarity scores.
        
        Args:
            predictions_df: DataFrame with model predictions
            taste_similarities_df: DataFrame with taste similarity scores
            
        Returns:
            Enhanced DataFrame with taste-adjusted predictions
        """
        print("\nEnhancing predictions with taste similarity...")
        
        # Merge predictions with taste similarities
        enhanced_df = predictions_df.merge(
            taste_similarities_df[['Drama_ID', 'Overall_Taste_Similarity']], 
            on='Drama_ID', 
            how='left'
        )
        
        # Calculate taste-adjusted predictions
        # Blend model predictions with taste similarity
        taste_weight = 0.3  # How much to weight taste similarity vs model predictions
        
        enhanced_df['Taste_Adjusted_Prediction'] = (
            (1 - taste_weight) * enhanced_df['Final_Prediction'] + 
            taste_weight * enhanced_df['Overall_Taste_Similarity'] * 10  # Scale to 0-10
        )
        
        # Calculate confidence boost from taste alignment
        enhanced_df['Taste_Confidence_Boost'] = (
            enhanced_df['Confidence_Score'] * 
            (1 + enhanced_df['Overall_Taste_Similarity'] * 0.5)  # Boost confidence by up to 50%
        )
        
        # Sort by taste-adjusted prediction
        enhanced_df = enhanced_df.sort_values('Taste_Adjusted_Prediction', ascending=False)
        
        print("Predictions enhanced with taste similarity")
        return enhanced_df
    
    def get_taste_recommendations(self, unwatched_dramas: List[Dict], 
                                top_n: int = 20) -> pd.DataFrame:
        """
        Get top recommendations based purely on taste similarity.
        
        Args:
            unwatched_dramas: List of unwatched dramas
            top_n: Number of top recommendations to return
            
        Returns:
            DataFrame with top taste-based recommendations
        """
        print(f"\nGenerating top {top_n} taste-based recommendations...")
        
        # Calculate taste similarities (already includes Taste_Reasoning)
        similarities_df = self.calculate_taste_similarities(unwatched_dramas)
        
        # Get top recommendations
        top_recommendations = similarities_df.head(top_n).copy()
        
        return top_recommendations
    
    def fit_dynamic_thresholds(self, similarities_df: pd.DataFrame) -> Dict:
        """
        Fit dynamic thresholds based on similarity score distributions.
        
        Args:
            similarities_df: DataFrame with similarity scores
            
        Returns:
            Dictionary with fitted thresholds
        """
        print("\n" + "="*80)
        print("DYNAMIC THRESHOLDING")
        print("="*80)
        
        thresholds = self.dynamic_thresholding.fit_thresholds(similarities_df)
        self.use_dynamic_thresholding = True
        
        print("\n📊 Threshold Summary:")
        print(self.dynamic_thresholding.get_threshold_summary())
        
        print("\n📈 Threshold Impact Analysis:")
        print(self.dynamic_thresholding.analyze_threshold_impact(similarities_df))
        
        return thresholds
    
    def _generate_taste_reasoning(self, similarities: Dict) -> str:
        """
        Generate taste reasoning using either fixed or dynamic thresholds.
        
        Args:
            similarities: Dictionary with similarity scores
            
        Returns:
            String with taste reasoning
        """
        if self.use_dynamic_thresholding:
            return self.dynamic_thresholding.generate_taste_reasoning(similarities)
        
        # Fallback to fixed thresholds
        reasons = []
        
        # Check categorical similarities with fixed thresholds
        if 'categorical' in similarities:
            for feature, score in similarities['categorical'].items():
                if feature == 'genres' and score >= 0.3:
                    reasons.append("high genre similarity")
                elif feature == 'tags' and score >= 0.25:
                    reasons.append("high tag similarity")
                elif feature == 'cast' and score >= 0.2:
                    reasons.append("high cast similarity")
                elif feature == 'crew' and score >= 0.15:
                    reasons.append("high crew similarity")
        
        # Check text similarities with fixed thresholds
        if 'text' in similarities:
            for feature, score in similarities['text'].items():
                if feature == 'synopsis_embedding' and score >= 0.4:
                    reasons.append("high synopsis similarity")
                elif feature == 'review_embedding' and score >= 0.35:
                    reasons.append("high review similarity")
                elif feature == 'synopsis_topics' and score >= 0.3:
                    reasons.append("high synopsis topic similarity")
                elif feature == 'review_topics' and score >= 0.3:
                    reasons.append("high review topic similarity")
        
        # Check semantic similarity with fixed threshold
        if 'semantic' in similarities:
            semantic_score = similarities['semantic']
            if semantic_score >= 0.6:
                reasons.append("high semantic similarity")
        
        if not reasons:
            return "moderate similarity across components"
        
        return ", ".join(reasons)
    
    def _display_taste_analysis(self, insights: Dict, profile: Dict):
        """Display comprehensive taste analysis."""
        
        print("\n📊 TASTE PROFILE INSIGHTS")
        print("-" * 50)
        
        # Rating patterns
        rating_patterns = insights['rating_patterns']
        print(f"📈 Rating Patterns:")
        print(f"   Average Rating: {rating_patterns['average_rating']:.2f}/10")
        print(f"   Rating Consistency: {rating_patterns['rating_consistency']:.2f}")
        print(f"   High-Rated Threshold: {rating_patterns['high_rated_threshold']:.1f}+")
        
        # Rating distribution
        dist = rating_patterns['rating_distribution']
        print(f"   Rating Distribution:")
        for range_key, count in dist.items():
            print(f"     {range_key}: {count} dramas")
        
        # Top preferences
        print(f"\n🎭 Top Preferences:")
        top_prefs = insights['top_preferences']
        
        for category, items in top_prefs.items():
            if items:
                print(f"   {category.title()}:")
                for item, score in items[:5]:  # Top 5
                    print(f"     • {item} (score: {score:.3f})")
        
        # Content preferences
        if 'content_preferences' in insights:
            content_prefs = insights['content_preferences']
            print(f"\n📝 Content Preferences:")
            # Removed sentiment-related preferences display
            if not content_prefs:
                print(f"   No specific content preferences identified")
        
        print(f"\n📊 Profile Statistics:")
        rating_patterns = insights.get('rating_patterns', {})
        print(f"   Average Rating: {rating_patterns.get('average_rating', 'N/A'):.2f}")
        print(f"   Rating Consistency: {rating_patterns.get('rating_consistency', 'N/A'):.2f}")
        print(f"   High-Rated Threshold: {rating_patterns.get('high_rated_threshold', 'N/A'):.1f}+")
    
    def save_taste_analysis(self, filepath: str = 'taste_analysis_results.csv'):
        """Save taste analysis results to CSV."""
        if not self.analysis_results:
            print("No analysis results to save. Run analyze_user_taste() first.")
            return
        
        # Save taste profile
        self.taste_profile.save_profile('user_taste_profile.json')
        
        # Save insights summary
        insights = self.analysis_results['insights']
        
        # Create summary DataFrame
        summary_data = []
        
        # Rating patterns
        rating_patterns = insights['rating_patterns']
        summary_data.append({
            'Category': 'Rating_Patterns',
            'Metric': 'Average_Rating',
            'Value': rating_patterns['average_rating']
        })
        summary_data.append({
            'Category': 'Rating_Patterns',
            'Metric': 'Rating_Consistency',
            'Value': rating_patterns['rating_consistency']
        })
        
        # Top preferences (flatten)
        for category, items in insights['top_preferences'].items():
            for i, (item, score) in enumerate(items[:5]):  # Top 5
                summary_data.append({
                    'Category': f'Top_{category.title()}',
                    'Metric': f'Rank_{i+1}',
                    'Value': f"{item} ({score:.3f})"
                })
        
        # Content preferences
        if 'content_preferences' in insights:
            content_prefs = insights['content_preferences']
            for key, value in content_prefs.items():
                summary_data.append({
                    'Category': 'Content_Preferences',
                    'Metric': key,
                    'Value': value
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filepath, index=False)
        print(f"Taste analysis saved to {filepath}")
    
    def compare_taste_vs_predictions(self, taste_similarities_df: pd.DataFrame, 
                                   predictions_df: pd.DataFrame) -> Dict:
        """
        Compare taste-based recommendations with model predictions.
        
        Args:
            taste_similarities_df: DataFrame with taste similarity scores
            predictions_df: DataFrame with model predictions
            
        Returns:
            Dictionary with comparison analysis
        """
        print("\nComparing taste-based vs model-based recommendations...")
        
        # Merge dataframes
        comparison_df = taste_similarities_df.merge(
            predictions_df[['Drama_ID', 'Final_Prediction', 'Confidence_Score']], 
            on='Drama_ID', 
            how='inner'
        )
        
        # Calculate correlations
        correlations = {
            'taste_vs_prediction': comparison_df['Overall_Taste_Similarity'].corr(comparison_df['Final_Prediction']),
            'taste_vs_confidence': comparison_df['Overall_Taste_Similarity'].corr(comparison_df['Confidence_Score'])
        }
        
        # Find dramas where taste and predictions disagree
        taste_threshold = comparison_df['Overall_Taste_Similarity'].quantile(0.8)  # Top 20% taste
        pred_threshold = comparison_df['Final_Prediction'].quantile(0.8)  # Top 20% predictions
        
        high_taste_low_pred = comparison_df[
            (comparison_df['Overall_Taste_Similarity'] >= taste_threshold) & 
            (comparison_df['Final_Prediction'] < pred_threshold)
        ]
        
        low_taste_high_pred = comparison_df[
            (comparison_df['Overall_Taste_Similarity'] < taste_threshold) & 
            (comparison_df['Final_Prediction'] >= pred_threshold)
        ]
        
        # Analysis results
        analysis = {
            'correlations': correlations,
            'high_taste_low_prediction': high_taste_low_pred[['Drama_Title', 'Overall_Taste_Similarity', 'Final_Prediction']].head(10),
            'low_taste_high_prediction': low_taste_high_pred[['Drama_Title', 'Overall_Taste_Similarity', 'Final_Prediction']].head(10),
            'total_dramas_compared': len(comparison_df)
        }
        
        # Display results
        print(f"\n📊 TASTE vs PREDICTION COMPARISON")
        print("-" * 50)
        print(f"Correlation (Taste vs Predictions): {correlations['taste_vs_prediction']:.3f}")
        print(f"Correlation (Taste vs Confidence): {correlations['taste_vs_confidence']:.3f}")
        print(f"Total Dramas Compared: {analysis['total_dramas_compared']}")
        
        print(f"\n🎯 High Taste, Low Prediction (Hidden Gems?):")
        for _, row in analysis['high_taste_low_prediction'].iterrows():
            print(f"   • {row['Drama_Title']} (Taste: {row['Overall_Taste_Similarity']:.3f}, Pred: {row['Final_Prediction']:.2f})")
        
        print(f"\n⚠️  Low Taste, High Prediction (Overrated?):")
        for _, row in analysis['low_taste_high_prediction'].iterrows():
            print(f"   • {row['Drama_Title']} (Taste: {row['Overall_Taste_Similarity']:.3f}, Pred: {row['Final_Prediction']:.2f})")
        
        return analysis 
    
    def calculate_taste_similarity(self, drama: Dict) -> Dict:
        """
        Calculate similarity between a drama and the user's taste profile.
        Wrapper around UserTasteProfile's calculate_taste_similarity method.
        
        Args:
            drama: Drama dictionary with features
            
        Returns:
            Dictionary with similarity scores for different aspects
        """
        if not hasattr(self, 'taste_profile') or not self.taste_profile.taste_profile:
            raise ValueError("Taste profile not built. Call analyze_user_taste() first.")
        
        # Use UserTasteProfile's method directly
        return self.taste_profile.calculate_taste_similarity(drama)
    
