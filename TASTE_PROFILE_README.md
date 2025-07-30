# User Taste Profile System

## Overview

The User Taste Profile System is a comprehensive recommendation engine that builds personalized taste profiles from your watched dramas and provides similarity-based recommendations for unwatched content. It integrates seamlessly with your existing drama rating prediction system.

## Features

### 🎯 **Taste Profile Building**

- **Categorical Preferences**: Analyzes your preferences for genres, tags, actors, directors, screenwriters, and composers
- **Text Content Preferences**: Creates embeddings of synopsis and review content you've enjoyed
- **Semantic Patterns**: Identifies patterns in synopsis similarity and review similarity
- **Rating Patterns**: Analyzes your rating distribution and consistency

### 🔍 **Similarity Scoring**

- **Multi-dimensional Analysis**: Compares unwatched dramas across categorical, text, and semantic dimensions
- **Weighted Scoring**: Higher-rated dramas have more influence on your taste profile
- **Cosine Similarity**: Uses advanced similarity metrics for text and semantic features
- **Overall Similarity**: Combines all dimensions into a single similarity score

### 📊 **Recommendation Engine**

- **Pure Taste Recommendations**: Recommendations based solely on taste similarity
- **Enhanced Predictions**: Combines ML predictions with taste similarity
- **Reasoning Generation**: Explains why each recommendation matches your taste
- **Hidden Gems Discovery**: Finds dramas that align with your taste but may be overlooked

### 📈 **Analysis & Insights**

- **Taste Interpretation**: Shows your top preferences in each category
- **Rating Analysis**: Reveals your rating patterns and consistency
- **Content Preferences**: Identifies your thematic and structural preferences
- **Comparison Analysis**: Compares taste-based vs model-based recommendations

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Watched       │    │   Taste Profile  │    │   Unwatched     │
│   Dramas        │───▶│   Builder        │───▶│   Dramas        │
│   + Ratings     │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Taste          │    │   Similarity    │
                       │   Analysis       │    │   Calculator    │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Insights &     │    │   Taste-Based   │
                       │   Interpretations│    │   Recommendations│
                       └──────────────────┘    └─────────────────┘
```

## Components

### 1. `UserTasteProfile` (`taste_profile.py`)

Core class that builds and manages taste profiles.

**Key Methods:**

- `build_taste_profile()`: Creates comprehensive taste profile from watched dramas
- `calculate_taste_similarity()`: Computes similarity between drama and taste profile
- `get_taste_insights()`: Provides interpretable insights about preferences

### 2. `TasteAnalyzer` (`taste_analyzer.py`)

High-level interface for taste analysis and recommendations.

**Key Methods:**

- `analyze_user_taste()`: Builds and analyzes taste profile
- `calculate_taste_similarities()`: Scores all unwatched dramas
- `enhance_predictions_with_taste()`: Combines ML predictions with taste
- `get_taste_recommendations()`: Pure taste-based recommendations

## Usage

### Basic Usage

```python
from taste_analyzer import TasteAnalyzer
from semantic_similarity import SemanticSimilarityExtractor

# Initialize
semantic_extractor = SemanticSimilarityExtractor()
taste_analyzer = TasteAnalyzer(semantic_extractor)

# Build taste profile
watched_ratings = [d.get('user_rating', 0) for d in watched_dramas]
taste_analysis = taste_analyzer.analyze_user_taste(watched_dramas, watched_ratings)

# Get recommendations
recommendations = taste_analyzer.get_taste_recommendations(unwatched_dramas, top_n=20)
```

### Integration with Main System

The taste profile system is automatically integrated into your main prediction pipeline:

```bash
# Run with taste profile analysis enabled (default)
python main.py --user_id Oamen --enable-taste-profile

# Adjust taste weight in predictions
python main.py --user_id Oamen --taste-weight 0.4

# Disable taste profile analysis
python main.py --user_id Oamen --no-enable-taste-profile
```

### Standalone Example

Run the example script to see the system in action:

```bash
python taste_profile_example.py
```

## Output Files

When you run the system, it generates several files:

### 📄 **Core Files**

- `drama_predictions.csv` - Enhanced predictions with taste adjustment
- `user_taste_profile.json` - Complete taste profile (can be reloaded)

### 📊 **Analysis Files**

- `taste_analysis_results.csv` - Taste profile insights and statistics
- `taste_similarities.csv` - Similarity scores for all unwatched dramas
- `enhanced_predictions.csv` - Predictions enhanced with taste similarity
- `taste_recommendations.csv` - Pure taste-based recommendations

### 📈 **Comparison Files**

- `loocv_predictions_watched.csv` - Cross-validation results for watched dramas

## Taste Profile Components

### 1. Categorical Preferences

- **Genres**: Which genres you prefer (weighted by ratings)
- **Tags**: Which themes/tags appeal to you
- **Cast**: Actors you consistently enjoy
- **Crew**: Directors, screenwriters, composers you prefer

### 2. Text Content Preferences

- **Synopsis Embeddings**: Semantic representation of storylines you like
- **Review Embeddings**: Types of audience reactions you align with

### 3. Semantic Patterns

- **Synopsis Similarity**: How similar your preferred content is to other content
- **Review Similarity**: How similar audience reactions are to your favorites

### 4. Rating Patterns

- **Average Rating**: Your typical rating level
- **Rating Consistency**: How consistent your ratings are
- **Rating Distribution**: Spread of your ratings across the scale

## Similarity Calculation

The system calculates similarity across multiple dimensions:

### Overall Similarity Formula

```
Overall = 0.3 × Categorical + 0.25 × Text + 0.45 × Semantic
```

### Component Similarities

- **Categorical**: Average of genre, tag, cast, crew similarities
- **Text**: Average of synopsis and review embedding similarities
- **Semantic**: Average of semantic pattern similarities

## Configuration Options

### Command Line Arguments

- `--enable-taste-profile`: Enable taste profile analysis (default: True)
- `--taste-weight`: Weight for taste similarity in enhanced predictions (default: 0.3)
- `--use-semantic-similarity`: Enable semantic similarity features (default: True)

### Feature Configuration

The taste profile system respects your existing feature configuration:

- `use_genres`: Include genre preferences
- `use_tags`: Include tag preferences  
- `use_cast`: Include actor preferences
- `use_crew`: Include crew preferences
- `use_sentiment`: Include sentiment preferences (not used in taste profile)
- `use_semantic_similarity`: Include semantic pattern preferences

## Advanced Features

### 1. Hidden Gems Discovery

The system identifies dramas that:

- Have high taste similarity but low model predictions
- May be overlooked but match your preferences perfectly

### 2. Overrated Detection

The system flags dramas that:

- Have low taste similarity but high model predictions
- May not align with your personal preferences

### 3. Confidence Boosting

Taste similarity can boost prediction confidence:

- Higher taste alignment = higher confidence
- Helps identify more reliable recommendations

### 4. Profile Persistence

Taste profiles can be saved and reloaded:

```python
# Save profile
taste_analyzer.save_taste_analysis()

# Load profile (in future runs)
taste_analyzer.taste_profile.load_profile('user_taste_profile.json')
```

## Performance Considerations

### Caching

- Semantic embeddings are cached for efficiency
- Taste profiles can be saved and reused
- Similarity calculations are optimized

### Scalability

- Processes 1000+ dramas efficiently
- Memory usage scales linearly with drama count
- Can handle large user libraries

### Accuracy

- Weighted by user ratings (higher ratings = more influence)
- Uses multiple similarity metrics for robustness
- Combines explicit and implicit preferences

## Troubleshooting

### Common Issues

1. **No taste profile built**
   - Ensure you have watched dramas with ratings
   - Check that semantic similarity is enabled

2. **Low similarity scores**
   - May indicate diverse taste preferences
   - Consider adjusting similarity thresholds

3. **Memory issues**
   - Reduce number of dramas processed
   - Use caching to avoid recomputation

### Debug Mode

Enable verbose output to see detailed processing:

```python
# The system provides detailed progress updates
print("Building user taste profile from watched dramas...")
print("Calculating taste similarities for X unwatched dramas...")
```

## Future Enhancements

### Planned Features

- **Temporal Analysis**: How tastes evolve over time
- **Collaborative Filtering**: Compare with similar users
- **Content-Based Filtering**: More sophisticated content analysis
- **Hybrid Recommendations**: Advanced blending of multiple approaches

### Extensibility

The system is designed to be easily extensible:

- Add new similarity metrics
- Include additional preference dimensions
- Customize weighting schemes
- Integrate with external recommendation systems

## Conclusion

The User Taste Profile System provides a comprehensive, interpretable, and effective approach to personalized drama recommendations. It leverages your existing infrastructure while adding sophisticated taste analysis capabilities that enhance both the accuracy and explainability of recommendations.
 