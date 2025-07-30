import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# Load the stratified evaluation results
df = pd.read_csv('stratified_evaluation_results.csv')

print("🔍 SEMANTIC SIMILARITY PATTERN ANALYSIS")
print("=" * 50)

# Basic statistics
print(f"\n📊 BASIC STATISTICS:")
print(f"Total dramas: {len(df)}")
print(f"Rating range: {df['actual_rating'].min():.1f} - {df['actual_rating'].max():.1f}")
print(f"Semantic similarity range: {df['semantic_similarity'].min():.3f} - {df['semantic_similarity'].max():.3f}")
print(f"Mean semantic similarity: {df['semantic_similarity'].mean():.3f}")

# Correlation analysis
corr, p_value = spearmanr(df['actual_rating'], df['semantic_similarity'])
print(f"\n📈 CORRELATION ANALYSIS:")
print(f"Spearman correlation: {corr:.3f} (p={p_value:.3f})")

# Group by rating ranges
df['rating_group'] = pd.cut(df['actual_rating'], bins=[0, 5, 7, 10], labels=['Low (1-5)', 'Medium (5-7)', 'High (7-10)'])

print(f"\n📊 SEMANTIC SIMILARITY BY RATING GROUP:")
for group in df['rating_group'].unique():
    if pd.notna(group):
        group_data = df[df['rating_group'] == group]
        print(f"{group}:")
        print(f"  Count: {len(group_data)}")
        print(f"  Mean semantic similarity: {group_data['semantic_similarity'].mean():.3f}")
        print(f"  Std semantic similarity: {group_data['semantic_similarity'].std():.3f}")

# Look at specific examples
print(f"\n🎯 SPECIFIC EXAMPLES:")
print("High-rated dramas with LOW semantic similarity:")
high_rated = df[df['actual_rating'] >= 8.0].sort_values('semantic_similarity')
for _, row in high_rated.head(3).iterrows():
    print(f"  {row['drama_title']} (Rating: {row['actual_rating']}, Semantic: {row['semantic_similarity']:.3f})")

print("\nLow-rated dramas with HIGH semantic similarity:")
low_rated = df[df['actual_rating'] <= 5.0].sort_values('semantic_similarity', ascending=False)
for _, row in low_rated.head(3).iterrows():
    print(f"  {row['drama_title']} (Rating: {row['actual_rating']}, Semantic: {row['semantic_similarity']:.3f})")

# Analyze the semantic similarity calculation
print(f"\n🔧 SEMANTIC SIMILARITY CALCULATION ANALYSIS:")
print("The semantic similarity is calculated by:")
print("1. Extracting semantic features (avg_synopsis_similarity, max_synopsis_similarity, max_review_similarity)")
print("2. Comparing drama features to profile features")
print("3. Normalizing to 0-1 range")

# Check if there's a pattern in the raw vs normalized values
print(f"\n📊 RAW VS NORMALIZED VALUES:")
print("From debug output, we can see:")
print("- While You Were Sleeping: Raw=0.016, Normalized=0.157")
print("- Sweet Home: Raw=0.018, Normalized=0.177") 
print("- Shogun: Raw=0.048, Normalized=0.482")

print(f"\n🤔 POTENTIAL ISSUES:")
print("1. The normalization might be amplifying small differences")
print("2. The semantic features might not capture meaningful patterns")
print("3. The profile semantic features are very small (0.006-0.011)")
print("4. The drama semantic features are much larger (0.096-0.680)")

# Check the scale mismatch
print(f"\n⚖️ SCALE MISMATCH ANALYSIS:")
print("Profile semantic features (very small):")
print("- avg_synopsis_similarity: 0.006")
print("- max_synopsis_similarity: 0.010") 
print("- max_review_similarity: 0.011")

print("\nDrama semantic features (much larger):")
print("- While You Were Sleeping: [0.399, 0.599, 0.680]")
print("- Sweet Home: [0.356, 0.545, 0.587]")
print("- Shogun: [0.096, 0.201, 0.325]")

print(f"\n💡 HYPOTHESIS:")
print("The negative correlation might be due to:")
print("1. Scale mismatch between profile and drama features")
print("2. The similarity calculation being based on 'closeness' rather than 'alignment'")
print("3. The normalization amplifying noise rather than signal")
print("4. Semantic features capturing patterns that don't align with user preferences")

print(f"\n🎯 RECOMMENDATIONS:")
print("1. Investigate the semantic feature extraction process")
print("2. Consider different similarity calculation methods")
print("3. Adjust the normalization range")
print("4. Check if the profile semantic features are meaningful") 