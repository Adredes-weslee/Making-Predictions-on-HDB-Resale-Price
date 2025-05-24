"""
Analyze the cardinality (unique values) of each column in the dataset.
This helps understand which columns contribute most to the feature explosion
when one-hot encoding is applied.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Get project root directory (go up one level from scripts folder)
root_dir = Path(__file__).parent.parent
data_path = os.path.join(root_dir, 'data', 'raw', 'train.csv')

# Load the data
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path, low_memory=False)  # Added low_memory=False to avoid mixed type warning

# Print basic dataset info
print(f"\nDataset shape: {df.shape} (rows, columns)")

# Calculate and print cardinality for each column
print("\n=== COLUMN CARDINALITY ANALYSIS ===")
print(f"{'Column Name':<30} {'Dtype':<12} {'Unique Values':<15} {'Top Values (examples)'}")
print("-" * 80)

# Sort columns by cardinality (descending)
cardinalities = [(col, df[col].nunique(), df[col].dtype) for col in df.columns]
cardinalities.sort(key=lambda x: x[1], reverse=True)

# Print details with examples
for col, unique_count, dtype in cardinalities:
    try:
        # For high-cardinality columns, show just a few examples
        if unique_count > 5:
            examples = str(df[col].value_counts().nlargest(3).index.tolist())
            if len(examples) > 30:
                examples = examples[:30] + "..."
        else:
            examples = str(df[col].unique().tolist())
            if len(examples) > 30:
                examples = examples[:30] + "..."
        
        # Print the formatted line
        print(f"{col:<30} {str(dtype):<12} {unique_count:<15} {examples}")
    except Exception as e:
        print(f"{col:<30} {str(dtype):<12} ERROR: {str(e)}")

print("\n=== HIGH CARDINALITY COLUMNS ===")
print("These columns would create many features when one-hot encoded:")
high_card_cols = [col for col, count, _ in cardinalities if count > 100 and count < len(df)/2]
for col in high_card_cols:
    nunique = df[col].nunique()
    print(f"- {col}: {nunique} unique values")

# Calculate potential one-hot encoded features
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
potential_features = sum(df[col].nunique() for col in categorical_cols)
print(f"\nEstimated features after one-hot encoding categorical columns: {potential_features}")

print("\nAnalysis complete!")

# Add this at the end of your script
print("\n=== RECOMMENDED COLUMNS TO DROP ===")
columns_to_drop = [
    # Unique identifiers with no predictive value
    "id",
    
    # Ultra-high cardinality text fields (causes feature explosion)
    "address",
    "postal",
    "block",
    # "street_name",
    
    # Named entities with too many unique values
    "bus_stop_name",
    # "pri_sch_name",
    "sec_sch_name",
    # "mrt_name",
    
    # Redundant coordinates (distance metrics are better)
    "Latitude",
    "Longitude",
    "bus_stop_latitude", 
    "bus_stop_longitude",
    "pri_sch_latitude",
    "pri_sch_longitude",
    "sec_sch_latitude",
    "sec_sch_longitude",
    "mrt_latitude",
    "mrt_longitude",
    
    # Redundant features (duplicates or derivatives)
    "floor_area_sqm",  # Duplicate of floor_area_sqft in different units
    "Tranc_YearMonth"   # Already broken down into Tranc_Year and Tranc_Month
]

print("The following columns should be dropped to prevent feature explosion:")
for col in columns_to_drop:
    nunique = df[col].nunique() if col in df.columns else "N/A"
    print(f"- {col}: {nunique} unique values")

# Calculate reduced feature count
remaining_categorical_cols = [col for col in categorical_cols if col not in columns_to_drop]
reduced_features = sum(df[col].nunique() for col in remaining_categorical_cols)
print(f"\nEstimated features after one-hot encoding REDUCED categorical columns: {reduced_features}")
print(f"Feature reduction: {potential_features} → {reduced_features} ({reduced_features/potential_features:.1%} of original)")


# Add this at the end of your script:
import json

# Create a configuration output for other scripts to use
print("\n=== CREATING FEATURE SELECTION CONFIG ===")

# Columns to automatically drop based on high cardinality
columns_to_drop = [
    "id",  # Identifier with no predictive value
    "address",  # 9157 unique values
    "postal",  # 9125 unique values
    "block",  # 2514 unique values
    # "street_name",  # 553 unique values
    "bus_stop_name",  # 1657 unique values
    # "pri_sch_name",  # 177 unique values 
    "sec_sch_name",  # 134 unique values
    # "mrt_name",  # 94 unique values
    
    # Redundant coordinates (distance metrics are better)
    "Latitude", "Longitude",
    "bus_stop_latitude", "bus_stop_longitude",
    "pri_sch_latitude", "pri_sch_longitude",
    "sec_sch_latitude", "sec_sch_longitude",
    "mrt_latitude", "mrt_longitude",
    
    # Redundant features (duplicates or derivatives)
    "floor_area_sqft",  # Duplicate of floor_area_sqm in different units
    "Tranc_YearMonth",  # Already broken down into Tranc_Year and Tranc_Month
    "total_dwelling_units",
    "1room_sold",
    "2room_sold",
    "3room_sold",
    "4room_sold",
    "5room_sold",
    "exec_sold",
    "multigen_sold",
    "studio_apartment_sold",
    "1room_rental",
    "2room_rental",
    "3room_rental",
    "other_room_rental",
    "planning_area",
    "Mall_Within_500m",
    "Mall_Within_1km",
    "Mall_Within_2km",
    "Hawker_Within_500m",
    "Hawker_Within_1km",
    "Hawker_Within_2km",
    "street_name",
    "mid_storey",
    "lower",
    "upper",
    "mid",
    "full_flat_type",
    "residential",
    "commercial",
    "pri_sch_name",
    "year_completed",
    "hawker_food_stalls",
    "hawker_market_stalls",
    "vacancy",
    "Tranc_Year",
    "Tranc_Month",
    
]

# Check how many of these columns exist in the actual data
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
print(f"Will drop {len(existing_columns_to_drop)} out of {len(columns_to_drop)} recommended columns")
print(f"Columns to drop: {existing_columns_to_drop}")

# Create feature selection config
feature_config = {
    "columns_to_drop": existing_columns_to_drop,
    "categorical_features": [col for col in df.select_dtypes(include=['object']).columns 
                            if col not in existing_columns_to_drop],
    "numerical_features": [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
                          if col not in existing_columns_to_drop and col != 'resale_price'],
    "timestamp": pd.Timestamp.now().isoformat(),
    "cardinality_stats": {col: int(df[col].nunique()) for col in df.columns}
}

# Write config to file
config_dir = os.path.join(root_dir, "configs")
os.makedirs(config_dir, exist_ok=True)
config_path = os.path.join(config_dir, "feature_selection_config.json")

with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(feature_config, f, indent=2)

print(f"Feature selection configuration saved to: {config_path}")
print("Use this configuration in model_config.yaml or directly in your scripts")