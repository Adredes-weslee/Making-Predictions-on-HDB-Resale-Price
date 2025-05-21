"""
Train and save a full pipeline model for HDB resale price prediction.

This script loads the processed training data, trains a model with preprocessing pipeline,
and saves the entire pipeline for consistent prediction. The pipeline includes both
the preprocessing steps and the trained model, ensuring that the same transformations
are applied consistently during both training and prediction.

Usage:
    python scripts/train_pipeline_model.py --model-type linear --output-name my_model

Options:
    --model-type: Type of model to train (linear, ridge, lasso)
    --output-name: Base name for the saved model files
    --data-path: Path to the training data CSV file
"""
import os
import argparse
from pathlib import Path


# Add the project root to the Python path
import sys
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.models.training import train_and_save_pipeline_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train a pipeline model for HDB resale price prediction")
    parser.add_argument("--model-type", type=str, default="linear", choices=["linear", "ridge", "lasso"],
                        help="Type of regression model to train")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Base name for the saved model files")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to the training data CSV file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set default output name if not provided
    output_name = args.output_name
    if output_name is None:
        output_name = f"pipeline_{args.model_type}_model"
    
    print(f"Training {args.model_type} model with pipeline...")
    
    # Train and save the model
    model_info = train_and_save_pipeline_model(
        model_type=args.model_type,
        data_path=args.data_path,
        model_name=output_name
    )
    
    print("\nModel training completed!")
    print(f"Model type: {args.model_type}")
    print(f"Training R²: {model_info['metrics']['train_r2']:.4f}")
    print(f"Test R²: {model_info['metrics']['test_r2']:.4f}")
    print(f"Training RMSE: ${model_info['metrics']['train_rmse']:.2f}")
    print(f"Test RMSE: ${model_info['metrics']['test_rmse']:.2f}")
    print(f"Model saved to: {model_info['pipeline_path']}")
    print(f"Feature info saved to: {model_info['feature_info_path']}")
    print(f"Total features used: {model_info['feature_count']}")
    print(f"Training samples: {model_info['training_samples']}")

if __name__ == "__main__":
    main()
