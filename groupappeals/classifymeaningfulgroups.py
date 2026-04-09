"""Meaningful groups classification functionality."""

import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from .device_utilities import determine_compute_device, convert_device_to_pipeline_id
from .pre_and_post_processing import split_meaningful_groups_into_columns


def classify_groups(texts, model_repo="rwillh11/mdeberta_groups_2.0", score_threshold=0.5, batch_size=32, device=None):
    """
    Classify group references into meaningful categories.
    
    Args:
        texts (list): List of group reference texts to classify
        model_repo (str): The Hugging Face model repository to use
        score_threshold (float): Threshold for accepting a label prediction (0.0-1.0)
        batch_size (int): Number of examples to process at once
        device (str, optional): Device to use for computation ('cuda', 'mps', or 'cpu')
        
    Returns:
        list: List of lists, where each inner list contains the predicted category labels for a group
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If model loading or processing fails
    """
    # Check inputs
    if not texts:
        raise ValueError("No texts provided for group classification")
    
    # Validate threshold
    if not (0.0 <= score_threshold <= 1.0):
        raise ValueError(f"Score threshold must be between 0.0 and 1.0, got {score_threshold}")
        
    # Determine device if not specified
    if device is None:
        device = determine_compute_device()
    
    print(f"Using device: {device}")
    
    
    # Filter out NaN or None values
    valid_texts = []
    valid_indices = []
    
    for i, text in enumerate(texts):
        if pd.notna(text) and text is not None and text != "":
            valid_texts.append(text)
            valid_indices.append(i)
    
    if not valid_texts:
        print("Warning: No valid texts to classify after filtering NaN/None values")
        return [[] for _ in range(len(texts))]
    
    try:
        # Load the multi-label text classification pipeline
        classifier = pipeline(
            "text-classification",
            model=model_repo,
            top_k=None,
            device=convert_device_to_pipeline_id(device)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}") from e
        
    try:
        # Process in batches
        all_valid_predictions = []
        
        # Iterate through texts in batches
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i+batch_size]
            
            # Process current batch
            batch_predictions = classifier(batch_texts)
            
            # Extract predicted labels based on threshold
            batch_results = []
            for prediction in batch_predictions:
                predicted_labels = [label_score['label'] for label_score in prediction 
                                  if label_score['score'] >= score_threshold]
                batch_results.append(predicted_labels)
            
            all_valid_predictions.extend(batch_results)
        
        # Create full results array with empty lists for invalid entries
        all_predictions = [[] for _ in range(len(texts))]
        
        # Fill in the valid predictions
        for i, pred_idx in enumerate(valid_indices):
            if i < len(all_valid_predictions):
                all_predictions[pred_idx] = all_valid_predictions[i]
        
        return all_predictions
        
    except Exception as e:
        raise RuntimeError(f"Error during group classification: {str(e)}") from e

def process_groups_csv(input_file, group_column="Exact.Group.Text", output_file=None, 
                     model_repo="rwillh11/mdeberta_groups_2.0", score_threshold=0.5, device=None, split_groups=False):
    """
    Process a CSV file to classify group references into meaningful categories.
    
    Args:
        input_file (str): Path to the input CSV file
        group_column (str): Name of the column containing group references
        output_file (str, optional): Path where the results should be saved
        model_repo (str): Model to use for classification
        score_threshold (float): Threshold for accepting a label prediction
        device (str, optional): Device to use for computation ('cuda', 'mps', or 'cpu')
        split_groups (bool): Whether to split meaningful groups into separate Group1, Group2, etc. columns
        
    Returns:
        pd.DataFrame: DataFrame containing the original data plus a "Meaningful Group" column
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input file is empty
        KeyError: If required columns don't exist
        RuntimeError: For other processing errors
    """
    try:
        # Load the data
        try:
            df = pd.read_csv(input_file, sep=None, engine='python')
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {input_file}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Input file is empty: {input_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to read input file: {str(e)}") from e
            
        # Check if required column exists
        if group_column not in df.columns:
            raise KeyError(f"Group column '{group_column}' not found in the input file")
        
        # Extract group texts
        texts = df[group_column].tolist()
        
        # Classify groups - use keyword arguments to avoid parameter order issues
        predictions = classify_groups(texts, model_repo=model_repo, score_threshold=score_threshold, device=device)
        
        # Add predictions to DataFrame
        df["Meaningful Group"] = predictions
        
        # Apply optional postprocessing steps
        if split_groups:
            print("Splitting meaningful groups into separate columns...")
            df = split_meaningful_groups_into_columns(df, 'Meaningful Group')
        
        # Save to file if specified
        if output_file:
            try:
                df.to_csv(output_file, index=False)
                print(f"Results saved to: {output_file}")
            except Exception as e:
                print(f"Warning: Failed to save results to {output_file}: {str(e)}")
                print("Continuing with in-memory results...")
        
        return df
        
    except Exception as e:
        error_msg = f"Error processing CSV file for group classification: {str(e)}"
        print(error_msg)
        raise
