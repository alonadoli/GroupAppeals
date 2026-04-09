"""Policy detection functionality."""

import torch
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
from .device_utilities import determine_compute_device, convert_device_to_pipeline_id
from .pre_and_post_processing import extract_clean_policy_label, extract_group_from_hypothesis

# =============================================================================
# SHARED HELPER FUNCTIONS
# =============================================================================

def generate_policy_nli_hypotheses(group):
    """
    Create policy detection hypotheses.
    
    Args:
        group (str): The group reference
        
    Returns:
        list: List of hypothesis strings
    """
    return [
        f"The text contains a policy directed towards {group}.",
        f"The text does not contain a policy directed towards {group}."
    ]

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def detect_policy(texts, groups, model_name="rwillh11/mdeberta_NLI_policy_noContext", batch_size=32, device=None):
    """
    Detect policies towards groups.
    
    Args:
        texts (list): List of text strings to analyze
        groups (list): List of group references corresponding to each text
        model_name (str): The Hugging Face model to use for policy detection
        batch_size (int): Number of examples to process at once
        device (str, optional): Device to use for computation ('cuda', 'mps', or 'cpu')
        
    Returns:
        list: List of dictionaries with policy labels and confidence scores
              Format: [{'policy': 'policy'|'no policy'|'unknown', 'confidence': float}, ...]
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If model loading or processing fails
    """
    try:
        # Check inputs
        if not texts:
            raise ValueError("No texts provided for policy detection")
            
        if len(texts) != len(groups):
            raise ValueError(f"Number of texts ({len(texts)}) does not match number of groups ({len(groups)})")

        # Check if group reference column has missing values
        missing_groups = sum(1 for group in groups if pd.isna(group))
        if missing_groups > 0:
            raise ValueError(
                f"Found {missing_groups} missing values in group references. "
                f"Please filter out missing group references before policy detection."
            )
            
        # Determine device if not specified
        if device is None:            
            device = determine_compute_device()
     
        print(f"Using device: {device}")
        
        # Set up device mapping for pipeline
        device_id = convert_device_to_pipeline_id(device)
        
        # Initialize the classifier
        try:
            classifier = pipeline("zero-shot-classification", model=model_name, device=device_id)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}") from e
        
        # Process in batches
        results = []
        
        # Process data in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_groups = groups[i:i+batch_size]
            
            batch_results = []
            for text, group in tqdm(zip(batch_texts, batch_groups), 
                                   total=len(batch_texts), 
                                   desc=f"Detecting policy (batch {i//batch_size + 1})"):    
                try:
                    # Create hypotheses
                    hypotheses = generate_policy_nli_hypotheses(group)
                    
                    # Get classification result
                    result = classifier(text, hypotheses, multi_label=False)
                    
                    # Add top prediction and confidence to results
                    batch_results.append({
                        'policy': result['labels'][0],
                        'confidence': result['scores'][0]
                    })
                except Exception as e:
                    print(f"Warning: Failed to process policy for text '{text[:50]}...' and group '{group}': {str(e)}")
                    batch_results.append({
                        'policy': "unknown",
                        'confidence': 0.0
                    })
            
            results.extend(batch_results)
        
        return results
    
    except ValueError as e:
        # Re-raise ValueError directly (for input validation errors)
        raise e
        
    except Exception as e:
        error_msg = f"Error during policy detection: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

def process_policy_csv(input_file, text_column="text", group_column="Exact.Group.Text", 
                      output_file=None, model_name="rwillh11/mdeberta_NLI_policy_noContext", batch_size=32,
                      device=None, clean_labels=False, quality_control=False):
    """
    Process a CSV file to detect policies towards groups.
    
    Args:
        input_file (str): Path to the input CSV file
        text_column (str): Name of the column containing the text
        group_column (str): Name of the column containing group references
        output_file (str, optional): Path where the results should be saved
        model_name (str): The model to use for policy detection
        batch_size (int): Number of examples to process at once (default: 32)
        device (str, optional): Device to use for computation ('cuda', 'mps', or 'cpu')
        clean_labels (bool): Whether to extract clean policy labels ('policy', 'no policy')
        quality_control (bool): Whether to run quality control checks on group extraction
        
    Returns:
        pd.DataFrame: DataFrame containing the original data plus a "Policy" column
        
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
            
        # Check if required columns exist
        if text_column not in df.columns:
            raise KeyError(f"Text column '{text_column}' not found in the input file")
        if group_column not in df.columns:
            raise KeyError(f"Group column '{group_column}' not found in the input file")
        
        # Extract text and groups
        texts = df[text_column].tolist()
        groups = df[group_column].tolist()
        
        # Detect policy
        policies = detect_policy(texts, groups, model_name, batch_size, device)
        
        # Add policy to DataFrame
        df["Policy"] = [r['policy'] for r in policies]
        df["Policy_Confidence"] = [r['confidence'] for r in policies]
        
        # Apply optional postprocessing steps
        if clean_labels:
            print("Extracting clean policy labels...")
            df['Policy_Clean'] = df['Policy'].apply(extract_clean_policy_label)
            
        if quality_control:
            print("Running quality control checks on group extraction...")
            df['Group_from_policy'] = df['Policy'].apply(lambda x: extract_group_from_hypothesis(x, 'policy'))
            
            # Check for mismatches
            mismatched_policy = df[df[group_column] != df['Group_from_policy']]
            if len(mismatched_policy) > 0:
                print(f"⚠️ Warning: Found {len(mismatched_policy)} rows where extracted group from policy doesn't match {group_column}")
                print("Sample mismatches:")
                print(mismatched_policy[[group_column, 'Group_from_policy']].head())
            else:
                print("✅ Policy quality control passed: all groups match")
            
            # Remove temporary quality control column
            df = df.drop('Group_from_policy', axis=1)
        
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
        error_msg = f"Error processing CSV file for policy detection: {str(e)}"
        print(error_msg)
        raise