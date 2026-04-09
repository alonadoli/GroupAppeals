"""Group entity extraction functionality."""

from transformers import pipeline
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import torch
from .device_utilities import determine_compute_device, convert_device_to_pipeline_id

# =============================================================================
# SHARED HELPER FUNCTIONS
# =============================================================================

def remove_trailing_punctuation(text):
    """
    Remove trailing punctuation from text.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with trailing punctuation removed
    """
    if pd.isna(text):
        return text
    return re.sub(r'[^\w\s](\s+)?$', '', text)

def group_token_classification_results(results):
    """
    Process raw results from the token classifier.
    
    Args:
        results (list): Raw token classification results
        
    Returns:
        list: Processed entity data with words grouped by entity
    """
    entities = []
    current_entity = {"words": [], "score": 0, "start": None, "end": None}

    for result in results:
        if result['entity'] == 'LABEL_1':
            word = result['word'].lstrip('▁')
            if not current_entity["words"]:
                current_entity = {"words": [word], "score": result['score'], "start": result['start'], "end": result['end']}
            elif result['start'] <= current_entity["end"] + 1:
                current_entity["words"].append(word)
                current_entity["score"] += result['score']
                current_entity["end"] = result['end']
            else:
                current_entity["score"] /= len(current_entity["words"])
                entities.append(current_entity)
                current_entity = {"words": [word], "score": result['score'], "start": result['start'], "end": result['end']}
        else:
            continue

    if current_entity["words"]:
        current_entity["score"] /= len(current_entity["words"])
        entities.append(current_entity)

    return entities

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def extract_entity_text(row):
    """
    Extract entity text from the original text.
    
    Args:
        row (pd.Series): DataFrame row containing text and position data
        
    Returns:
        str or np.nan: Extracted text or NaN if positions are missing
    """
    if pd.isna(row['Start']) or pd.isna(row['End']):
        return np.nan
    else:
        start_pos = int(row['Start'])
        end_pos = int(row['End'])
        return row['text'][start_pos:end_pos + 1]

def create_entity_dataframe(structured_output):
    """
    Convert structured output to a DataFrame.
    
    ID numbering scheme:
    - 0 entities found: original_id.0
    - 1 entity found: original_id.1  
    - 2 entities found: original_id.1, original_id.2
    - 3 entities found: original_id.1, original_id.2, original_id.3
    - etc.
    
    Args:
        structured_output (list): Processed entity data
        
    Returns:
        pd.DataFrame: DataFrame containing extracted entities
    """
    rows = []
    
    for item in structured_output:
        if item["entities"]:
            # Create unique identifiers: .1, .2, .3, etc. for each entity found
            for idx, entity in enumerate(item["entities"], 1):
                rows.append({
                    "text_id": f"{item['id']}.{idx}",
                    "text": item["text"],
                    "Entity": ' '.join(entity['words']).strip(),
                    "Average Score": entity['score'],
                    "Start": entity['start'],
                    "End": entity['end']
                })
        else:
            # Use .0 suffix when no entities found for consistent numbering scheme
            rows.append({
                "text_id": f"{item['id']}.0",
                "text": item["text"],
                "Entity": pd.NA,
                "Average Score": pd.NA,
                "Start": pd.NA,
                "End": pd.NA
            })
    
    # Create DataFrame
    df_entities = pd.DataFrame(rows)
    
    # Add Exact Group Text column
    df_entities['Exact.Group.Text'] = df_entities.apply(extract_entity_text, axis=1)
    df_entities['Exact.Group.Text'] = df_entities['Exact.Group.Text'].apply(remove_trailing_punctuation)
    df_entities['Exact.Group.Text'] = df_entities['Exact.Group.Text'].apply(
        lambda x: np.nan if pd.isna(x) or len(str(x)) < 3 else x
    )
    
    return df_entities

def extract_entities(texts, ids=None, model_name="rwillh11/mdberta-token-bilingual-noContext_Enhanced", device=None):
    """
    Extract entities from a list of texts.
    
    Args:
        texts (list): List of texts to process
        ids (list, optional): List of IDs for each text
        model_name (str): Model name to use for extraction
        device (str, optional): Device to use for computation ('cuda', 'mps', or 'cpu')
        
    Returns:
        pd.DataFrame: DataFrame with extracted entities with the following columns:
          - text_id: The identifier for each text
          - text: The original text
          - Entity: The extracted group reference (joined tokens)
          - Average Score: Confidence score for the entity (average across tokens)
          - Start: Character position where the entity begins in the original text
          - End: Character position where the entity ends in the original text
          - Exact.Group.Text: The exact text extracted from the original text using start/end positions
    
    Raises:
        ValueError: If no texts are provided or if IDs don't match texts
        RuntimeError: If model loading or processing fails
    """
    try:
        # Check inputs
        if not texts:
            raise ValueError("No texts provided for entity extraction")
            
        # Determine device if not specified
        if device is None:
            device = determine_compute_device()
        
        print(f"Using device: {device}")
        
        # Set up device mapping for pipeline
        device_id = convert_device_to_pipeline_id(device)
            
        # Initialize the model
        try:
            token_classifier = pipeline("token-classification", model=model_name, device=device_id)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}") from e
        
        # Generate IDs if not provided
        if ids is None:
            ids = list(range(len(texts)))
        
        if len(ids) != len(texts):
            raise ValueError(f"Number of IDs ({len(ids)}) does not match number of texts ({len(texts)})")
            
        # Process each text
        structured_output = []
        for sentence, text_id in tqdm(zip(texts, ids), total=len(texts), desc="Classifying"):
            # Skip empty or null sentences
            if pd.isna(sentence) or sentence == "" or sentence is None:
                sentence_info = {"text": sentence, "id": text_id, "entities": []}
                structured_output.append(sentence_info)
                continue
                
            sentence_info = {"text": sentence, "id": text_id, "entities": []}
            try:
                results = token_classifier(sentence)
                sentence_info["entities"] = group_token_classification_results(results)
            except Exception as e:
                print(f"Warning: Failed to process text with ID {text_id}: {str(e)}")
                # Continue with empty entities rather than failing the entire batch
            
            structured_output.append(sentence_info)
        
        # Convert to DataFrame
        return create_entity_dataframe(structured_output)

    except ValueError as e:
        # Re-raise ValueError directly (for input validation errors)
        raise e
    except Exception as e:
        error_msg = f"Error during entity extraction: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

def process_csv(input_file, text_column="text", id_column="text_id", output_file=None, device=None):
    """
    Process a CSV file to extract entities.
    
    IMPORTANT: This function inherits IDs from your input data. Use meaningful composite IDs 
    (e.g., "party_date_sentence") rather than simple numeric IDs to maintain traceability 
    through the analysis pipeline.
    
    Args:
        input_file (str): Path to input CSV file
        text_column (str): Column containing text to analyze
        id_column (str): Column containing IDs (should be meaningful identifiers)
        output_file (str, optional): Path to save output CSV
        
    Returns:
        pd.DataFrame: DataFrame with extracted entities (preserves original IDs)
        
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
        if id_column not in df.columns:
            raise KeyError(f"ID column '{id_column}' not found in the input file")
            
        # Extract entities
        result_df = extract_entities(df[text_column].tolist(), df[id_column].tolist(), device=device)
        
        # Save to file if specified
        if output_file:
            try:
                result_df.to_csv(output_file, index=False)
                print(f"Results saved to: {output_file}")
            except Exception as e:
                print(f"Warning: Failed to save results to {output_file}: {str(e)}")
                print("Continuing with in-memory results...")
        
        return result_df
        
    except Exception as e:
        error_msg = f"Error processing CSV file: {str(e)}"
        print(error_msg)
        raise
