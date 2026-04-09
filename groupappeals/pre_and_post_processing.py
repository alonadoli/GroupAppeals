"""
Pre-processing and post-processing functions for GroupAppeals analysis.

This module provides functions for:
- Pre-processing: Data preparation including composite ID creation
- Post-processing: Cleaning model outputs, splitting meaningful groups into columns, 
  merging results, and creating complete output files
"""

import pandas as pd
import numpy as np
import re
import ast
from typing import Optional, List, Dict, Any


# =============================================================================
# PRE-PROCESSING FUNCTIONS
# =============================================================================

def create_composite_id(df, party_col="party", date_col="date", sentence_col="sentence_id", separator="_"):
    """
    Create composite IDs from party, date, and sentence components.
    
    This function creates meaningful identifiers that can be tracked through the entire analysis pipeline.
    
    Note: Column names are customizable - use party_col="candidate" for candidate data, 
    party_col="organization" for organization data, etc.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing party, date, and sentence ID columns
        party_col (str): Column name for party (default: "party")
        date_col (str): Column name for date (default: "date")
        sentence_col (str): Column name for sentence ID (default: "sentence_id")
        separator (str): Separator character for composite ID (default: "_")
        
    Returns:
        pd.Series: Series of composite IDs in format "party_date_sentenceID"
        
    Example:
        >>> df = pd.DataFrame({
        ...     'party': ['PartyA', 'PartyB'], 
        ...     'date': [2020, 2020], 
        ...     'sentence_id': [1, 2]
        ... })
        >>> df['text_id'] = create_composite_id(df)
        >>> print(df['text_id'].tolist())
        ['PartyA_2020_1', 'PartyB_2020_2']
        
    Raises:
        KeyError: If any of the required columns don't exist in the DataFrame
        ValueError: If the DataFrame is empty
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
        
    required_cols = [party_col, date_col, sentence_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Required columns not found in DataFrame: {missing_cols}")
    
    return df[party_col].astype(str) + separator + df[date_col].astype(str) + separator + df[sentence_col].astype(str)



# =============================================================================
# POST-PROCESSING FUNCTIONS
# =============================================================================

def extract_clean_stance_label(stance_text):
    """
    Extract clean stance label from verbose model output.
    
    Args:
        stance_text (str): Verbose stance output like "The text is positive towards workers."
        
    Returns:
        str: Clean stance label ('positive', 'negative', 'neutral') or None
    """
    if pd.isna(stance_text) or not isinstance(stance_text, str):
        return None
        
    stance_text_lower = stance_text.lower()
    
    if re.search(r'positive', stance_text_lower):
        return 'positive'
    elif re.search(r'negative', stance_text_lower):
        return 'negative'
    elif re.search(r'neutral', stance_text_lower):
        return 'neutral'
    else:
        return None


def extract_clean_policy_label(policy_text):
    """
    Extract clean policy label from verbose model output.
    
    Args:
        policy_text (str): Verbose policy output like "The text contains a policy directed towards workers."
        
    Returns:
        str: Clean policy label ('policy', 'no policy') or None
    """
    if pd.isna(policy_text) or not isinstance(policy_text, str):
        return None
        
    policy_text_lower = policy_text.lower()
    
    if re.search(r'contains a policy', policy_text_lower):
        return 'policy'
    elif re.search(r'does not contain a policy', policy_text_lower):
        return 'no policy'
    else:
        return None


def extract_group_from_hypothesis(hypothesis_text, hypothesis_type='stance'):
    """
    Extract group name from model hypothesis output for quality control.
    
    Args:
        hypothesis_text (str): Full hypothesis like "The text is positive towards workers."
        hypothesis_type (str): Type of hypothesis ('stance' or 'policy')
        
    Returns:
        str: Extracted group name or None
    """
    if pd.isna(hypothesis_text) or not isinstance(hypothesis_text, str):
        return None
    
    if hypothesis_type == 'stance':
        # Extract from "towards [group]."
        match = re.search(r'towards\s+(.*?)\.$', hypothesis_text)
    elif hypothesis_type == 'policy':
        # Extract from "directed towards [group]."
        match = re.search(r'directed towards\s+(.*?)\.$', hypothesis_text)
    else:
        return None
        
    if match:
        return match.group(1).strip()
    else:
        return None


def parse_predicted_labels(label_string):
    """
    Parse predicted labels string into list of labels.
    
    Args:
        label_string: String representation of list like "['Families', 'Workers']"
                     Can also handle pandas Series (will process first element)
        
    Returns:
        list: List of labels, empty list if parsing fails
    """
    # Handle pandas Series by taking the first element
    if isinstance(label_string, pd.Series):
        if len(label_string) == 0:
            return []
        label_string = label_string.iloc[0]
    
    # Handle numpy arrays or other array-like objects (but not dicts/sets)
    if hasattr(label_string, '__len__') and hasattr(label_string, '__getitem__') and not isinstance(label_string, (str, list, dict)):
        if len(label_string) == 0:
            return []
        # Convert to string if it's a single element array
        if len(label_string) == 1:
            label_string = str(label_string[0])
        else:
            # If it's a multi-element array, this is likely an error
            raise ValueError(f"Expected single value or string, got array with {len(label_string)} elements")
    
    # Check for null/empty values - use pandas-safe methods
    try:
        if pd.isna(label_string):
            return []
    except (ValueError, TypeError):
        # Fallback for non-pandas-compatible types
        if label_string is None:
            return []
    
    if label_string == '' or label_string == '[]':
        return []
    
    # If it's already a list, return it
    if isinstance(label_string, list):
        return label_string
    
    if isinstance(label_string, str):
        # Handle string representations of lists
        label_string = label_string.strip()
        
        # Try to parse as literal list first
        if label_string.startswith('[') and label_string.endswith(']'):
            try:
                parsed = ast.literal_eval(label_string)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except (ValueError, SyntaxError):
                pass
        
        # Fallback: remove brackets and quotes, then split
        cleaned = re.sub(r'^[\[\]]+|[\[\]]+$', '', label_string)  # Remove only outer brackets
        cleaned = re.sub(r'^[\'\"]+|[\'\"]+$', '', cleaned.strip())  # Remove outer quotes
        if cleaned.strip() == '':
            return []
        labels = [label.strip().strip('\'"') for label in cleaned.split(',') if label.strip()]
        return labels
    
    # Handle other types by converting to string (with recursion protection)
    try:
        str_val = str(label_string)
        if str_val in ['nan', 'None', '', 'NaN']:
            return []
        # Only recurse if we got a different string representation
        if str_val != str(label_string) or not isinstance(label_string, str):
            return parse_predicted_labels(str_val)
        else:
            # Avoid infinite recursion - return empty list for unparseable strings
            return []
    except (ValueError, TypeError, RecursionError):
        return []


def determine_max_groups(meaningful_groups_column):
    """
    Automatically determine the maximum number of groups to split into columns.
    
    Args:
        meaningful_groups_column (pd.Series): Column containing meaningful group lists
        
    Returns:
        int: Maximum number of groups found in any single row
    """
    max_groups = 0
    
    for groups in meaningful_groups_column:
        parsed_groups = parse_predicted_labels(groups)
        if len(parsed_groups) > max_groups:
            max_groups = len(parsed_groups)
    
    return max_groups


def split_meaningful_groups_into_columns(df, meaningful_groups_column='Meaningful Group', max_groups=None):
    """
    Split meaningful groups into separate Group1, Group2, etc. columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing meaningful groups
        meaningful_groups_column (str): Name of column containing group lists
        max_groups (int, optional): Maximum number of columns to create. If None, auto-determined.
        
    Returns:
        pd.DataFrame: DataFrame with Group1, Group2, etc. columns added and original column removed
    """
    df = df.copy()
    
    # Auto-determine max groups if not specified
    if max_groups is None:
        max_groups = determine_max_groups(df[meaningful_groups_column])
        print(f"Auto-determined max groups: {max_groups}")
    
    # Define extraction function outside loop for efficiency
    def extract_group(x, group_index):
        parsed = parse_predicted_labels(x)
        return parsed[group_index] if group_index < len(parsed) else None
    
    # Create Group1, Group2, etc. columns
    for i in range(max_groups):
        col_name = f'Group{i+1}'
        df[col_name] = df[meaningful_groups_column].apply(lambda x: extract_group(x, i))
    
    # Keep original column - it may be needed for further processing
    # df = df.drop(meaningful_groups_column, axis=1)  # Commented out to preserve column
    
    print(f"Created {max_groups} group columns: {[f'Group{i+1}' for i in range(max_groups)]}")
    
    return df


