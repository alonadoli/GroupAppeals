"""
Full pipeline for GroupAppeals analysis.

This module provides a comprehensive pipeline that takes raw, unformatted text
and runs it through all stages of GroupAppeals analysis (token classification, stance detection,
policy detection, meaningful group classification).

The pipeline works directly with plain text input.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, List, Dict, Any

# Import all required functions
from .extractgrouptoken import process_csv
from .stancedetection import process_stance_csv
from .policydetection import process_policy_csv
from .classifymeaningfulgroups import process_groups_csv
from .pre_and_post_processing import (
    parse_predicted_labels,
    extract_clean_stance_label,
    extract_clean_policy_label,
    split_meaningful_groups_into_columns
)


def run_full_pipeline(
    input_file: str,
    output_file: Optional[str] = None,
    text_column: str = "text",
    id_column: str = "text_id",
    group_columns: Optional[List[str]] = None,
    order_columns: Optional[List[str]] = None,
    models: Optional[Dict[str, str]] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    create_composite_id: Optional[List[str]] = None,
    composite_separator: str = "_",
    clean_labels: bool = True,
    split_groups: bool = True,
    quality_control: bool = False,
    save_intermediate_outputs: bool = False,
    intermediate_output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Run the complete GroupAppeals analysis pipeline from raw text to final results.
    
    This function processes raw text through all 6 stages:
    1. Extract groups (token classification)
    2. Filter data (remove missing groups, prepare for analysis)
    3. Stance detection
    4. Policy detection  
    5. Group classification
    6. Combine all results by text_id
    
    Args:
        input_file (str): Path to input CSV file with raw text
        output_file (str, optional): Path to save final results CSV
        text_column (str): Column containing raw text (default: "text")
        id_column (str): Column containing identifiers (default: "text_id")
        group_columns (list, optional): Columns to group by for processing
        order_columns (list, optional): Columns to sort by within each group
        models (dict, optional): Custom model names for each step
        batch_size (int): Batch size for processing (default: 32)
        device (str, optional): Device to use ('cuda', 'mps', or 'cpu')
        create_composite_id (list, optional): Columns to combine into unit_id
        composite_separator (str): Separator for composite ID (default: "_")
        clean_labels (bool): Extract clean stance/policy labels ('positive', 'policy', etc.)
        split_groups (bool): Split meaningful groups into Group1, Group2, etc. columns
        quality_control (bool): Run quality control checks on group extraction
        save_intermediate_outputs (bool): Save output from each stage separately (default: False)
        intermediate_output_dir (str, optional): Directory for intermediate outputs (default: same as output_file)
        
    Returns:
        pd.DataFrame: Complete results with analysis from all pipeline stages
        
    Final Output Columns:
        - text_id: Original identifier
        - text: Input text
        - Exact.Group.Text: Extracted group reference
        - Average Score: Confidence from token classifier
        - Start: Token position start
        - End: Token position end
        - Stance: Stance classification result
        - Stance_Confidence: Confidence score for stance prediction
        - Policy: Policy classification result
        - Policy_Confidence: Confidence score for policy prediction
        - Meaningful Group: Group classification result
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required columns don't exist or data is invalid
        RuntimeError: For processing errors
        
    Example:
        # Basic usage
        results = run_full_pipeline("raw_data.csv", "complete_results.csv")
        
        # With grouping and ordering
        results = run_full_pipeline(
            input_file="manifestos.csv",
            output_file="analysis_results.csv",
            group_columns=["party", "election"],
            order_columns=["text_id"],
            models={
                "extraction": "custom/extraction-model",
                "stance": "custom/stance-model", 
                "policy": "custom/policy-model",
                "classification": "custom/classification-model"
            }
        )
    """
    
    # Set default models
    default_models = {
        "extraction": "rwillh11/mdberta-token-bilingual-noContext_Enhanced",
        "stance": "rwillh11/mdeberta_NLI_stance_NoContext",
        "policy": "rwillh11/mdeberta_NLI_policy_noContext", 
        "classification": "rwillh11/mdeberta_groups_2.0"
    }
    
    if models:
        default_models.update(models)
    models = default_models
    
    try:
        print("🚀 Starting Full GroupAppeals Analysis Pipeline")
        print("=" * 60)
        
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # Set default output file if not specified
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_complete_analysis{ext}"
        
        # Extract output directory for temporary and intermediate files
        output_dir = os.path.dirname(output_file)
        if not output_dir:
            output_dir = "."
        
        # Set up intermediate output directory to save intermediate files
        if save_intermediate_outputs:
            if intermediate_output_dir is None:
                # Use same directory as output file
                intermediate_output_dir = output_dir
            
            # Create directory if it doesn't exist
            os.makedirs(intermediate_output_dir, exist_ok=True)
            
            # Define intermediate output file names
            base_name = os.path.splitext(os.path.basename(output_file))[0]
            token_output_file = os.path.join(intermediate_output_dir, f"{base_name}_token_classification.csv")
            no_groups_file = os.path.join(intermediate_output_dir, f"{base_name}_no_detected_groups.csv")
            stance_output_file = os.path.join(intermediate_output_dir, f"{base_name}_stance_detection.csv")
            policy_output_file = os.path.join(intermediate_output_dir, f"{base_name}_policy_detection.csv")
            groups_output_file = os.path.join(intermediate_output_dir, f"{base_name}_meaningful_groups.csv")
        
        # =====================================================================
        # STEP 0: Process input data and create composite IDs if needed
        # =====================================================================
        if create_composite_id:
            print(f"\n🔧 Step 0: Creating composite IDs from columns: {create_composite_id}")
            
            # Load the input data
            try:
                input_df = pd.read_csv(input_file, sep=None, engine='python')
            except Exception as e:
                raise RuntimeError(f"Failed to read input file for composite ID creation: {str(e)}") from e
            
            # Check if composite ID columns exist
            missing_columns = [col for col in create_composite_id if col not in input_df.columns]
            if missing_columns:
                raise KeyError(f"Composite ID columns not found in input file: {missing_columns}")
            
            # Create composite unit_id (equivalent to R's paste(..., sep="_"))
            input_df['unit_id'] = input_df[create_composite_id].astype(str).agg(composite_separator.join, axis=1)
            
            # Create processed input file with text and unit_id columns
            processed_input = input_df[[text_column, 'unit_id']].copy()
            
            # Save processed input if intermediate outputs requested
            if save_intermediate_outputs:
                processed_input_file = os.path.join(intermediate_output_dir, f"{base_name}_processed_input.csv")
                processed_input.to_csv(processed_input_file, index=False)
                print(f"   💾 Processed input saved to: {processed_input_file}")
            
            # Update parameters for token classification
            processed_input_file = os.path.join(output_dir, f"temp_processed_input_{os.path.basename(output_file)}")
            processed_input.to_csv(processed_input_file, index=False)
            input_file = processed_input_file
            id_column = 'unit_id'
            
            print(f"✅ Step 0 Complete: Created {len(input_df)} composite IDs")
        
        # =====================================================================
        # STEP 1: Extract groups (token classification)
        # =====================================================================
        print("\n🔍 Step 1: Extracting group references with token classification...")
        
        step1_output = os.path.join(output_dir, f"temp_step1_{os.path.basename(output_file)}")
        
        extraction_df = process_csv(
            input_file=input_file,
            text_column=text_column,
            id_column=id_column,
            output_file=step1_output,
            device=device
        )
        
        print(f"✅ Step 1 Complete: Extracted groups from {len(extraction_df)} rows")
        
        # Save intermediate token classification output if requested
        if save_intermediate_outputs:
            print(f"   💾 Saving token classification output to: {token_output_file}")
            # Save complete output with all columns from input (preserves all data)
            extraction_df.to_csv(token_output_file, index=False)
            print(f"   ✅ Token classification output saved")
        
        # =====================================================================
        # STEP 2: Filter data for stance/policy detection (remove missing groups)
        # =====================================================================
        print("\n🔄 Step 2: Filtering data for stance and policy detection...")
        
        step2_output = os.path.join(output_dir, f"temp_step2_{os.path.basename(output_file)}")
        
        # Load and filter data - separate rows with/without detected groups
        filtered_df = pd.read_csv(step1_output)
        original_count = len(filtered_df)
        
        # Separate rows with and without group references
        no_detected_groups = filtered_df[filtered_df['Exact.Group.Text'].isna()].copy()
        filtered_df = filtered_df.dropna(subset=['Exact.Group.Text']).copy()
        
        # The text column is already available from token classification
        
        # Save the filtered DataFrame
        filtered_df.to_csv(step2_output, index=False)
        
        print(f"✅ Step 2 Complete: Filtered {len(filtered_df)} rows for stance/policy detection")
        print(f"   (Separated {len(no_detected_groups)} rows with missing group references)")
        
        # Save intermediate no_detected_groups output if requested
        if save_intermediate_outputs:
            print(f"   💾 Saving no detected groups output to: {no_groups_file}")
            no_detected_groups.to_csv(no_groups_file, index=False)
            print(f"   ✅ No detected groups output saved")
        
        # =====================================================================
        # STEP 3: Stance detection
        # =====================================================================
        print("\n😊 Step 3: Detecting stance towards groups...")
        
        step3_output = os.path.join(output_dir, f"temp_step3_{os.path.basename(output_file)}")
        
        stance_df = process_stance_csv(
            input_file=step2_output,
            text_column='text',
            group_column='Exact.Group.Text',
            output_file=step3_output,
            model_name=models["stance"],
            batch_size=batch_size,
            device=device,
            clean_labels=clean_labels,
            quality_control=quality_control
        )
        
        # Apply stance post-processing if requested
        if clean_labels:
            print("   🧹 Applying stance label post-processing...")
            stance_df['Stance_Clean'] = stance_df['Stance'].apply(extract_clean_stance_label)
            clean_stance_counts = stance_df['Stance_Clean'].value_counts()
            print("   Clean stance distribution:")
            for stance, count in clean_stance_counts.items():
                print(f"     {stance}: {count}")
        else:
            stance_counts = stance_df["Stance"].value_counts()
            print("   Raw stance distribution:")
            for stance, count in stance_counts.items():
                print(f"     {stance}: {count}")
        
        print(f"✅ Step 3 Complete: Stance detection finished")
        
        # Save intermediate stance detection output if requested
        if save_intermediate_outputs:
            print(f"   💾 Saving stance detection output to: {stance_output_file}")
            # Save complete output with all columns from input (preserves all data)
            stance_df.to_csv(stance_output_file, index=False)
            print(f"   ✅ Stance detection output saved")
        
        # =====================================================================
        # STEP 4: Policy detection
        # =====================================================================
        print("\n📋 Step 4: Detecting policies directed towards groups...")
        
        step4_output = os.path.join(output_dir, f"temp_step4_{os.path.basename(output_file)}")
        
        policy_df = process_policy_csv(
            input_file=step3_output,
            text_column='text',
            group_column='Exact.Group.Text',
            output_file=step4_output,
            model_name=models["policy"],
            batch_size=batch_size,
            device=device,
            clean_labels=clean_labels,
            quality_control=quality_control
        )
        
        # Apply policy post-processing if requested
        if clean_labels:
            print("   🧹 Applying policy label post-processing...")
            policy_df['Policy_Clean'] = policy_df['Policy'].apply(extract_clean_policy_label)
            clean_policy_counts = policy_df['Policy_Clean'].value_counts()
            print("   Clean policy distribution:")
            for policy, count in clean_policy_counts.items():
                print(f"     {policy}: {count}")
        else:
            policy_counts = policy_df["Policy"].value_counts()
            print("   Raw policy distribution:")
            for policy, count in policy_counts.items():
                print(f"     {policy}: {count}")
        
        print(f"✅ Step 4 Complete: Policy detection finished")
        
        # Save intermediate policy detection output if requested
        if save_intermediate_outputs:
            print(f"   💾 Saving policy detection output to: {policy_output_file}")
            # Save complete output with all columns from input (preserves all data)
            policy_df.to_csv(policy_output_file, index=False)
            print(f"   ✅ Policy detection output saved")
        
        # =====================================================================
        # STEP 5: Group classification
        # =====================================================================
        print("\n🏷️  Step 5: Classifying groups into meaningful categories...")
        
        step5_output = os.path.join(output_dir, f"temp_step5_{os.path.basename(output_file)}")
        
        full_models_output = process_groups_csv(
            input_file=step4_output,
            group_column='Exact.Group.Text',
            output_file=step5_output,
            model_repo=models["classification"],
            score_threshold=0.5,
            device=device,
            split_groups=False  # We'll handle splitting in post-processing
        )
        
        # Apply meaningful groups post-processing
        print("   🧹 Applying meaningful groups post-processing...")
        
        # Count categories
        all_categories = []
        for labels in full_models_output["Meaningful Group"]:
            parsed_labels = parse_predicted_labels(labels)
            if parsed_labels:
                all_categories.extend(parsed_labels)
        
        if all_categories:
            category_counts = pd.Series(all_categories).value_counts()
            print("   Category distribution:")
            for category, count in category_counts.items():
                print(f"     {category}: {count}")
        else:
            print("   No meaningful group categories assigned")
        
        # Split meaningful groups into separate columns if requested
        if split_groups:
            print("   🔄 Splitting meaningful groups into separate columns...")
            full_models_output = split_meaningful_groups_into_columns(
                full_models_output, 
                meaningful_groups_column='Meaningful Group'
            )
        
        print(f"✅ Step 5 Complete: Group classification finished")
        
        # Save intermediate meaningful groups output if requested
        if save_intermediate_outputs:
            print(f"   💾 Saving meaningful groups output to: {groups_output_file}")
            # Save complete output with all columns from input (preserves all data)
            full_models_output.to_csv(groups_output_file, index=False)
            print(f"   ✅ Meaningful groups output saved")
        
        # =====================================================================
        # STEP 6: Prepare final dataset
        # =====================================================================
        print("\n🔗 Step 6: Preparing final dataset...")
        
        # Add missing columns to no_detected_groups to match full_models_output structure
        base_columns = ['Stance', 'Stance_Confidence', 'Policy', 'Policy_Confidence', 'Meaningful Group']
        
        # Add clean label columns if they exist
        if clean_labels:
            base_columns.extend(['Stance_Clean', 'Policy_Clean'])
        
        # Add Group columns if they exist
        group_columns = [col for col in full_models_output.columns if col.startswith('Group') and col != 'Meaningful Group']
        missing_columns = base_columns + group_columns
        
        for col in missing_columns:
            if col in full_models_output.columns and col not in no_detected_groups.columns:
                no_detected_groups[col] = pd.NA
        
        # Concatenate rows with groups (full_models_output) and rows without groups (no_detected_groups)
        final_df = pd.concat([full_models_output, no_detected_groups], ignore_index=True)
        
        # Validation checks
        expected_total_rows = len(full_models_output) + len(no_detected_groups)
        actual_total_rows = len(final_df)
        if expected_total_rows == actual_total_rows:
            print(f"✅ Row count validation passed: {actual_total_rows} total rows")
        else:
            print(f"⚠️ Row count validation failed: expected {expected_total_rows}, got {actual_total_rows}")
        
        # Coverage validation: check that all base text_ids from input are present in final output
        # Extract base IDs (remove .0, .1, .2 suffixes from token classification)
        def get_base_id(text_id):
            """Extract base ID by removing .0, .1, .2 suffix if present."""
            if pd.isna(text_id):
                return text_id
            text_id_str = str(text_id)
            # Check if it ends with .N pattern (dot followed by digits)
            if '.' in text_id_str and text_id_str.split('.')[-1].isdigit():
                return '.'.join(text_id_str.split('.')[:-1])
            return text_id_str
        
        # Load original input to get the true original IDs (before token classification numbering)
        try:
            if create_composite_id:
                # If we created composite IDs, get the base IDs from processed input
                input_df = pd.read_csv(input_file)
                original_base_ids = set(input_df[id_column].astype(str).unique())
            else:
                # Get base IDs from original input file
                input_df = pd.read_csv(input_file)
                original_base_ids = set(input_df[id_column].astype(str).unique())
        except:
            # Fallback: extract base IDs from token classification output
            original_base_ids = set(get_base_id(text_id) for text_id in extraction_df['text_id'].unique())
        
        final_base_ids = set(get_base_id(text_id) for text_id in final_df['text_id'].unique())
        missing_base_ids = original_base_ids - final_base_ids
        
        if not missing_base_ids:
            print(f"✅ Coverage validation passed: all {len(original_base_ids)} unique base text_ids preserved")
        else:
            print(f"⚠️ Coverage validation failed: {len(missing_base_ids)} base text_ids missing from final output")
            print(f"   Missing IDs: {list(missing_base_ids)[:5]}...")  # Show first 5 missing IDs
        
        # Reorder columns for better readability
        base_column_order = [
            'text_id',
            'original_text', 
            'text',
            'Exact.Group.Text',
            'Average Score',
            'Start',
            'End', 
            'Stance',
            'Stance_Confidence',
            'Policy',
            'Policy_Confidence',
            'Meaningful Group'
        ]
        
        # Add clean label columns if they exist
        if clean_labels:
            # Insert clean columns right after their raw counterparts
            stance_idx = base_column_order.index('Stance_Confidence') + 1
            policy_idx = base_column_order.index('Policy_Confidence') + 1
            base_column_order.insert(stance_idx, 'Stance_Clean')
            base_column_order.insert(policy_idx + 1, 'Policy_Clean')  # +1 because we inserted Stance_Clean
        
        # Add Group columns dynamically (Group1, Group2, etc.)
        group_columns = [col for col in final_df.columns if col.startswith('Group') and col != 'Meaningful Group']
        group_columns.sort()  # Ensure consistent ordering: Group1, Group2, Group3, etc.
        
        # Complete column order with Group columns after Meaningful Group
        column_order = base_column_order + group_columns
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in final_df.columns]
        final_df = final_df[available_columns]
        
        print(f"✅ Step 6 Complete: Combined full analysis and no-groups data")
        print(f"   Final dataset contains {len(final_df)} rows with {len(available_columns)} columns")
        print(f"   - Rows with full analysis: {len(full_models_output)}")
        print(f"   - Rows without group detection: {len(no_detected_groups)}")
        
        # =====================================================================
        # Save final results
        # =====================================================================
        print(f"\n💾 Saving final results to: {output_file}")
        
        try:
            final_df.to_csv(output_file, index=False)
            print(f"✅ Results successfully saved to: {output_file}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to save results to {output_file}: {str(e)}")
            print("   Continuing with in-memory results...")
        
        # =====================================================================
        # Clean up temporary files
        # =====================================================================
        print("\n🧹 Cleaning up temporary files...")
        
        temp_files = [
            step1_output, step2_output, step3_output, 
            step4_output, step5_output
        ]
        
        # Add processed input file to cleanup if it was created
        if create_composite_id:
            processed_input_temp = os.path.join(output_dir, f"temp_processed_input_{os.path.basename(output_file)}")
            if os.path.exists(processed_input_temp):
                temp_files.append(processed_input_temp)
        
        cleaned_count = 0
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    cleaned_count += 1
            except Exception as e:
                print(f"   Warning: Could not remove {temp_file}: {str(e)}")
        
        print(f"✅ Cleaned up {cleaned_count} temporary files")
        
        # =====================================================================
        # Final summary
        # =====================================================================
        print("\n" + "=" * 60)
        print("🎉 FULL PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"📊 FINAL RESULTS SUMMARY:")
        print(f"   • Total rows processed: {len(final_df)}")
        print(f"   • Rows with group extractions: {final_df['Exact.Group.Text'].notna().sum()}")
        print(f"   • Rows with stance analysis: {final_df['Stance'].notna().sum()}")
        print(f"   • Rows with policy analysis: {final_df['Policy'].notna().sum()}")
        print(f"   • Rows with group classification: {final_df['Meaningful Group'].notna().sum()}")

        print(f"\n💾 Results saved to: {output_file}")
        
        if save_intermediate_outputs:
            print(f"\n📁 Intermediate outputs saved to: {intermediate_output_dir}")
            print(f"   • Token classification: {os.path.basename(token_output_file)}")
            print(f"   • No detected groups: {os.path.basename(no_groups_file)}")
            print(f"   • Stance detection: {os.path.basename(stance_output_file)}")
            print(f"   • Policy detection: {os.path.basename(policy_output_file)}")
            print(f"   • Meaningful groups: {os.path.basename(groups_output_file)}")
        
        print("=" * 60)
        
        return final_df
        
    except Exception as e:
        error_msg = f"Error in full pipeline: {str(e)}"
        print(f"\n❌ {error_msg}")
        
        # Try to clean up any temporary files that might exist
        try:
            temp_files = [f for f in os.listdir(output_dir) if f.startswith('temp_') and os.path.basename(output_file) in f]
            for temp_file in temp_files:
                try:
                    os.remove(os.path.join(output_dir, temp_file))
                except:
                    pass
        except:
            pass
            
        raise RuntimeError(error_msg) from e