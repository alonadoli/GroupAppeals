"""Command-line interface for GroupAppeals package."""

import argparse
import sys
import os
import pandas as pd
from . import (extract_entities, process_csv,
               detect_stance, process_stance_csv,
               detect_policy, process_policy_csv,
               process_groups_csv, run_full_pipeline)
from .pre_and_post_processing import parse_predicted_labels

def main():
    """Main entry point for the GroupAppeals CLI."""
    parser = argparse.ArgumentParser(
        description="GroupAppeals - A tool for analyzing group references in text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract groups from text:
  groupappeals extract --input input.csv --output groups.csv

  # Detect stance towards groups:
  groupappeals stance --input groups.csv --output stance_results.csv --clean-labels

  # Detect policies directed towards groups:
  groupappeals policy --input stance_results.csv --output policy_results.csv --clean-labels

  # Classify groups into meaningful categories:
  groupappeals classify --input policy_results.csv --output final_results.csv

  # Run full pipeline (basic usage):
  groupappeals pipeline --input input.csv --output final_results.csv

  # Run full pipeline with composite IDs:
  groupappeals pipeline --input manifestos.csv --output results.csv \\
    --create-composite-id party,date,sentence_id --clean-labels --split-groups

  # Run full pipeline with custom composite ID columns:
  groupappeals pipeline --input data.csv --output results.csv \\
    --create-composite-id political_party,election_year,sentence_id --clean-labels
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True
    
    # Common arguments
    input_arg = lambda p: p.add_argument("--input", required=True, help="Input CSV file")
    output_arg = lambda p: p.add_argument("--output", help="Output CSV file (optional)")
    text_col_arg = lambda p: p.add_argument("--text-column", default="text", help="Column containing text (default: text)")
    raw_text_col_arg = lambda p: p.add_argument("--text-column", default="text", help="Column containing raw text (default: text)")
    id_col_arg = lambda p: p.add_argument("--id-column", default="text_id", help="Column containing IDs (default: text_id)")
    group_col_arg = lambda p: p.add_argument("--group-column", default="Exact.Group.Text", help="Column containing group references (default: Exact.Group.Text)")
    device_arg = lambda p: p.add_argument("--device", choices=["cuda", "mps", "cpu"], help="Computation device (default: auto-detect)")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract group references from text")
    input_arg(extract_parser)
    output_arg(extract_parser)
    text_col_arg(extract_parser)
    id_col_arg(extract_parser)
    device_arg(extract_parser)
    
    # Stance command
    stance_parser = subparsers.add_parser("stance", help="Detect stance towards groups")
    input_arg(stance_parser)
    output_arg(stance_parser)
    text_col_arg(stance_parser)
    group_col_arg(stance_parser)
    stance_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing (default: 32)")
    device_arg(stance_parser)
    stance_parser.add_argument("--clean-labels", action="store_true",
                              help="Extract clean stance labels ('positive', 'negative', 'neutral')")
    stance_parser.add_argument("--quality-control", action="store_true",
                              help="Run quality control checks on group extraction")
    
    # Policy command
    policy_parser = subparsers.add_parser("policy", help="Detect policies directed towards groups")
    input_arg(policy_parser)
    output_arg(policy_parser)
    text_col_arg(policy_parser)
    group_col_arg(policy_parser)
    policy_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing (default: 32)")
    device_arg(policy_parser)
    policy_parser.add_argument("--clean-labels", action="store_true",
                              help="Extract clean policy labels ('policy', 'no policy')")
    policy_parser.add_argument("--quality-control", action="store_true",
                              help="Run quality control checks on group extraction")
    
    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify groups into meaningful categories")
    input_arg(classify_parser)
    output_arg(classify_parser)
    group_col_arg(classify_parser)
    classify_parser.add_argument("--threshold", type=float, default=0.5, 
                               help="Threshold for accepting predictions (default: 0.5)")
    classify_parser.add_argument("--split-groups", action="store_true", 
                               help="Split meaningful groups into separate Group1, Group2, etc. columns")
    device_arg(classify_parser)
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full analysis pipeline")
    input_arg(pipeline_parser)
    output_arg(pipeline_parser)
    raw_text_col_arg(pipeline_parser)
    id_col_arg(pipeline_parser)
    pipeline_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing (default: 32)")
    pipeline_parser.add_argument("--create-composite-id", help="Columns to combine into composite ID (comma-separated, e.g., 'party,date,sentence_id')")
    pipeline_parser.add_argument("--clean-labels", action="store_true", help="Extract clean stance/policy labels ('positive', 'policy', etc.)")
    pipeline_parser.add_argument("--split-groups", action="store_true", help="Split meaningful groups into Group1, Group2, etc. columns")
    device_arg(pipeline_parser)
    
    args = parser.parse_args()
    
    try:
        # Check that the input file exists
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found")
            return 1
            
        # Set the default output file if not specified
        if args.output is None:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_output{ext}"
        
        # Execute the appropriate command
        if args.command == "extract":
            print(f"Extracting group references from {args.input}...")
            result = process_csv(
                input_file=args.input,
                text_column=args.text_column,
                id_column=args.id_column,
                output_file=args.output,
                device=args.device
            )
            print(f"Extracted {len(result)} rows. Results saved to {args.output}")
            
        elif args.command == "stance":
            print(f"Detecting stance towards groups in {args.input}...")
            result = process_stance_csv(
                input_file=args.input,
                text_column=args.text_column,
                group_column=args.group_column,
                output_file=args.output,
                batch_size=args.batch_size,
                device=args.device,
                clean_labels=args.clean_labels,
                quality_control=args.quality_control
            )
            stance_counts = result["Stance"].value_counts()
            print(f"Stance detection complete. Results saved to {args.output}")
            print("\nStance distribution:")
            for stance, count in stance_counts.items():
                print(f"  {stance}: {count}")
            
        elif args.command == "policy":
            print(f"Detecting policies directed towards groups in {args.input}...")
            result = process_policy_csv(
                input_file=args.input,
                text_column=args.text_column,
                group_column=args.group_column,
                output_file=args.output,
                batch_size=args.batch_size,
                device=args.device,
                clean_labels=args.clean_labels,
                quality_control=args.quality_control
            )
            policy_counts = result["Policy"].value_counts()
            print(f"Policy detection complete. Results saved to {args.output}")
            print("\nPolicy distribution:")
            for policy, count in policy_counts.items():
                print(f"  {policy}: {count}")
            
        elif args.command == "classify":
            print(f"Classifying groups in {args.input}...")
            result = process_groups_csv(
                input_file=args.input,
                group_column=args.group_column,
                output_file=args.output,
                score_threshold=args.threshold,
                device=args.device,
                split_groups=args.split_groups
            )
            # Count categories
            all_categories = []
            for labels in result["Meaningful Group"]:
                parsed_labels = parse_predicted_labels(labels)
                if parsed_labels:
                    all_categories.extend(parsed_labels)
            
            category_counts = pd.Series(all_categories).value_counts()
            print(f"Group classification complete. Results saved to {args.output}")
            print("\nCategory distribution:")
            for category, count in category_counts.items():
                print(f"  {category}: {count}")
            
        elif args.command == "pipeline":
            print("🚀 Running the full GroupAppeals analysis pipeline...")
            
            # Parse composite ID creation
            create_composite_id = None
            if args.create_composite_id:
                # Parse comma-separated column names into list
                create_composite_id = [col.strip() for col in args.create_composite_id.split(',')]
            
            # Map CLI arguments to run_full_pipeline parameters
            pipeline_kwargs = {
                'input_file': args.input,
                'output_file': args.output,
                'text_column': args.text_column,
                'create_composite_id': create_composite_id,
                'batch_size': args.batch_size,
                'device': args.device,
                'clean_labels': args.clean_labels,
                'split_groups': args.split_groups,
            }
            
            # Print configuration summary
            if create_composite_id:
                print(f"  🆔 Creating composite ID from: {', '.join(create_composite_id)}")
            if args.clean_labels:
                print("  🏷️  Clean labels will be extracted")
            if args.split_groups:
                print("  📊 Groups will be split into separate columns")
            
            try:
                # Run the full pipeline
                result_df = run_full_pipeline(**pipeline_kwargs)
                
                # Display summary statistics
                print(f"\n🎉 Pipeline complete! Processed {len(result_df)} rows.")
                
                # Show stance distribution if available
                if 'Stance' in result_df.columns and result_df['Stance'].notna().sum() > 0:
                    print(f"\n📈 Stance distribution:")
                    stance_counts = result_df['Stance'].value_counts()
                    for stance, count in stance_counts.items():
                        print(f"  {stance}: {count}")
                
                # Show policy distribution if available
                if 'Policy' in result_df.columns and result_df['Policy'].notna().sum() > 0:
                    print(f"\n📋 Policy distribution:")
                    policy_counts = result_df['Policy'].value_counts()
                    for policy, count in policy_counts.items():
                        print(f"  {policy}: {count}")
                
                # Show group classification distribution if available
                if 'Meaningful Group' in result_df.columns:
                    all_categories = []
                    for labels in result_df['Meaningful Group'].dropna():
                        parsed_labels = parse_predicted_labels(labels)
                        if parsed_labels:
                            all_categories.extend(parsed_labels)
                    
                    if all_categories:
                        print(f"\n🏷️  Group classification distribution:")
                        category_counts = pd.Series(all_categories).value_counts()
                        for category, count in category_counts.head(10).items():
                            print(f"  {category}: {count}")
                        if len(category_counts) > 10:
                            print(f"  ... and {len(category_counts) - 10} more categories")
                
                print(f"\n💾 Results saved to: {args.output}")
                
            except Exception as e:
                print(f"❌ Pipeline failed: {str(e)}")
                raise
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
