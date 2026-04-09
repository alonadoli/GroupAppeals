"""GroupAppeals package for text analysis."""

# Core module imports
from .extractgrouptoken import extract_entities, process_csv
from .stancedetection import detect_stance, process_stance_csv
from .policydetection import detect_policy, process_policy_csv
from .classifymeaningfulgroups import classify_groups, process_groups_csv


# Pre and post-processing utility imports
from .pre_and_post_processing import (
    extract_clean_stance_label,
    extract_clean_policy_label, 
    extract_group_from_hypothesis,
    parse_predicted_labels,
    split_meaningful_groups_into_columns,
    create_composite_id
)

# Device utility imports
from .device_utilities import determine_compute_device, convert_device_to_pipeline_id

# Full pipeline import
from .fullpipeline import run_full_pipeline

__version__ = "1.0.0"

# CLI entry point
def cli_main():
    """CLI entry point."""
    from .cli import main
    return main()
