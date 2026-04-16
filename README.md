# GroupAppeals

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for analyzing social group appeals in political text using fine-tuned multilingual language models.

## Overview

GroupAppeals provides a comprehensive toolkit for identifying and analyzing how political parties reference and appeal to social groups in their communications. The package uses state-of-the-art NLP models trained on political manifestos to:

- **Extract group references** from text (tokens like "workers", "immigrants", "families with young children")
- **Detect stance** toward the groups identified in the text (positive, negative, or neutral)
- **Identify whether policies** are directed at the specific groups identified in the text (no thematic policy classification)
- **Classify group tokens** identified in the text into meaningful group categories

## Key Features

- 🌍 **Multilingual support** - Works with English, German, Spanish, Dutch, Danish, French, Italian and Swedish text
- 🔧 **Modular design** - Use individual components or run the complete pipeline
- 📊 **Batch processing** - Efficiently analyze large datasets from CSV files
- 🎯 **High accuracy** - Models achieve 81%+ accuracy across tasks
- 📝 **Detailed output** - Includes confidence scores, text positions, and semantic categories
- 🚀 **Automatic hardware optimization** - CUDA → MPS → CPU fallback

## Research Background

The conceptual and operational basis of these models as well as the methodology used to construct them are described in:

- **Dolinsky, A. O., Huber, L. M., & Horne, W.** (Accepted for Publication 2026). *Who do Parties Speak To? Introducing the PSoGA: A New Comprehensive Database of Parties' Social Group Appeals*. British Journal of Political Science.
- **Horne, W., Dolinsky, A. O., & Huber, L. M.** (2025). *Using LLMs to Detect Group Appeals in Parties’ Election Manifestos*. Working Paper. https://osf.io/fp2h3_v3
- **Huber, L.M., & Dolinsky, A.O.** (2023).How parties shape their relationship with social groups: A roadmap to the study of group-based appeals. Working Paper. https://osf.io/preprints/osf/szaqw_v1

**Model Training Details:**
- Models were trained and validated using political parties' general election manifestos.
- Models were also validated for performance in processing parliamentary speeches (English only) 
- Token classification uses plain text format without special formatting requirements
- Stance and policy models use Natural Language Inference (NLI) approaches
- Natural sentences were used as the unit of analysis for the token classifier, stance and policy detection; tokens were used as the unit of analysis for the multi-label meaningful group classifier
- Training data includes English, German text. The four models were further validated on held-out samples English, German, Dutch, Danish, Spanish, French, Italian and Swedish

### Current Implementation

The package provides a streamlined approach:

- ✅ **Token extraction**: Plain text processing with transformer token classification
- ✅ **Stance detection**: NLI-based approach for stance detection 
- ✅ **Policy detection**: NLI-based approach for policy identification (binary detector)
- ✅ **Group classification**: Multi-label classification of social group categories

We thank Josh Allen (joshuafayallen), Dylan Paltra (dpltr22) and Marvin Stecker (vestedinterests) for their support and contributions to the release of this package.

### Performance Considerations

- **Batch processing** optimizes performance for multiple texts
- **Pipeline integration** handles data flow between components
- **Hardware acceleration** - GPU is auto-detected (CUDA → MPS → CPU). Override with `device="cuda"` in Python or `--device cuda` in the CLI for any command (`extract`, `stance`, `policy`, `classify`, `pipeline`)

### Best Practices

1. **Use meaningful IDs** (party_date_sentence) for traceability through the pipeline
2. **Prepare data properly** using the pre-processing functions
3. **Test with small datasets** first to validate format and performance

## Pipeline Data Flow

The package follows a 6-step workflow:

### 1. Data Preparation
- **Input**: Raw CSV with party (or any other political actor), date, sentence_id, text columns
- **Processing**: Create composite IDs for traceability
- **Output**: Text with meaningful text_id identifiers

### 2. Token Classification
- **Input**: Text with text_id
- **Processing**: Extract social group references using transformer model
- **Output**: Entities with positions and confidence scores

### 3. Data Filtering  
- **Processing**: Separate texts with/without social group mentions
- **Purpose**: Optimize downstream processing and enable complete dataset reconstruction

### 4. Analysis Steps
- **Stance Detection**: Determine stance toward identified groups (positive, negative, neutral)
- **Policy Detection**: Identify whether policy content directed at groups is included in the text 
- **Meaningful Groups**: Classify groups into meaningful categories

### 5. Post-processing
- **Data cleaning**: Process model outputs into clean, usable formats
  - Create `Stance_Clean` and `Policy_Clean` columns with simplified labels ('positive'/'negative'/'neutral', 'policy'/'no policy')
  - Split meaningful groups into separate `Group1`, `Group2`, etc. columns when requested
- **Output**: Both raw model predictions and clean processed labels

### 6. Final Dataset Assembly
- **Merging**: Combine processed texts (with groups) and unprocessed texts (without groups) into complete dataset
- **Validation**: Ensure all original texts are preserved with appropriate analysis results

## Documentation

For detailed usage examples, API reference, and advanced features, see our [complete documentation](https://github.com/alonadoli/GroupAppeals/blob/main/DOCUMENTATION.md).

## Requirements

- Python 3.8+
- PyTorch 2.3.1+ (for NumPy 2.x compatibility and Apple Silicon MPS support)
- Transformers 4.20.0+
- Pandas 1.3.0+
- NumPy 1.21.0+
- tqdm 0.62.0+
- openpyxl 3.0.0+

### System Requirements

**Note:** This package requires PyTorch 2.3.1+ for NumPy 2.x compatibility. Intel-based Macs may experience installation issues due to limited PyTorch wheel availability for older architectures. The package is fully supported on Apple Silicon Macs, Linux, and Windows systems.

## Citation

If you use GroupAppeals in your research, please cite:

```bibtex
@misc{groupappeals2026,
  title={GroupAppeals: A Python Package for Analyzing Social Group Appeals in Political Texts},
  author={Dolinsky, Alona O. and Horne, Will and Huber, Lena Maria},
  year={2026},
  url={https://github.com/alonadoli/GroupAppeals}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Alona O. Dolinsky** - adolins2@jhu.edu
- **Will Horne** - rwhorne@clemson.edu  
- **Lena Maria Huber** - lena.huber@uni-mannheim.de

## Acknowledgments

This work was supported by funds from the EU's Horizon Europe MSCA Postdoctoral Fellowships under grant agreement no. 101107835.

## Package Details:

## Input Data Requirements

GroupAppeals processes plain text data and handles ID management for pipeline traceability:

### Data Preparation

**Step 1: Raw Data Format**

Your CSV should contain these columns:
- `party` (or political actor)
- `date` (or election year/time identifier - converted to string, does not require date formatting)
- `sentence_id` (row within text)
- `text` (the actual text content)

**Step 2: ID Creation**

Use the pre-processing functions to create composite IDs:

```python
from groupappeals.pre_and_post_processing import create_composite_id

# Create meaningful composite IDs for traceability
df['text_id'] = create_composite_id(df, 
                                  party_col="party", 
                                  date_col="date", 
                                  sentence_col="sentence_id")
```

**Step 3: Pipeline Processing**

The package processes plain text through transformer models without requiring special formatting.

**Benefits:**
- **ID traceability**: Track results back to original party/date/sentence
- **Flexible input**: Works with any political actor type (parties, candidates, organizations)
- **Pipeline integration**: Seamless data flow between analysis steps
- **Complete reconstruction**: Merge processed and unprocessed texts

## Installation

```bash
pip install groupappeals
```

## Quick Start

### Complete Pipeline (Recommended for Most Users)

Use the full pipeline when you want complete analysis. The pipeline handles composite ID creation, group extraction, stance detection, policy detection, and group classification.

```python
from groupappeals.fullpipeline import run_full_pipeline

# Complete analysis starting from raw political text
results = run_full_pipeline(
    input_file="raw_political_text.csv",
    output_file="complete_analysis.csv",
    create_composite_id=["party", "date", "sentence_id"]
)

print(f"Analyzed {len(results)} sentences")
print("Sample results:")
print(results[['text_id', 'Exact.Group.Text', 'Stance', 'Policy']].head())
```

### Step-by-Step Processing (For Custom Control)

> **Two ways to call each module:** Each module provides a **CSV function** (e.g. `process_csv`) that reads from a file path, and an equivalent **Python list function** (e.g. `extract_entities`) that works directly with data already in memory. Both produce identical output. The CSV functions are shown in the examples below; the equivalent Python list function is noted under each one.

Use individual modules when you need:
- **Intermediate result inspection** between steps
- **Selective processing** (skip certain analyses)
- **Different parameters** for each step

```python
from groupappeals.pre_and_post_processing import create_composite_id
from groupappeals.extractgrouptoken import process_csv
import pandas as pd

# Step 1: Prepare data with meaningful IDs
df = pd.read_csv("raw_political_text.csv")
df['text_id'] = create_composite_id(df, 
                                  party_col="party", 
                                  date_col="election_year", 
                                  sentence_col="sentence_id")

# Select required columns and save
prepared_df = df[['text', 'text_id']]
prepared_df.to_csv("prepared_data.csv", index=False)

# Step 2: Extract group references
token_results = process_csv(
    input_file="prepared_data.csv",
    text_column="text",
    id_column="text_id",
    output_file="extracted_groups.csv"
)

# Step 3: Filter for entities (for downstream processing)
entities_only = token_results[token_results['Exact.Group.Text'].notna()]

print(f"Found {len(entities_only)} social group mentions")
print("Sample extractions:")
print(entities_only[['text_id', 'Exact.Group.Text', 'Average Score']].head())
```

### Standalone Stance Detection

Use this for stance analysis without the full pipeline. The CSV function is `process_stance_csv`; the equivalent Python list function is `detect_stance`.

```python
from groupappeals.stancedetection import process_stance_csv
import pandas as pd

# Create sample data with text and group columns
data = {
    'text': [
        "We will support working families with new policies.",
        "Small business owners deserve better treatment.",
        "Students need more affordable education."
    ],
    'group': ["working families", "small business owners", "students"],
    'text_id': ["example_1", "example_2", "example_3"]
}

df = pd.DataFrame(data)
df.to_csv("stance_input.csv", index=False)

# Process CSV file for stance detection
stance_results = process_stance_csv(
    input_file="stance_input.csv",
    text_column="text",
    group_column="group",
    output_file="stance_results.csv",
    clean_labels=True  # Creates Stance_Clean column with simplified labels
)

print(f"Processed {len(stance_results)} text-group pairs")

# Display results
for _, row in stance_results.iterrows():
    print(f"Text: {row['text'][:50]}...")
    print(f"Group: {row['group']}")
    print(f"Raw Stance: {row['Stance']}")
    if 'Stance_Clean' in stance_results.columns:
        print(f"Clean Stance: {row['Stance_Clean']} (confidence: {row['Stance_Confidence']:.3f})")
    print("---")
```

### Standalone Policy Detection

Use this for policy analysis without the full pipeline. The CSV function is `process_policy_csv`; the equivalent Python list function is `detect_policy`.

```python
from groupappeals.policydetection import process_policy_csv
import pandas as pd

# Create sample data with text and group columns
data = {
    'text': [
        "We will implement new childcare support for working families.",
        "Small business owners face challenges in our economy.",
        "Students deserve access to quality education."
    ],
    'group': ["working families", "small business owners", "students"],
    'text_id': ["example_1", "example_2", "example_3"]
}

df = pd.DataFrame(data)
df.to_csv("policy_input.csv", index=False)

# Process CSV file for policy detection
policy_results = process_policy_csv(
    input_file="policy_input.csv",
    text_column="text",
    group_column="group",
    output_file="policy_results.csv",
    clean_labels=True  # Creates Policy_Clean column with 'policy'/'no policy' labels
)

print(f"Processed {len(policy_results)} text-group pairs")

# Display results
for _, row in policy_results.iterrows():
    print(f"Text: {row['text'][:50]}...")
    print(f"Group: {row['group']}")
    print(f"Raw Policy: {row['Policy']}")
    if 'Policy_Clean' in policy_results.columns:
        print(f"Clean Policy: {row['Policy_Clean']} (confidence: {row['Policy_Confidence']:.3f})")
    print("---")

print("\nPolicy distribution:")
if 'Policy_Clean' in policy_results.columns:
    print(policy_results['Policy_Clean'].value_counts())
else:
    print(policy_results['Policy'].value_counts())
```

### Standalone Meaningful Groups Classification

Use this for categorizing social group references without the full pipeline. The CSV function is `process_groups_csv`; the equivalent Python list function is `classify_groups`.

```python
from groupappeals.classifymeaningfulgroups import process_groups_csv
import pandas as pd

# Create sample data with group references
data = {
    'group_text': [
        "working families",
        "small business owners", 
        "students",
        "elderly citizens",
        "immigrants"
    ],
    'text_id': ["group_1", "group_2", "group_3", "group_4", "group_5"]
}

df = pd.DataFrame(data)
df.to_csv("groups_input.csv", index=False)

# Process CSV file for group classification
classification_results = process_groups_csv(
    input_file="groups_input.csv",
    group_column="group_text",
    output_file="classified_groups.csv",
    score_threshold=0.5,
    split_groups=True  # Create separate Group1, Group2, etc. columns
)

print(f"Processed {len(classification_results)} group references")

# Display results
print("Sample classifications:")
for _, row in classification_results.iterrows():
    print(f"Group: {row['group_text']}")
    print(f"Categories: {row['Meaningful Group']}")
    
    # Show individual Group columns if split_groups=True
    group_cols = [col for col in row.index if col.startswith('Group') and pd.notna(row[col])]
    if group_cols:
        group_values = [f"{col}: '{row[col]}'" for col in group_cols]
        print(f"Split Categories: {', '.join(group_values)}")
    print("---")

print("\nOverall statistics:")
print(classification_results[['group_text', 'Meaningful Group']].head())
```

## Models and Performance

The package uses four specialized models for comprehensive social group analysis:

| Model | Task | Languages | Model Architecture |
|-------|------|-----------|------------------|
| **Token Classifier** | Group extraction | EN, DE, ES, NL, DA, FR, IT, SV | Transformer token classification |
| **Stance NLI** | Positive/negative/neutral stance | EN, DE, ES, NL, DA, FR, IT, SV | Natural Language Inference |
| **Policy NLI** | Policy detection | EN, DE, ES, NL, DA, FR, IT, SV | Natural Language Inference |
| **Group Classifier** | Semantic categorization | EN, DE, ES, NL, DA, FR, IT, SV | Multi-label classification |

## Input Data Format

### Raw Data Format (Recommended Starting Point)

Your CSV file should contain these columns for optimal traceability:

```csv
party,date,sentence_id,text
PartyA,2023,1,"We will support working families with new childcare policies."
PartyA,2023,2,"Small businesses are the backbone of our economy."
PartyB,2023,1,"Students deserve access to affordable education."
```

### Prepared Data Format (After Processing)

After using `create_composite_id()`, your data will have meaningful unit IDs:

```csv
text,text_id
"We will support working families with new childcare policies.","PartyA_2023_1"
"Small businesses are the backbone of our economy.","PartyA_2023_2"
"Students deserve access to affordable education.","PartyB_2023_1"
```

### Token Classification Output Format

The `extract_entities()` function produces this structure:

```csv
text_id,text,Entity,Average Score,Start,End,Exact.Group.Text
"PartyA_2023_1.1","We will support working families...","working families",0.95,16,31,"working families"
"PartyA_2023_2.0","Small businesses are the backbone...","",,,""
"PartyB_2023_1.1","Students deserve access...","Students",0.87,0,8,"Students"
```

**Note:** The `.0`, `.1`, `.2` numbering indicates:
- `.0` = No entities found
- `.1` = First entity found  
- `.2` = Second entity found (if multiple entities in same text)

## Example Output

### Complete Pipeline Output Format

The full pipeline produces comprehensive results with both raw model outputs and clean processed labels:

```csv
text_id,text,Exact.Group.Text,Average Score,Stance,Stance_Confidence,Stance_Clean,Policy,Policy_Confidence,Policy_Clean,Meaningful Group,Group1,Group2
"PartyA_2023_1.1","We will support working families with new policies.","working families",0.95,"The text is positive towards working families.",0.95,"positive","The text contains a policy directed towards working families.",0.89,"policy","['Families', 'Workers']","Families","Workers"
"PartyA_2023_2.1","Small businesses are the backbone of our economy.","small businesses",0.89,"The text is positive towards small businesses.",0.89,"positive","The text does not contain a policy directed towards small businesses.",0.67,"no policy","['Economic Groups']","Economic Groups",""
"PartyB_2023_1.0","This is a general statement about the economy.","","","","","","","","","",""
```

### Key Output Features:

- **Raw Model Outputs**: Complete verbose predictions from each model
- **Clean Labels**: Simplified labels (`positive`/`negative`/`neutral`, `policy`/`no policy`)
- **Group Categories**: Both list format (`Meaningful Group`) and split columns (`Group1`, `Group2`, etc.)
- **Confidence Scores**: Model confidence for stance and policy predictions
- **Token Positions**: Character positions of extracted groups (`Start`, `End` columns)
- **Complete Coverage**: All input texts preserved, even those without group references

## Command Line Interface

### Complete Pipeline

```bash
# Complete pipeline WITH composite ID creation (recommended)
groupappeals pipeline --input manifestos.csv --output results.csv \
  --create-composite-id party,year,sentence_id --clean-labels --split-groups

# Complete pipeline with EXISTING text_id column
groupappeals pipeline --input texts.csv --output results.csv \
  --clean-labels --split-groups
```

### Individual Module Usage

```bash
# Individual modules (for step-by-step processing)
groupappeals extract --input texts.csv --output groups.csv
groupappeals stance --input groups.csv --output stance.csv --clean-labels
groupappeals policy --input stance.csv --output policy.csv --clean-labels
groupappeals classify --input policy.csv --output final.csv --split-groups
```

### Advanced Options

```bash
# Specify custom columns
groupappeals extract --input data.csv --output groups.csv \
  --text-column my_text --id-column my_id

# Custom batch size
groupappeals stance --input groups.csv --output stance.csv --batch-size 16

# Device selection (optional - auto-detected by default)
groupappeals pipeline --input data.csv --output results.csv --device cuda
groupappeals pipeline --input data.csv --output results.csv --device mps  # Apple Silicon
groupappeals pipeline --input data.csv --output results.csv --device cpu
```
