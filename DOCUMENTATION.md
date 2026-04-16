# GroupAppeals Package Documentation

## Overview

GroupAppeals is a Python package that provides a comprehensive toolkit for analyzing social group references in text using state-of-the-art language models. Originally developed for political science research on party manifestos, the package can be applied to any text requiring group reference analysis.

**Key Features:**
- **Plain text processing** - no special formatting required
- **Flexible usage** - use individual modules or complete automated pipeline  
- **Multi-device support** - automatic CUDA/MPS/CPU optimization
- **Comprehensive output** - raw predictions plus clean, interpretable labels
- **Traceability** - maintains data lineage with meaningful composite IDs

The package consists of **four core modules** that can be used independently or as an integrated pipeline:

1. **Group Entity Extraction** (`extractgrouptoken`): Identifies social group references in text using token classification
2. **Stance Detection** (`stancedetection`): Determines stance (positive/negative/neutral) toward identified groups using NLI
3. **Policy Detection** (`policydetection`): Identifies whether policies directed toward groups are included in the text using binary NLI classification
4. **Meaningful Groups Classification** (`classifymeaningfulgroups`): Categorizes group references into substantive social categories using multi-label classification

**Three Usage Patterns:**
- **Standalone**: Use individual modules independently for specific analysis tasks
- **Manual Pipeline**: Chain modules together with custom intermediate processing
- **Automated Pipeline**: Execute complete end-to-end analysis with `run_full_pipeline()`

---

## Package Architecture

### Core Module Integration

The GroupAppeals package uses a modular architecture where each component can operate independently or as part of an integrated workflow.


### Automated Pipeline Flow (`run_full_pipeline`)

```
CSV Input (text + metadata columns)
    ↓
1. DATA PREPARATION
   • Create composite IDs from multiple columns
   • Validate input format and columns
    ↓
2. GROUP EXTRACTION
   • Token classification: identify group references
   • Generate .1, .2, .3... IDs for multiple groups per text
    ↓
3. DATA OPTIMIZATION
   • Separate texts with/without groups
   • Optimize data flow for NLI processing
    ↓
4. STANCE DETECTION
   • NLI classification: positive/negative/neutral
   • Process only texts with identified groups
    ↓
5. POLICY DETECTION
   • NLI classification: policy/no policy  
   • Build on stance detection results
    ↓
6. GROUP CLASSIFICATION
   • Multi-label classification into social categories
   • Apply configurable confidence threshold
    ↓
7. POST-PROCESSING
   • Extract clean labels (optional)
   • Split groups into columns (optional)
    ↓
8. FINAL ASSEMBLY
   • Merge all results into complete dataset
   • Preserve all original input texts
    ↓
OUTPUT: Comprehensive CSV with raw + processed results

```

### Advanced Features

#### Intelligent Data Processing
- **Efficient batch processing**: Configurable batch sizes for large datasets
- **Smart data filtering**: Only process texts with identified groups for stance/policy
- **Robust error handling**: Continue processing despite individual text failures

#### Flexible Output Options
- **Raw model predictions**: Full confidence scores and detailed classifications
- **Clean interpretable labels**: Simple categorical outputs (`"positive"`, `"policy"`, etc.)
- **Structured group categories**: Multi-label classifications in list or column format
- **Complete data preservation**: All input texts retained regardless of processing success

---

## Installation

Install via pip:

```bash
pip install groupappeals
```

### System Requirements

**Note:** This package requires PyTorch 2.3.1+ for NumPy 2.x compatibility. Intel-based Macs may experience installation issues due to limited PyTorch wheel availability for older architectures. The package is fully supported on Apple Silicon Macs, Linux, and Windows systems.

---

## Getting Started

### Input Requirements

GroupAppeals works with **plain text CSV files** - no special formatting required. Your CSV should contain:

**Required columns:**

- **Text column**: Raw text data (e.g., quasi sentences, sentences, paragraphs, documents)

**For ID tracking (choose one approach):**

- **Option 1**: Single ID column with existing unique identifiers (e.g., `text_id`)
- **Option 2**: Multiple metadata columns to combine into composite IDs (e.g., `party`, `year`, `sentence_id`)

**Optional columns:**

- Any additional metadata (speaker, date, topic, etc.) - these are preserved in the output

**Example CSV structure:**

```csv
party,year,sentence_id,text
SPD,2021,1,"We support workers' rights."
CDU,2021,1,"Economic growth benefits everyone."
SPD,2021,2,"Healthcare is a fundamental right."
```

### Quick Start

**Complete pipeline (recommended for most users):**

```python
from groupappeals import run_full_pipeline

# Analyze political manifestos with composite IDs
results = run_full_pipeline(
    input_file="manifestos.csv",
    output_file="analysis_results.csv",
    create_composite_id=["party", "year", "sentence_id"],
    clean_labels=True,
    split_groups=True
)
```

**What this does:**

1. Creates traceable IDs from your metadata columns (e.g., `"SPD_2021_1"`)
2. Extracts group references from text
3. Detects stance toward each group (positive/negative/neutral)
4. Identifies policy statements directed at groups
5. Classifies groups into meaningful categories
6. Returns clean, structured results

**Command line interface:**

```bash
# Complete pipeline WITH composite ID creation (recommended)
groupappeals pipeline --input manifestos.csv --output results.csv \
  --create-composite-id party,year,sentence_id --clean-labels --split-groups

# Complete pipeline with EXISTING text_id column
groupappeals pipeline --input texts.csv --output results.csv \
  --clean-labels --split-groups

# Individual modules (for step-by-step processing)
groupappeals extract --input texts.csv --output groups.csv
groupappeals stance --input groups.csv --output stance.csv --clean-labels
groupappeals policy --input stance.csv --output policy.csv --clean-labels
groupappeals classify --input policy.csv --output final.csv --split-groups
```

### Understanding Composite IDs (Advanced)

**When to use `create_composite_id`:**

- You're running the full pipeline and want traceable identifiers
- You have metadata columns (party, speaker, date, etc.) that provide context
- You want to preserve metadata information throughout the analysis

**When to use `id_column`:**

- You already have unique identifiers in your data
- You're doing step-by-step processing with individual functions

**How composite IDs work:**

- `create_composite_id=["party", "year", "sentence_id"]` combines columns into `"SPD_2021_1"`
- The pipeline automatically handles ID creation and tracking
- Token extraction may create sub-IDs like `"SPD_2021_1.1"`, `"SPD_2021_1.2"` for multiple groups in one text

**Note:** The 'date' column is used only as a text identifier in the composite ID. It does not need to be formatted as a date - you can use years (2023), election cycles (election2024), or any other identifier.

**Warning:** Simple numeric IDs (1, 2, 3...) lose important context throughout the analysis pipeline. Use composite IDs for better traceability.

---

## Device Utilities Module

### Overview

The Device Utilities module provides automatic device detection and configuration for optimal performance across all GroupAppeals modules. It seamlessly handles CUDA, MPS (Apple Silicon), and CPU devices without requiring manual configuration.

### Functions

#### `determine_compute_device()`

Automatically detects the best available compute device for model inference.

**Returns:**
- `str`: Device identifier ('cuda', 'mps', or 'cpu')

**Detection Priority:**
1. **CUDA**: Selected if NVIDIA GPU with CUDA support is available
2. **MPS**: Selected if Apple Silicon Mac with MPS support is available  
3. **CPU**: Used as fallback for all other systems

#### `convert_device_to_pipeline_id(device)`

Converts device string to the format required by Hugging Face transformers pipelines.

**Parameters:**
- `device` (str): Device string ('cuda', 'mps', or 'cpu')

**Returns:**
- `int` or `str`: Pipeline device ID (0 for 'cuda', 'mps' for 'mps', -1 for 'cpu')

### Usage Examples

#### Automatic Device Selection

```python
from groupappeals import run_full_pipeline

# Device is automatically detected and used
results = run_full_pipeline(
    input_file="data.csv",
    output_file="results.csv"
)
```

#### Manual Device Override

```python
from groupappeals import run_full_pipeline

# Force specific device usage
results = run_full_pipeline(
    input_file="data.csv", 
    output_file="results.csv",
    device="cuda"  # or "mps" or "cpu"
)
```

#### Device Detection

```python
from groupappeals.device_utilities import determine_compute_device

device = determine_compute_device()
print(f"Optimal device: {device}")
```

### Hardware Support

#### NVIDIA CUDA
- **Requirements**: NVIDIA GPU with CUDA support
- **Performance**: 5-10x speedup over CPU
- **Setup**: Requires CUDA-compatible PyTorch installation

#### Apple Silicon MPS
- **Requirements**: Apple M1/M2/M3/M4 processor with macOS 12.3+
- **Performance**: 3-5x speedup over CPU with lower power consumption
- **Setup**: Automatic with PyTorch 1.12+

#### CPU
- **Requirements**: Any modern CPU
- **Performance**: Baseline performance, suitable for smaller datasets
- **Setup**: Always available

---

## Group Entity Extraction Module

### Overview

The Group Entity Extraction module identifies and extracts social group references from text using a specialized token classification model. This module can be used as a standalone tool for entity extraction or as the first step in the comprehensive GroupAppeals analysis pipeline. It processes plain text without any special formatting requirements.

### Functions

#### `extract_entities(texts, ids=None, model_name="rwillh11/mdberta-token-bilingual-noContext_Enhanced", device=None)`

Extracts social group entities from a list of texts.

**Parameters:**
- `texts` (list): List of text strings to analyze
- `ids` (list, optional): List of IDs for each text (auto-generated if not provided)
- `model_name` (str): Hugging Face model name (default: enhanced bilingual model)
- `device` (str, optional): Device ('cuda', 'mps', 'cpu') - auto-detected if not specified

**Returns:**
pandas DataFrame with columns:
- `text_id`: Unique identifier with entity numbering (e.g., "id.1", "id.2")
- `text`: Original text content
- `Entity`: Extracted group reference (joined tokens)
- `Average Score`: Confidence score (average across tokens)
- `Start`: Character start position in original text
- `End`: Character end position in original text
- `Exact.Group.Text`: Exact text extracted using positions (cleaned, ≥3 characters)

**ID Numbering System:**
- **0 entities found**: `original_id.0`
- **1 entity found**: `original_id.1`
- **Multiple entities**: `original_id.1`, `original_id.2`, `original_id.3`...

#### `process_csv(input_file, text_column="text", id_column="text_id", output_file=None, device=None)`

Processes a CSV file to extract entities from text.

**Parameters:**
- `input_file` (str): Path to input CSV file
- `text_column` (str): Column containing text to analyze (default: "text")
- `id_column` (str): Column containing identifiers (default: "text_id")
- `output_file` (str, optional): Full file path to save results (e.g., `"results.csv"` or `"/path/to/output.csv"`). Must include the filename — a directory path alone will fail.
- `device` (str, optional): Device to use for computation ('cuda', 'mps', or 'cpu') - auto-detected if not specified

**Returns:**
pandas DataFrame with extracted entities, preserving original structure

**Important:** Use meaningful composite IDs (e.g., "party_date_sentence") rather than simple numeric IDs to maintain traceability through the analysis pipeline.

### Input Requirements

**CSV File Format:**
- Any CSV separator supported (comma, semicolon, tab, etc.)
- Must contain a text column (default: "text") with plain text content
- Must contain an ID column (default: "text_id") with unique identifiers
- No other columns required

**Text Format:**
- Plain text without any special formatting requirements
- No length restrictions
- Handles empty/null text values gracefully

### Data Preparation

#### Basic Preparation
```python
import pandas as pd

# Minimal required CSV structure
df = pd.DataFrame({
    'text_id': ['doc1', 'doc2', 'doc3'],
    'text': [
        'We support working families in our community.',
        'Small businesses need tax relief.',
        'This helps students and teachers.'
    ]
})
df.to_csv('input.csv', index=False)
```

#### Creating Meaningful Composite IDs
```python
from groupappeals.pre_and_post_processing import create_composite_id

# Load data with multiple identifying columns
df = pd.read_csv('raw_data.csv')  # Contains: party, year, sentence_id, text

# Create composite IDs for better traceability
df['text_id'] = create_composite_id(df, 
                                  party_col='party', 
                                  date_col='year', 
                                  sentence_col='sentence_id')
# Result: text_id like "PartyA_2020_1", "PartyB_2020_1", etc.

prepared_df = df[['text_id', 'text']]
prepared_df.to_csv('prepared_input.csv', index=False)
```

### Output Format

The module returns a pandas DataFrame with these exact columns:

| Column | Description | Example |
|--------|-------------|---------|
| `text_id` | ID with entity numbering | "party1_2020_1.1" |
| `text` | Original text content | "We support working families." |
| `Entity` | Raw extracted tokens | "working families" |
| `Average Score` | Confidence score (0-1) | 0.892 |
| `Start` | Character start position | 11 |
| `End` | Character end position | 26 |
| `Exact.Group.Text` | Cleaned final entity | "working families" |

**Output Details:**
- **ID Numbering**: `.0` for no entities, `.1`, `.2`, `.3`... for multiple entities per text
- **Entity Filtering**: Groups under 3 characters are removed
- **Text Cleaning**: Trailing punctuation removed from `Exact.Group.Text`
- **Missing Values**: `pd.NA` used for texts with no detected entities

### Usage Examples

#### Basic Entity Extraction

```python
from groupappeals import extract_entities

texts = [
    "We need to support working families in our community.",
    "Small business owners deserve better tax policies.",
    "This policy will help students and teachers."
]

# Use meaningful IDs for traceability
ids = ["party1_2020_sentence1", "party2_2020_sentence1", "party1_2020_sentence2"]

token_results = extract_entities(texts, ids=ids)

# Display extracted groups
print("Extracted entities:")
for _, row in token_results.iterrows():
    if pd.notna(row['Exact.Group.Text']):
        print(f"{row['text_id']}: '{row['Exact.Group.Text']}' (confidence: {row['Average Score']:.3f})")
```

#### CSV File Processing

```python
from groupappeals import process_csv

# Process CSV file with meaningful IDs
token_results = process_csv(
    input_file="political_texts.csv",
    text_column="text",
    id_column="composite_id",  # e.g., "party_year_sentence"
    output_file="extracted_groups.csv"
)

print(f"Processed {len(token_results)} rows")
print(f"Found groups in {token_results['Exact.Group.Text'].notna().sum()} texts")
```

#### Integration with Composite ID Creation

```python
from groupappeals.pre_and_post_processing import create_composite_id
from groupappeals import process_csv
import pandas as pd

# Load and prepare data
df = pd.read_csv("raw_data.csv")

# Create meaningful composite IDs
df['text_id'] = create_composite_id(df, 
                                  party_col="party", 
                                  date_col="year", 
                                  sentence_col="sentence_id")

# Process entity extraction
token_results = process_csv(
    input_file="prepared_data.csv",
    text_column="text",
    id_column="text_id"
)
```

### Standalone vs Pipeline Usage

#### Standalone Usage
Use this module independently when you only need to extract group references:

```python
from groupappeals import extract_entities

# Extract entities from your texts
token_results = extract_entities(texts, ids)
groups_found = token_results[token_results['Exact.Group.Text'].notna()]
```

#### Pipeline Integration
The module integrates seamlessly with other GroupAppeals components for comprehensive analysis:

```python
from groupappeals import extract_entities, detect_stance

# Step 1: Extract entities
token_results = extract_entities(texts, ids)

# Step 2: Continue with stance detection on texts with groups
with_groups = token_results[token_results['Exact.Group.Text'].notna()]
stance_results = detect_stance(
    texts=with_groups['text'].tolist(),
    groups=with_groups['Exact.Group.Text'].tolist()
)
```

#### Full Pipeline Usage
For complete analysis, use the integrated pipeline:

```python
from groupappeals import run_full_pipeline

# Complete analysis from raw text to final results
results = run_full_pipeline(
    input_file="data.csv",
    output_file="complete_analysis.csv"
)
```

---

## Stance Detection Module

### Overview

The Stance Detection module analyzes the stance (positive, negative, or neutral) expressed towards social group references using a Natural Language Inference approach. This module can be used as a standalone tool for stance analysis or integrated with other GroupAppeals components for comprehensive text analysis.

### How It Works

The stance detection uses **Natural Language Inference (NLI)** to determine stance:

1. **Input**: Text + Group reference (e.g., "working families")
2. **Hypothesis Generation**: Creates three competing hypotheses:
   - "The text is positive towards working families."
   - "The text is negative towards working families."
   - "The text is neutral, or contains no stance, towards working families."
3. **NLI Classification**: Model determines which hypothesis is most supported by the text
4. **Output**: Returns the stance label from the most likely hypothesis with confidence score

### Functions

#### `detect_stance(texts, groups, model_name="rwillh11/mdeberta_NLI_stance_NoContext", batch_size=32, device=None)`

Detects stance towards groups from lists of texts and groups.

**Parameters:**
- `texts` (list): List of text strings to analyze
- `groups` (list): List of group references corresponding to each text
- `model_name` (str): Hugging Face NLI model name (default: "rwillh11/mdeberta_NLI_stance_NoContext")
- `batch_size` (int): Number of examples to process at once (default: 32)
- `device` (str, optional): Device ('cuda', 'mps', 'cpu') - auto-detected if not specified

**Returns:**
- `list`: List of dictionaries with stance results. Each dict contains:
  - `'stance'`: Stance hypothesis ("The text is positive/negative/neutral towards [group].")
  - `'confidence'`: Confidence score (float between 0.0 and 1.0)

#### `process_stance_csv(input_file, text_column="text", group_column="Exact.Group.Text", output_file=None, model_name="rwillh11/mdeberta_NLI_stance_NoContext", batch_size=32, device=None, clean_labels=False, quality_control=False)`

Processes CSV files to detect stance towards groups.

**Parameters:**
- `input_file` (str): Path to input CSV file
- `text_column` (str): Column containing text to analyze (default: "text")
- `group_column` (str): Column containing group references (default: "Exact.Group.Text")
- `output_file` (str, optional): Full file path to save results (e.g., `"results.csv"` or `"/path/to/output.csv"`). Must include the filename — a directory path alone will fail.
- `model_name` (str): Hugging Face NLI model name
- `batch_size` (int): Batch size for processing (default: 32)
- `device` (str, optional): Device to use for computation
- `clean_labels` (bool): Whether to extract clean stance labels (default: False)
- `quality_control` (bool): Whether to run quality control checks (default: False)

**Returns:**
- `pd.DataFrame`: Original DataFrame with added columns:
  - `"Stance"`: Stance labels (full hypothesis text)
  - `"Stance_Confidence"`: Confidence scores
  - `"Stance_Clean"`: Clean labels ('positive', 'negative', 'neutral') if clean_labels=True

### Input Requirements

**CSV File Format:**
- Must contain a text column (default: "text") with plain text content
- Must contain a group column (default: "Exact.Group.Text") with group references
- Group references cannot have missing/NaN values
- Additional columns preserved in output

**Data Format:**
- Plain text without special formatting requirements
- Group references as simple strings (e.g., "working families", "students")
- Typically uses output from Group Entity Extraction module

### Data Preparation

#### From Entity Extraction Output
```python
from groupappeals import process_csv, process_stance_csv

# Step 1: Extract entities (creates text + Exact.Group.Text columns)
token_results = process_csv("input.csv", text_column="text", id_column="text_id")

# Step 2: Filter for rows with detected groups (required for stance detection)
with_groups = token_results[token_results['Exact.Group.Text'].notna()]

# Save prepared data
with_groups.to_csv("prepared_for_stance.csv", index=False)
```

#### Manual Preparation
```python
import pandas as pd

# Create stance detection input manually
df = pd.DataFrame({
    'text_id': ['doc1.1', 'doc2.1', 'doc3.1'],
    'text': [
        'We will support working families with new policies.',
        'Small business owners deserve better treatment.',
        'Students need more affordable education.'
    ],
    'Exact.Group.Text': ['working families', 'small business owners', 'students']
})
df.to_csv('stance_input.csv', index=False)
```

### Output Format

The module returns a pandas DataFrame with original columns plus:

| Column | Description | Example |
|--------|-------------|---------|
| `Stance` | Full NLI hypothesis | "The text is positive towards working families." |
| `Stance_Confidence` | Confidence score (0-1) | 0.847 |
| `Stance_Clean` | Clean label (if enabled) | "positive" |

**Clean Labels (when clean_labels=True):**
- `"positive"` - Text takes positive stance towards the group
- `"negative"` - Text take negative stance towards the group  
- `"neutral"` - Text is neutral or contains no clear stance towards the group

### Standalone vs Pipeline Usage

#### Standalone Usage
Use this module independently when you already have group references:

```python
from groupappeals import detect_stance

texts = ["We support working families.", "Students deserve better funding."]
groups = ["working families", "students"]

results = detect_stance(texts, groups)
for result in results:
    print(f"Stance: {result['stance']}")
    print(f"Confidence: {result['confidence']:.3f}")
```

#### Pipeline Integration
Integrate with entity extraction for complete analysis:

```python
from groupappeals import process_csv, process_stance_csv

# Step 1: Extract entities
token_results = process_csv("text_data.csv")

# Step 2: Detect stance on texts with groups
entities_with_groups = token_results[token_results['Exact.Group.Text'].notna()]
stance_results = process_stance_csv(
    input_file="temp_entities.csv",  # or pass DataFrame directly
    text_column="text",
    group_column="Exact.Group.Text",
    clean_labels=True
)
```

#### Full Pipeline Usage
For complete analysis including stance detection:

```python
from groupappeals import run_full_pipeline

# Complete analysis including stance detection
results = run_full_pipeline(
    input_file="raw_data.csv",
    output_file="complete_analysis.csv",
    clean_labels=True  # Include clean stance labels
)
```

### Error Handling

- **Missing Groups**: Raises ValueError if any group references are missing/NaN
- **Model Loading**: Clear error messages for model loading failures
- **Processing Errors**: Individual text failures result in "unknown" stance, processing continues
- **File Handling**: Proper error handling for CSV reading/writing operations

---

## Policy Detection Module

### Overview

The Policy Detection module identifies whether text contains concrete policy proposals directed towards specific social groups using a Natural Language Inference approach. This module can be used as a standalone tool for policy analysis or integrated with other GroupAppeals components for comprehensive text analysis.

### How It Works

The policy detection uses **Natural Language Inference (NLI)** with binary classification:

1. **Input**: Text + Group reference (e.g., "working families")
2. **Hypothesis Generation**: Creates two competing hypotheses:
   - "The text contains a policy directed towards working families."
   - "The text does not contain a policy directed towards working families."
3. **NLI Classification**: Model determines which hypothesis is most supported by the text
4. **Output**: Returns 'policy' or 'no policy' label with confidence score

### Policy vs Non-Policy Examples

**Policy Examples (would return 'policy'):**
- "We will provide tax credits to working families"
- "Small businesses will receive regulatory relief programs"
- "Students will get free tuition at public universities"

**Non-Policy Examples (would return 'no policy'):**
- "Working families are important to our society"
- "Students deserve better opportunities"

### Functions

#### `detect_policy(texts, groups, model_name="rwillh11/mdeberta_NLI_policy_noContext", batch_size=32, device=None)`

Detects policies towards groups from lists of texts and groups.

**Parameters:**
- `texts` (list): List of text strings to analyze
- `groups` (list): List of group references corresponding to each text
- `model_name` (str): Hugging Face NLI model name (default: "rwillh11/mdeberta_NLI_policy_noContext")
- `batch_size` (int): Number of examples to process at once (default: 32)
- `device` (str, optional): Device ('cuda', 'mps', 'cpu') - auto-detected if not specified

**Returns:**
- `list`: List of dictionaries with policy results. Each dict contains:
  - `'policy'`: Policy hypothesis ("The text contains/does not contain a policy directed towards [group].")
  - `'confidence'`: Confidence score (float between 0.0 and 1.0)

#### `process_policy_csv(input_file, text_column="text", group_column="Exact.Group.Text", output_file=None, model_name="rwillh11/mdeberta_NLI_policy_noContext", batch_size=32, device=None, clean_labels=False, quality_control=False)`

Processes CSV files to detect policies towards groups.

**Parameters:**
- `input_file` (str): Path to input CSV file
- `text_column` (str): Column containing text to analyze (default: "text")
- `group_column` (str): Column containing group references (default: "Exact.Group.Text")
- `output_file` (str, optional): Full file path to save results (e.g., `"results.csv"` or `"/path/to/output.csv"`). Must include the filename — a directory path alone will fail.
- `model_name` (str): Hugging Face NLI model name
- `batch_size` (int): Batch size for processing (default: 32)
- `device` (str, optional): Device to use for computation
- `clean_labels` (bool): Whether to extract clean policy labels (default: False)
- `quality_control` (bool): Whether to run quality control checks (default: False)

**Returns:**
- `pd.DataFrame`: Original DataFrame with added columns:
  - `"Policy"`: Policy labels (full hypothesis text)
  - `"Policy_Confidence"`: Confidence scores
  - `"Policy_Clean"`: Clean labels ('policy', 'no policy') if clean_labels=True

### Input Requirements

**CSV File Format:**
- Must contain a text column (default: "text") with plain text content
- Must contain a group column (default: "Exact.Group.Text") with group references
- Group references cannot have missing/NaN values
- Additional columns preserved in output

**Data Format:**
- Plain text without special formatting requirements
- Group references as simple strings (e.g., "working families", "students")
- Typically uses output from Group Entity Extraction module

### Data Preparation

#### From Entity Extraction Output
```python
from groupappeals import process_csv, process_policy_csv

# Step 1: Extract entities (creates text + Exact.Group.Text columns)
token_results = process_csv("input.csv", text_column="text", id_column="text_id")

# Step 2: Filter for rows with detected groups (required for policy detection)
with_groups = token_results[token_results['Exact.Group.Text'].notna()]

# Save prepared data
with_groups.to_csv("prepared_for_policy.csv", index=False)
```

#### Manual Preparation
```python
import pandas as pd

# Create policy detection input manually
df = pd.DataFrame({
    'text_id': ['doc1.1', 'doc2.1', 'doc3.1'],
    'text': [
        'We will implement new childcare support for working families.',
        'Small business owners face challenges in our economy.',
        'Students will receive increased funding for education programs.'
    ],
    'Exact.Group.Text': ['working families', 'small business owners', 'students']
})
df.to_csv('policy_input.csv', index=False)
```

### Output Format

The module returns a pandas DataFrame with original columns plus:

| Column | Description | Example |
|--------|-------------|---------|
| `Policy` | Full NLI hypothesis | "The text contains a policy directed towards working families." |
| `Policy_Confidence` | Confidence score (0-1) | 0.923 |
| `Policy_Clean` | Clean label (if enabled) | "policy" |

**Clean Labels (when clean_labels=True):**
- `"policy"` - Text contains concrete policy proposals directed towards the group
- `"no policy"` - Text does not contain policy proposals towards the group

### Standalone vs Pipeline Usage

#### Standalone Usage
Use this module independently when you already have group references:

```python
from groupappeals import detect_policy

texts = ["We will provide tax credits to working families.", "Students deserve better opportunities."]
groups = ["working families", "students"]

results = detect_policy(texts, groups)
for result in results:
    print(f"Policy: {result['policy']}")
    print(f"Confidence: {result['confidence']:.3f}")
```

#### Pipeline Integration
Integrate with entity extraction for complete analysis:

```python
from groupappeals import process_csv, process_policy_csv

# Step 1: Extract entities
token_results = process_csv("text_data.csv")

# Step 2: Detect policies on texts with groups
entities_with_groups = token_results[token_results['Exact.Group.Text'].notna()]
policy_results = process_policy_csv(
    input_file="temp_entities.csv",  # or pass DataFrame directly
    text_column="text",
    group_column="Exact.Group.Text",
    clean_labels=True
)
```

#### Full Pipeline Usage
For complete analysis including policy detection:

```python
from groupappeals import run_full_pipeline

# Complete analysis including policy detection
results = run_full_pipeline(
    input_file="raw_data.csv",
    output_file="complete_analysis.csv",
    clean_labels=True  # Include clean policy labels
)
```

### Error Handling

- **Missing Groups**: Raises ValueError if any group references are missing/NaN
- **Model Loading**: Clear error messages for model loading failures
- **Processing Errors**: Individual text failures result in "unknown" policy, processing continues
- **File Handling**: Proper error handling for CSV reading/writing operations

---

## Meaningful Groups Classification Module

### Overview

The Meaningful Groups Classification module categorizes extracted group references into meaningful semantic categories using a multi-label classification model. This module can be used as a standalone tool for group categorization or integrated with other GroupAppeals components for comprehensive text analysis.

### How It Works

The meaningful groups classification uses **multi-label text classification**:

1. **Input**: Group reference text (e.g., "working families", "small businesses")
2. **Model Processing**: Transformer model processes text and outputs probability scores for each category
3. **Thresholding**: Labels with scores above the threshold are accepted as predictions
4. **Output**: Returns list of category labels for each group reference

### Available Categories

The model classifies groups into semantic categories including:

- **Demographic Groups**: Age, gender-based groups ("young people", "women", "seniors")
- **Economic Groups**: Income, occupation-based groups ("working families", "middle class", "small businesses")
- **Geographic Groups**: Regional groups ("rural communities", "urban residents")
- **Political Groups**: Political affiliations ("voters", "conservatives", "activists")
- **Social Groups**: Cultural, religious groups ("families", "immigrants", "veterans")
- **Professional Groups**: Occupation-specific ("teachers", "healthcare workers", "farmers")

## The full codebook is available here: https://osf.io/r2dqg

### Functions

#### `classify_groups(texts, model_repo="rwillh11/mdeberta_groups_2.0", score_threshold=0.5, batch_size=32, device=None)`

Classifies group references into meaningful categories.

**Parameters:**
- `texts` (list): List of group reference texts to classify
- `model_repo` (str): Hugging Face model repository (default: "rwillh11/mdeberta_groups_2.0")
- `score_threshold` (float): Threshold for accepting predictions (0.0-1.0, default: 0.5)
- `batch_size` (int): Batch size for processing (default: 32)
- `device` (str, optional): Device ('cuda', 'mps', 'cpu') - auto-detected if not specified

**Returns:**
- `list`: List of lists containing predicted category labels for each group

#### `process_groups_csv(input_file, group_column="Exact.Group.Text", output_file=None, model_repo="rwillh11/mdeberta_groups_2.0", score_threshold=0.5, device=None, split_groups=False)`

Processes CSV files to classify group references into categories.

**Parameters:**
- `input_file` (str): Path to input CSV file
- `group_column` (str): Column containing group references (default: "Exact.Group.Text")
- `output_file` (str, optional): Full file path to save results (e.g., `"results.csv"` or `"/path/to/output.csv"`). Must include the filename — a directory path alone will fail.
- `model_repo` (str): Hugging Face model repository
- `score_threshold` (float): Threshold for accepting predictions (default: 0.5)
- `device` (str, optional): Device to use for computation
- `split_groups` (bool): Whether to split categories into separate columns (default: False)

**Returns:**
- `pd.DataFrame`: Original DataFrame with added "Meaningful Group" column (+ Group1, Group2, etc. if split_groups=True)

### Input Requirements

**CSV File Format:**
- Must contain a group column (default: "Exact.Group.Text") with group references
- NaN/empty values in group column are handled automatically
- Additional columns preserved in output

**Data Format:**
- Group references as simple strings (e.g., "working families", "students")
- Typically uses output from Group Entity Extraction module

### Data Preparation

#### From Entity Extraction Output
```python
from groupappeals import process_csv, process_groups_csv

# Step 1: Extract entities (creates Exact.Group.Text column)
token_results = process_csv("input.csv", text_column="text", id_column="text_id")

# Step 2: Classify groups (handles NaN values automatically)
classification_results = process_groups_csv(
    input_file="temp_entities.csv",  # or pass DataFrame directly
    group_column="Exact.Group.Text",
    score_threshold=0.5
)
```

#### Manual Preparation
```python
import pandas as pd

# Create group classification input manually
df = pd.DataFrame({
    'text_id': ['doc1.1', 'doc2.1', 'doc3.1'],
    'Exact.Group.Text': ['working families', 'small business owners', 'students']
})
df.to_csv('groups_input.csv', index=False)
```

### Output Format

The module returns a pandas DataFrame with original columns plus:

| Column | Description | Example |
|--------|-------------|---------|
| `Meaningful Group` | List of category labels | ['Economic Groups', 'Social Groups'] |
| `Group1` | First category (if split_groups=True) | "Economic Groups" |
| `Group2` | Second category (if split_groups=True) | "Social Groups" |
| `Group3` | Third category (if split_groups=True) | None |

**Category Assignment:**
- Groups can receive multiple category labels when appropriate
- Empty list `[]` for groups below confidence threshold
- Categories sorted by confidence score

### Threshold Behavior

```python
from groupappeals import classify_groups

groups = ["working families", "students"]

# Lower threshold = more categories assigned
results_low = classify_groups(groups, score_threshold=0.3)
# Higher threshold = fewer, more confident categories  
results_high = classify_groups(groups, score_threshold=0.7)

print(f"Threshold 0.3: {results_low}")
print(f"Threshold 0.7: {results_high}")
```

### Standalone vs Pipeline Usage

#### Standalone Usage
Use this module independently when you already have group references:

```python
from groupappeals import classify_groups

group_texts = ["working families", "small business owners", "students"]

results = classify_groups(group_texts, score_threshold=0.5)
for group, categories in zip(group_texts, results):
    print(f"{group}: {categories}")
```

#### Pipeline Integration
Integrate with entity extraction for complete analysis:

```python
from groupappeals import process_csv, process_groups_csv

# Step 1: Extract entities
token_results = process_csv("text_data.csv")

# Step 2: Classify groups (works on all rows, handles NaN automatically)
classification_results = process_groups_csv(
    input_file="temp_entities.csv",
    group_column="Exact.Group.Text",
    split_groups=True  # Create separate Group1, Group2 columns
)
```

#### Full Pipeline Usage
For complete analysis including group classification:

```python
from groupappeals import run_full_pipeline

# Complete analysis including group classification
results = run_full_pipeline(
    input_file="raw_data.csv",
    output_file="complete_analysis.csv",
    split_groups=True  # Include separate group category columns
)
```

### Error Handling

- **NaN/Empty Values**: Automatically filters and handles invalid group references
- **Model Loading**: Clear error messages for model loading failures
- **Processing Errors**: Individual group failures don't stop batch processing
- **File Handling**: Proper error handling for CSV reading/writing operations
- **Threshold Validation**: Ensures threshold is between 0.0 and 1.0

---

## Complete Pipeline Examples

### Basic Full Pipeline Usage

```python
from groupappeals import run_full_pipeline

# Simplest usage - analyze text with all default settings
results = run_full_pipeline(
    input_file="data.csv",
    output_file="results.csv"
)

print(f"Analyzed {len(results)} texts")
print(f"Found {results['Exact.Group.Text'].notna().sum()} group references")
```

### Complete Analysis with Composite IDs

```python
from groupappeals import run_full_pipeline

# Comprehensive analysis for political text data
results = run_full_pipeline(
    input_file="political_manifestos.csv",
    output_file="complete_analysis.csv",
    text_column="text",                                    # Column with raw text
    create_composite_id=["party", "election_year", "sentence_id"],  # Create meaningful IDs
    clean_labels=True,                                     # Get clean stance/policy labels
    split_groups=True,                                     # Separate group categories into columns
    batch_size=32,                                         # Process in batches of 32
    device="cuda",                                         # Use GPU acceleration
    save_intermediate_outputs=True                         # Save results from each step
)

# Display results summary
print(f"✅ Complete analysis finished!")
print(f"📊 Processed {len(results)} text segments")
print(f"🎯 Found groups in {results['Exact.Group.Text'].notna().sum()} texts")

# Show stance distribution
if 'Stance_Clean' in results.columns:
    print(f"📈 Stance distribution:")
    for stance, count in results['Stance_Clean'].value_counts().items():
        print(f"   {stance}: {count}")

# Show policy distribution  
if 'Policy_Clean' in results.columns:
    print(f"📋 Policy distribution:")
    for policy, count in results['Policy_Clean'].value_counts().items():
        print(f"   {policy}: {count}")
```

### Custom Model Configuration

```python
from groupappeals import run_full_pipeline

# Use custom models for each analysis step
custom_models = {
    "extraction": "your-org/custom-token-classifier",
    "stance": "your-org/custom-stance-model",
    "policy": "your-org/custom-policy-model", 
    "classification": "your-org/custom-group-classifier"
}

results = run_full_pipeline(
    input_file="data.csv",
    output_file="custom_analysis.csv",
    models=custom_models,
    create_composite_id=["speaker", "date", "paragraph"],
    clean_labels=True,
    split_groups=True
)
```

### Processing Large Datasets

```python
from groupappeals import run_full_pipeline
import pandas as pd

# Process large dataset with intermediate outputs for monitoring
results = run_full_pipeline(
    input_file="large_dataset.csv",
    output_file="large_results.csv",
    text_column="content",
    create_composite_id=["source", "year", "doc_id"],
    batch_size=16,                              # Smaller batches for memory management
    device="cuda",                              # Use GPU for speed
    save_intermediate_outputs=True,             # Monitor progress at each step
    intermediate_output_dir="./intermediate/",  # Organize intermediate files
    clean_labels=True,
    split_groups=True
)

print(f"Processing complete: {len(results)} texts analyzed")
```

### Step-by-Step Manual Pipeline

For maximum control over each analysis step, you can chain the modules manually. This approach assumes you've already extracted group entities (see [Group Entity Extraction Module](#group-entity-extraction-module) for extraction examples).

**Prerequisites:** You should have a CSV file with extracted entities containing at minimum:

- `text_id`: Unique identifiers
- `text`: Original text content
- `Exact.Group.Text`: Extracted group references

```python
from groupappeals import process_stance_csv, process_policy_csv, process_groups_csv
import pandas as pd

# Load your extracted entities (from previous extraction step)
entities_df = pd.read_csv("entities.csv")

# Step 1: Filter for texts with detected groups
print("🔍 Step 1: Filtering texts with group references...")
with_groups = entities_df[entities_df['Exact.Group.Text'].notna()]
print(f"   Found {len(with_groups)} texts with group references")

if len(with_groups) > 0:
    with_groups.to_csv("with_groups.csv", index=False)

    # Step 2: Detect stance toward groups
    print("😊 Step 2: Detecting stance...")
    stance_results = process_stance_csv(
        input_file="with_groups.csv",
        text_column="text",
        group_column="Exact.Group.Text",
        output_file="stance_results.csv",
        clean_labels=True
    )
    print(f"   Processed {len(stance_results)} stance predictions")

    # Step 3: Detect policies directed at groups
    print("📋 Step 3: Detecting policies...")
    policy_results = process_policy_csv(
        input_file="stance_results.csv",
        text_column="text",
        group_column="Exact.Group.Text",
        output_file="policy_results.csv",
        clean_labels=True
    )
    print(f"   Processed {len(policy_results)} policy predictions")

    # Step 4: Classify groups into meaningful categories
    print("🏷️ Step 4: Classifying groups...")
    classification_results = process_groups_csv(
        input_file="policy_results.csv",
        group_column="Exact.Group.Text",
        output_file="final_results.csv",
        split_groups=True
    )
    print(f"   Classified {len(classification_results)} group references")

    print(f"✅ Manual pipeline complete: {len(classification_results)} results")
else:
    print("⚠️ No group references found in the data")
```

**Note:** For a complete end-to-end example including data preparation and entity extraction, see the [Group Entity Extraction Module](#group-entity-extraction-module) section above.

### Input Data Preparation Examples

#### Basic CSV Structure
```python
import pandas as pd

# Minimal required structure
df = pd.DataFrame({
    'text_id': ['doc1', 'doc2', 'doc3'],
    'text': [
        'We support working families in our communities.',
        'Small businesses deserve better tax policies.',
        'Students need affordable education opportunities.'
    ]
})
df.to_csv('simple_input.csv', index=False)

# Run full pipeline
results = run_full_pipeline('simple_input.csv', 'simple_results.csv')
```

#### Political Text with Metadata
```python
import pandas as pd

# Rich metadata for political analysis
df = pd.DataFrame({
    'party': ['Party A', 'Party B', 'Party A'],
    'election_year': [2020, 2020, 2024],
    'sentence_id': [1, 1, 1],
    'text': [
        'We will support working families with new childcare policies.',
        'Small business owners need tax relief from our government.',
        'Students deserve free tuition at public universities.'
    ]
})
df.to_csv('political_input.csv', index=False)

# Process with composite ID creation
results = run_full_pipeline(
    input_file='political_input.csv',
    output_file='political_results.csv',
    create_composite_id=['party', 'election_year', 'sentence_id'],
    clean_labels=True,
    split_groups=True
)
```

### Output Structure Examples

After running the full pipeline, you'll get a comprehensive DataFrame with:

```python
# Example of what the output contains
print(results.columns.tolist())
# ['text_id', 'text', 'Exact.Group.Text', 'Average Score',
#  'Start', 'End', 'Stance', 'Stance_Confidence', 'Stance_Clean',
#  'Policy', 'Policy_Confidence', 'Policy_Clean', 'Meaningful Group',
#  'Group1', 'Group2', 'Group3']

# Sample row
sample = results.iloc[0]
print(f"Text ID: {sample['text_id']}")
print(f"Original Text: {sample['text']}")
print(f"Group Found: {sample['Exact.Group.Text']}")
print(f"Stance: {sample['Stance_Clean']} (confidence: {sample['Stance_Confidence']:.3f})")
print(f"Policy: {sample['Policy_Clean']} (confidence: {sample['Policy_Confidence']:.3f})")
print(f"Group Categories: {sample['Meaningful Group']}")
```

---

## Command Line Interface

The GroupAppeals CLI provides individual commands for each analysis step and a complete pipeline command for end-to-end processing.

### Available Commands

```bash
groupappeals extract    # Extract group references from text
groupappeals stance     # Detect stance towards groups  
groupappeals policy     # Detect policies directed towards groups
groupappeals classify   # Classify groups into meaningful categories
groupappeals pipeline   # Run complete analysis pipeline
```

### Basic Usage

#### Individual Commands

```bash
# Step 1: Extract group references
groupappeals extract --input data.csv --output groups.csv

# Step 2: Detect stance towards groups
groupappeals stance --input groups.csv --output stance_results.csv

# Step 3: Detect policies directed towards groups  
groupappeals policy --input stance_results.csv --output policy_results.csv

# Step 4: Classify groups into categories
groupappeals classify --input policy_results.csv --output final_results.csv
```

#### Complete Pipeline (Recommended)

```bash
# Run full analysis pipeline
groupappeals pipeline --input data.csv --output complete_results.csv
```

### Pipeline Command Options

#### Basic Pipeline Usage

```bash
# Simplest usage
groupappeals pipeline --input data.csv --output results.csv

# With composite ID creation
groupappeals pipeline --input political_data.csv --output results.csv \
  --create-composite-id party,election_year,sentence_id
```

#### Advanced Pipeline Options

```bash
# Complete pipeline with all features
groupappeals pipeline --input manifestos.csv --output analysis.csv \
  --text-column content \
  --create-composite-id party,year,sentence_number \
  --clean-labels \
  --split-groups \
  --batch-size 32 \
  --device cuda

# Process large datasets with smaller batches
groupappeals pipeline --input large_dataset.csv --output results.csv \
  --batch-size 16 \
  --device cuda \
  --clean-labels
```

### Individual Command Options

#### Extract Command

```bash
# Basic extraction
groupappeals extract --input data.csv --output groups.csv

# Custom column names
groupappeals extract --input data.csv --output groups.csv \
  --text-column content --id-column document_id

# With GPU acceleration
groupappeals extract --input data.csv --output groups.csv --device cuda
```

#### Stance Command

```bash
# Basic stance detection
groupappeals stance --input groups.csv --output stance.csv

# With clean labels and quality control
groupappeals stance --input groups.csv --output stance.csv \
  --clean-labels --quality-control

# Custom batch size and device
groupappeals stance --input groups.csv --output stance.csv \
  --batch-size 16 --device mps
```

#### Policy Command

```bash
# Basic policy detection
groupappeals policy --input stance.csv --output policies.csv

# With clean labels
groupappeals policy --input stance.csv --output policies.csv \
  --clean-labels --quality-control

# Custom processing parameters
groupappeals policy --input stance.csv --output policies.csv \
  --batch-size 32 --device cuda
```

#### Classify Command

```bash
# Basic group classification
groupappeals classify --input policies.csv --output classified.csv

# With custom threshold and column splitting
groupappeals classify --input policies.csv --output classified.csv \
  --threshold 0.7 --split-groups

# Custom device selection
groupappeals classify --input policies.csv --output classified.csv \
  --device cuda --split-groups
```

### Device Selection

```bash
# Automatic device detection (default)
groupappeals pipeline --input data.csv --output results.csv

# Force specific device
groupappeals pipeline --input data.csv --output results.csv --device cuda
groupappeals pipeline --input data.csv --output results.csv --device mps
groupappeals pipeline --input data.csv --output results.csv --device cpu
```

### Column Configuration

```bash
# Custom column names
groupappeals extract --input data.csv --output groups.csv \
  --text-column article_text --id-column article_id

groupappeals stance --input groups.csv --output stance.csv \
  --text-column article_text --group-column "Exact.Group.Text"

# Pipeline with custom text column
groupappeals pipeline --input data.csv --output results.csv \
  --text-column article_content
```

### Performance Options

```bash
# Optimize batch size for your hardware
groupappeals pipeline --input data.csv --output results.csv --batch-size 64

# Smaller batches for limited memory
groupappeals pipeline --input data.csv --output results.csv --batch-size 8

# Individual commands with custom batch sizes
groupappeals stance --input groups.csv --output stance.csv --batch-size 16
groupappeals policy --input stance.csv --output policy.csv --batch-size 16
```

### Output Options

```bash
# Get clean, simplified labels
groupappeals pipeline --input data.csv --output results.csv --clean-labels

# Split group categories into separate columns
groupappeals pipeline --input data.csv --output results.csv --split-groups

# Both clean labels and split groups
groupappeals pipeline --input data.csv --output results.csv \
  --clean-labels --split-groups
```

### Quality Control

```bash
# Enable quality control for stance detection
groupappeals stance --input groups.csv --output stance.csv --quality-control

# Enable quality control for policy detection  
groupappeals policy --input stance.csv --output policies.csv --quality-control
```

### Input File Requirements

Your CSV file needs:
- A text column (default: "text") with plain text content
- An ID column (default: "text_id") with unique identifiers

For pipeline with composite IDs:
- Additional columns specified in `--create-composite-id`

### Example Workflows

#### Simple Analysis
```bash
# Quick analysis of political text
groupappeals pipeline --input speeches.csv --output analysis.csv \
  --create-composite-id speaker,date,paragraph \
  --clean-labels
```

#### Research Workflow
```bash
# Comprehensive analysis with all features
groupappeals pipeline --input manifestos.csv --output full_analysis.csv \
  --create-composite-id party,election_year,sentence_id \
  --clean-labels \
  --split-groups \
  --device cuda
```

#### Step-by-Step Processing
```bash
# Manual control over each step
groupappeals extract --input data.csv --output step1_groups.csv
groupappeals stance --input step1_groups.csv --output step2_stance.csv --clean-labels
groupappeals policy --input step2_stance.csv --output step3_policy.csv --clean-labels  
groupappeals classify --input step3_policy.csv --output final_results.csv --split-groups
```

### Command Help

```bash
# Get general help
groupappeals --help

# Get help for specific commands
groupappeals pipeline --help
groupappeals extract --help
groupappeals stance --help
groupappeals policy --help
groupappeals classify --help
```

---

## Performance Optimization

### Hardware Recommendations

| Dataset Size | CPU | RAM | GPU | Processing Time* |
|--------------|-----|-----|-----|------------------|
| Small (<1K texts) | Any modern CPU | 4GB | Optional | 5-15 minutes |
| Medium (1K-10K) | Multi-core CPU | 8GB | Recommended | 30-120 minutes |
| Large (10K-100K) | High-end CPU | 16GB+ | Required | 2-12 hours |
| Very Large (100K+) | Server CPU | 32GB+ | Multiple GPUs | 12+ hours |

*Approximate times for complete pipeline

### Apple Silicon Support

GroupAppeals fully supports Apple Silicon Macs (M1, M2, M3) with GPU acceleration:

- **Automatic Detection**: MPS (Metal Performance Shaders) is automatically detected and used
- **Performance**: 3-5x speedup compared to CPU processing on Apple Silicon
- **Energy Efficient**: Lower power consumption than CPU processing
- **Requirements**: macOS 12.3+ and PyTorch 1.12+

```bash
# Automatic Apple Silicon GPU usage (recommended)
groupappeals pipeline --input data.csv --output results.csv

# Explicit MPS usage
groupappeals pipeline --input data.csv --output results.csv --device mps
```

```python
# Python API - automatic device detection
from groupappeals.stancedetection import detect_stance_with_context
results = detect_stance_with_context(texts, groups)  # Uses MPS automatically

# Explicit MPS usage
results = detect_stance_with_context(texts, groups, device="mps")
```

### Performance Tips

1. **Enable GPU processing** for 5-10x speed improvement on large datasets
   - **NVIDIA GPUs**: Use `--device cuda` or `device="cuda"`
   - **Apple Silicon**: Use `--device mps` or `device="mps"` (or let auto-detect)
   - **CPU**: Use `--device cpu` or `device="cpu"`
2. **Optimize batch sizes** based on available memory
3. **Process in chunks** for very large datasets
4. **Use SSD storage** for faster file I/O

### Memory Management

```python
# For large datasets, process in chunks
import pandas as pd
import os
from groupappeals import process_csv

def process_large_file(input_file, chunk_size=1000):
    """Process large files in chunks to manage memory."""

    # Read file info
    total_rows = sum(1 for line in open(input_file)) - 1  # subtract header
    print(f"Processing {total_rows} rows in chunks of {chunk_size}")

    results = []
    for chunk_num, chunk_df in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        print(f"Processing chunk {chunk_num + 1}...")

        # Save chunk temporarily
        chunk_file = f"temp_chunk_{chunk_num}.csv"
        chunk_df.to_csv(chunk_file, index=False)

        # Process chunk
        chunk_results = process_csv(
            input_file=chunk_file,
            text_column="text",
            id_column="text_id",
            output_file=None
        )
        results.append(chunk_results)
        
        # Cleanup
        os.remove(chunk_file)
    
    # Combine results
    final_results = pd.concat(results, ignore_index=True)
    return final_results
```

---

## Error Handling and Troubleshooting

GroupAppeals includes comprehensive error handling to help you identify and resolve issues quickly. This section covers common problems and their solutions.

### Common Installation Issues

#### Package Not Found

**Problem**: `ModuleNotFoundError: No module named 'groupappeals'`
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'groupappeals'
```

**Solution**:
```bash
# Install the package
pip install groupappeals

# If using conda environment
pip install groupappeals  # Use pip even in conda environments

# If installing from source
pip install -e .
```

#### Dependency Issues

**Problem**: Version conflicts with transformers or torch
```
ERROR: pip's dependency resolver does not currently take into account all the packages...
```

**Solution**:
```bash
# Create fresh environment
conda create -n groupappeals python=3.9
conda activate groupappeals
pip install groupappeals

# Or update existing packages
pip install --upgrade torch transformers pandas
```

### Hardware and Performance Issues

#### GPU Memory Issues

**Problem**: `RuntimeError: CUDA out of memory`
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:
```python
from groupappeals import run_full_pipeline

# Solution 1: Reduce batch size
results = run_full_pipeline(
    input_file="data.csv",
    output_file="results.csv",
    batch_size=8,  # Reduced from default 32
    device="cuda"
)

# Solution 2: Switch to CPU
results = run_full_pipeline(
    input_file="data.csv",
    output_file="results.csv",
    device="cpu"
)

# Solution 3: Use Apple Silicon MPS (Mac M1/M2/M3/M4)
results = run_full_pipeline(
    input_file="data.csv",
    output_file="results.csv",
    device="mps"
)
```

#### Missing Required Columns

**Problem**: `KeyError: Text column 'text' not found in the input file`
```
KeyError: Text column 'text' not found in the input file
```

**Solution**:
```python
import pandas as pd

# Check your column names
df = pd.read_csv("data.csv")
print("Available columns:", df.columns.tolist())

# Use custom column names
from groupappeals import run_full_pipeline

results = run_full_pipeline(
    input_file="data.csv",
    output_file="results.csv",
    text_column="your_text_column_name",  # Adjust to your column name
    id_column="your_id_column_name"       # Adjust to your ID column name
)
```

#### Empty or Invalid Data

**Problem**: `ValueError: DataFrame is empty`
```
ValueError: DataFrame is empty
```

**Solution**:
```python
import pandas as pd

# Validate your data
def validate_input_data(file_path):
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    if df.empty:
        print("ERROR: DataFrame is empty")
        return False
    
    # Check for required columns
    if 'text' not in df.columns:
        print("WARNING: 'text' column not found")
    
    # Check for null/empty values
    if 'text' in df.columns:
        null_count = df['text'].isnull().sum()
        empty_count = (df['text'] == '').sum()
        print(f"Null texts: {null_count}")
        print(f"Empty texts: {empty_count}")
        
        if null_count == len(df):
            print("ERROR: All text values are null")
            return False
    
    return True

# Validate before processing
if validate_input_data("data.csv"):
    results = run_full_pipeline("data.csv", "results.csv")
```

#### ID-Related Issues

```python
# Problem: Simple numeric IDs lose context
# Your results show: text_id=1,2,3... with no way to trace back to original party/date

# Solution: Use composite IDs from the start
from groupappeals.pre_and_post_processing import create_composite_id
import pandas as pd

df = pd.read_csv("data.csv")
df['text_id'] = create_composite_id(
    df,
    party_col="party",
    date_col="date",
    sentence_col="sentence"
)
# Result: text_id="PartyName_2019_1" - traceable and meaningful

# Problem: Inconsistent ID formats across pipeline steps
# Error: Can't match results back to original sources

# Solution: Plan your ID structure from the beginning
# 1. Use the same composite ID structure throughout
# 2. Document your ID format for team members
# 3. Test with small samples first
```

#### Model Loading Issues

```python
# Error: Model not found
RuntimeError: Failed to load model 'rwillh11/base-mdbertav3-token-classification-groups-bilingual'

# Solutions:
# 1. Check internet connection
# 2. Verify model name spelling
# 3. Check Hugging Face Hub status
# 4. Clear transformers cache
import shutil
import os
from pathlib import Path

# Clear Hugging Face cache
cache_dir = Path.home() / ".cache" / "huggingface"
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print("Cleared Hugging Face cache")

# Alternative: Clear specific model cache
# import torch
# torch.hub.set_dir('/tmp/torch_cache')  # Use temporary directory
```

#### Memory Issues

```python
# Error: CUDA out of memory
RuntimeError: CUDA out of memory

# Solutions:
# 1. Reduce batch size
from groupappeals import process_stance_csv_with_context
process_stance_csv_with_context(
    input_file=input_file,
    model_name="rwillh11/mdeberta_NLI_bilingual_2.0",
    batch_size=8
)
# 2. Use CPU instead
process_stance_csv_with_context(
    input_file=input_file,
    model_name="rwillh11/mdeberta_NLI_bilingual_2.0",
    device="cpu"
)
# 3. Process in smaller chunks using text formatting
```

#### Column Name Issues

```python
# Error: Column not found
KeyError: "Text column 'speech_content' not found in the input file"

# Solutions:
# 1. Check column names in your CSV
df.columns.tolist()
# 2. Specify correct column names
process_csv(
    input_file=input_file,
    text_column="actual_column_name",
    id_column="text_id"
)
```

### Debugging Tips

```python
# Enable debug mode for detailed output
import logging
logging.basicConfig(level=logging.DEBUG)

# Test with small sample first
import pandas as pd
from groupappeals import process_csv

# Assume df is your main DataFrame
sample_df = df.head(10)
sample_df.to_csv("test_sample.csv", index=False)

# Test extraction on sample
token_results = process_csv(
    input_file="test_sample.csv",
    text_column="text",
    id_column="text_id",
    output_file="test_results.csv"
)

# Check results
print(f"Processed {len(token_results)} rows")
print(f"Found {len(token_results[token_results['Exact.Group.Text'].notna()])} group mentions")
print("\nSample extractions:")
print(token_results[token_results['Exact.Group.Text'].notna()][['text_id', 'Exact.Group.Text', 'Average Score']].head())
```

### Model Loading Issues

#### Network/Download Issues

**Problem**: `RuntimeError: Model loading failed` or connection timeouts
```
RuntimeError: Failed to load model 'rwillh11/mdeberta_NLI_stance_NoContext'
```

**Solutions**:
```python
# Try with explicit device specification
from groupappeals import extract_entities

try:
    results = extract_entities(texts, device="cpu")
except Exception as e:
    print(f"Model loading failed: {e}")
    print("Check internet connection and try again")

# For offline usage, pre-download models
from transformers import AutoModel, AutoTokenizer

# Pre-download main models
model_names = [
    "rwillh11/mdberta-token-bilingual-noContext_Enhanced",
    "rwillh11/mdeberta_NLI_stance_NoContext", 
    "rwillh11/mdeberta_NLI_policy_noContext",
    "rwillh11/mdeberta_groups_2.0"
]

for model_name in model_names:
    try:
        AutoModel.from_pretrained(model_name)
        AutoTokenizer.from_pretrained(model_name)
        print(f"Downloaded: {model_name}")
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")
```

### Processing Issues

#### No Groups Found

**Problem**: Pipeline completes but no group references are detected
```python
# Check if groups were actually found
results = run_full_pipeline("data.csv", "results.csv")
groups_found = results['Exact.Group.Text'].notna().sum()
print(f"Groups found: {groups_found}")

if groups_found == 0:
    print("No groups detected. Check your text content:")
    # Sample some texts to see what the model is processing
    sample_texts = pd.read_csv("data.csv")['text'].head(5).tolist()
    for i, text in enumerate(sample_texts):
        print(f"Text {i+1}: {text[:100]}...")
```

#### Missing Values in Stance/Policy Detection

**Problem**: `ValueError: Cannot detect stance: group references contain missing values`
```
ValueError: Cannot detect stance: group references contain missing values
```

**Solution**:
```python
# The full pipeline handles this automatically, but for manual steps:
from groupappeals import process_csv, process_stance_csv
import pandas as pd

# Step 1: Extract entities
token_results = process_csv("data.csv")

# Step 2: Filter for texts WITH groups before stance detection
with_groups = token_results[token_results['Exact.Group.Text'].notna()]
print(f"Texts with groups: {len(with_groups)}")

if len(with_groups) > 0:
    # Step 3: Process stance only on texts that have groups
    with_groups.to_csv("temp_with_groups.csv", index=False)
    stance_results = process_stance_csv(
        input_file="temp_with_groups.csv",
        text_column="text",
        group_column="Exact.Group.Text"
    )
else:
    print("No group references found in any texts")
```

### Debugging and Diagnostics

#### Enable Detailed Output

```python
# Test with a small sample first
import pandas as pd

df = pd.read_csv("large_dataset.csv")
sample = df.head(5)  # Test with just 5 rows
sample.to_csv("test_sample.csv", index=False)

# Process sample with detailed output
results = run_full_pipeline(
    input_file="test_sample.csv",
    output_file="test_results.csv",
    save_intermediate_outputs=True  # Save intermediate files for inspection
)

print("Processing completed successfully with sample data")
print(f"Results shape: {results.shape}")
print(f"Columns: {results.columns.tolist()}")
```

#### Performance Monitoring

```python
import time
import psutil

def monitor_pipeline_execution(input_file, output_file):
    """Monitor resource usage during pipeline execution."""
    
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()
    
    print(f"Starting pipeline execution...")
    print(f"Initial memory usage: {start_memory:.1f} MB")
    
    try:
        results = run_full_pipeline(
            input_file=input_file,
            output_file=output_file,
            batch_size=16  # Conservative batch size
        )
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"✅ Pipeline completed successfully")
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        print(f"Peak memory usage: {end_memory:.1f} MB")
        print(f"Processed {len(results)} rows")
        
        return results
        
    except Exception as e:
        end_time = time.time()
        print(f"❌ Pipeline failed after {end_time - start_time:.2f} seconds")
        print(f"Error: {str(e)}")
        raise

# Monitor your pipeline
results = monitor_pipeline_execution("data.csv", "results.csv")
```

### Getting Additional Help

If you encounter persistent issues:

1. **Validate your input data** using the validation functions above
2. **Test with small samples** to isolate the problem
3. **Check system resources** (memory, disk space, GPU memory)
4. **Try different device options** (CPU vs GPU vs MPS)
5. **Reduce batch sizes** for memory-constrained systems

```python
# Quick diagnostic function
def diagnose_system():
    """Run basic system diagnostics for GroupAppeals."""
    
    # Check package installation
    try:
        import groupappeals
        print(f"✅ GroupAppeals version: {groupappeals.__version__}")
    except ImportError:
        print("❌ GroupAppeals not installed")
        return
    
    # Check device availability
    from groupappeals.device_utilities import determine_compute_device
    device = determine_compute_device()
    print(f"✅ Optimal device: {device}")
    
    # Check system resources
    import psutil
    memory = psutil.virtual_memory()
    print(f"✅ Available memory: {memory.available / 1024 / 1024 / 1024:.1f} GB")
    
    # Test basic functionality
    try:
        from groupappeals import extract_entities
        test_result = extract_entities(["Test text with workers"], ["test1"])
        print("✅ Basic functionality test passed")
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")

# Run diagnostics
diagnose_system()
```

---

## Complete API Reference

This section provides a complete reference for all functions and classes in the GroupAppeals package.

### Device Utilities

#### `determine_compute_device()`
Automatically determines the optimal device for model inference.

**Returns:**
- `str`: Device string ('cuda', 'mps', or 'cpu')

#### `convert_device_to_pipeline_id(device)`
Converts device string to pipeline device ID format required by Hugging Face pipelines.

**Parameters:**
- `device` (str): Device string ('cuda', 'mps', or 'cpu')

**Returns:**
- `int` or `str`: Device ID for pipeline (0 for 'cuda', 'mps' for 'mps', -1 for 'cpu')

### Full Pipeline Function

#### `run_full_pipeline(input_file, output_file=None, text_column="text", id_column="text_id", group_columns=None, order_columns=None, models=None, batch_size=32, device=None, create_composite_id=None, composite_separator="_", clean_labels=True, split_groups=True, quality_control=False, save_intermediate_outputs=False, intermediate_output_dir=None)`

Run the complete GroupAppeals analysis pipeline from raw text to final results.

**Parameters:**
- `input_file` (str): Path to input CSV file with raw text
- `output_file` (str, optional): Path to save final results CSV
- `text_column` (str): Column containing raw text (default: "text")
- `id_column` (str): Column containing identifiers (default: "text_id")
- `group_columns` (list, optional): Columns to group by for processing
- `order_columns` (list, optional): Columns to sort by within each group
- `models` (dict, optional): Custom model names for each step
- `batch_size` (int): Batch size for processing (default: 32)
- `device` (str, optional): Device to use ('cuda', 'mps', or 'cpu') - auto-detected if None
- `create_composite_id` (list, optional): Columns to combine into composite ID
- `composite_separator` (str): Separator for composite ID (default: "_")
- `clean_labels` (bool): Extract clean stance/policy labels (default: True)
- `split_groups` (bool): Split meaningful groups into separate columns (default: True)
- `quality_control` (bool): Run quality control checks (default: False)
- `save_intermediate_outputs` (bool): Save intermediate results (default: False)
- `intermediate_output_dir` (str, optional): Directory for intermediate outputs

**Returns:**
- `pd.DataFrame`: Complete results with all analysis columns

**Raises:**
- `FileNotFoundError`: If input file doesn't exist
- `KeyError`: If specified columns don't exist
- `ValueError`: If input file is empty or invalid
- `RuntimeError`: For processing errors

### Group Entity Extraction Functions

#### `extract_entities(texts, ids=None, model_name="rwillh11/mdberta-token-bilingual-noContext_Enhanced", device=None)`

Extract social group entities from a list of texts.

**Parameters:**
- `texts` (list): List of text strings to analyze
- `ids` (list, optional): List of IDs for each text (auto-generated if not provided)
- `model_name` (str): Hugging Face model name (default: enhanced bilingual model)
- `device` (str, optional): Device ('cuda', 'mps', 'cpu') - auto-detected if not specified

**Returns:**
- `pd.DataFrame`: DataFrame with columns: text_id, text, Entity, Average Score, Start, End, Exact.Group.Text

**Raises:**
- `ValueError`: If inputs are invalid
- `RuntimeError`: If model loading or processing fails

#### `process_csv(input_file, text_column="text", id_column="text_id", output_file=None, device=None)`

Process a CSV file to extract entities from text.

**Parameters:**
- `input_file` (str): Path to input CSV file
- `text_column` (str): Column containing text to analyze (default: "text")
- `id_column` (str): Column containing identifiers (default: "text_id")
- `output_file` (str, optional): Full file path to save results (e.g., `"results.csv"` or `"/path/to/output.csv"`). Must include the filename — a directory path alone will fail.
- `device` (str, optional): Device to use for computation ('cuda', 'mps', or 'cpu') - auto-detected if not specified

**Returns:**
- `pd.DataFrame`: DataFrame with extracted entities

### Stance Detection Functions

#### `detect_stance(texts, groups, model_name="rwillh11/mdeberta_NLI_stance_NoContext", batch_size=32, device=None)`

Detect stance towards groups from lists of texts and groups.

**Parameters:**
- `texts` (list): List of text strings to analyze
- `groups` (list): List of group references corresponding to each text
- `model_name` (str): Hugging Face NLI model name (default: stance model)
- `batch_size` (int): Batch size for processing (default: 32)
- `device` (str, optional): Device ('cuda', 'mps', 'cpu') - auto-detected if not specified

**Returns:**
- `list`: List of dictionaries with stance results
  - `'stance'`: Full NLI hypothesis text
  - `'confidence'`: Confidence score (0.0-1.0)

**Raises:**
- `ValueError`: If inputs are invalid or groups contain missing values
- `RuntimeError`: If model loading or processing fails

#### `process_stance_csv(input_file, text_column="text", group_column="Exact.Group.Text", output_file=None, model_name="rwillh11/mdeberta_NLI_stance_NoContext", batch_size=32, device=None, clean_labels=False, quality_control=False)`

Process CSV files to detect stance towards groups.

**Parameters:**
- `input_file` (str): Path to input CSV file
- `text_column` (str): Column containing text (default: "text")
- `group_column` (str): Column containing group references (default: "Exact.Group.Text")
- `output_file` (str, optional): Full file path to save results (e.g., `"results.csv"` or `"/path/to/output.csv"`). Must include the filename — a directory path alone will fail.
- `model_name` (str): Hugging Face model name
- `batch_size` (int): Batch size (default: 32)
- `device` (str, optional): Device to use
- `clean_labels` (bool): Extract clean stance labels (default: False)
- `quality_control` (bool): Run quality control checks (default: False)

**Returns:**
- `pd.DataFrame`: Original DataFrame with added Stance and Stance_Confidence columns

### Policy Detection Functions

#### `detect_policy(texts, groups, model_name="rwillh11/mdeberta_NLI_policy_noContext", batch_size=32, device=None)`

Detect policies directed towards groups from lists of texts and groups.

**Parameters:**
- `texts` (list): List of text strings to analyze
- `groups` (list): List of group references corresponding to each text
- `model_name` (str): Hugging Face NLI model name (default: policy model)
- `batch_size` (int): Batch size for processing (default: 32)
- `device` (str, optional): Device ('cuda', 'mps', 'cpu') - auto-detected if not specified

**Returns:**
- `list`: List of dictionaries with policy results
  - `'policy'`: Full NLI hypothesis text
  - `'confidence'`: Confidence score (0.0-1.0)

**Raises:**
- `ValueError`: If inputs are invalid or groups contain missing values
- `RuntimeError`: If model loading or processing fails

#### `process_policy_csv(input_file, text_column="text", group_column="Exact.Group.Text", output_file=None, model_name="rwillh11/mdeberta_NLI_policy_noContext", batch_size=32, device=None, clean_labels=False, quality_control=False)`

Process CSV files to detect policies directed towards groups.

**Parameters:**
- `input_file` (str): Path to input CSV file
- `text_column` (str): Column containing text (default: "text")
- `group_column` (str): Column containing group references (default: "Exact.Group.Text")
- `output_file` (str, optional): Full file path to save results (e.g., `"results.csv"` or `"/path/to/output.csv"`). Must include the filename — a directory path alone will fail.
- `model_name` (str): Hugging Face model name
- `batch_size` (int): Batch size (default: 32)
- `device` (str, optional): Device to use
- `clean_labels` (bool): Extract clean policy labels (default: False)
- `quality_control` (bool): Run quality control checks (default: False)

**Returns:**
- `pd.DataFrame`: Original DataFrame with added Policy and Policy_Confidence columns

### Group Classification Functions

#### `classify_groups(texts, model_repo="rwillh11/mdeberta_groups_2.0", score_threshold=0.5, batch_size=32, device=None)`

Classify group references into meaningful semantic categories.

**Parameters:**
- `texts` (list): List of group reference texts to classify
- `model_repo` (str): Hugging Face model repository (default: groups model)
- `score_threshold` (float): Threshold for accepting predictions (0.0-1.0, default: 0.5)
- `batch_size` (int): Batch size for processing (default: 32)
- `device` (str, optional): Device ('cuda', 'mps', 'cpu') - auto-detected if not specified

**Returns:**
- `list`: List of lists containing predicted category labels for each group

**Raises:**
- `ValueError`: If inputs are invalid or threshold out of range
- `RuntimeError`: If model loading or processing fails

#### `process_groups_csv(input_file, group_column="Exact.Group.Text", output_file=None, model_repo="rwillh11/mdeberta_groups_2.0", score_threshold=0.5, device=None, split_groups=False)`

Process CSV files to classify group references into categories.

**Parameters:**
- `input_file` (str): Path to input CSV file
- `group_column` (str): Column containing group references (default: "Exact.Group.Text")
- `output_file` (str, optional): Full file path to save results (e.g., `"results.csv"` or `"/path/to/output.csv"`). Must include the filename — a directory path alone will fail.
- `model_repo` (str): Hugging Face model repository
- `score_threshold` (float): Threshold for accepting predictions (default: 0.5)
- `device` (str, optional): Device to use
- `split_groups` (bool): Split categories into separate columns (default: False)

**Returns:**
- `pd.DataFrame`: Original DataFrame with added Meaningful Group column (+ Group1, Group2, etc. if split_groups=True)

### Pre and Post-Processing Functions

#### `create_composite_id(df, party_col="party", date_col="date", sentence_col="sentence_id", separator="_")`

Create composite IDs from multiple columns for better traceability.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `party_col` (str): Column name for party/organization (default: "party")
- `date_col` (str): Column name for date/time identifier (default: "date")
  *Note: This is converted to string and does not require date formatting. Any identifier works.*
- `sentence_col` (str): Column name for sentence ID (default: "sentence_id")
- `separator` (str): Separator character (default: "_")

**Returns:**
- `pd.Series`: Series of composite IDs in format "party_date_sentenceID"

**Raises:**
- `KeyError`: If required columns don't exist
- `ValueError`: If DataFrame is empty

#### `extract_clean_stance_label(stance_text)`

Extract clean stance label from verbose model output.

**Parameters:**
- `stance_text` (str): Verbose stance output

**Returns:**
- `str`: Clean stance label ('positive', 'negative', 'neutral') or None

#### `extract_clean_policy_label(policy_text)`

Extract clean policy label from verbose model output.

**Parameters:**
- `policy_text` (str): Verbose policy output

**Returns:**
- `str`: Clean policy label ('policy', 'no policy') or None

#### `split_meaningful_groups_into_columns(df, meaningful_groups_column='Meaningful Group', max_groups=None)`

Split meaningful groups list into separate Group1, Group2, etc. columns.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing meaningful groups
- `meaningful_groups_column` (str): Column name containing group lists (default: 'Meaningful Group')
- `max_groups` (int, optional): Maximum columns to create (auto-determined if None)

**Returns:**
- `pd.DataFrame`: DataFrame with Group1, Group2, etc. columns added

## License and Citation

### License

This project is licensed under the MIT License.

### Citation

If you use GroupAppeals in your research, please cite:

```bibtex
@misc{groupappeals2025,
  title={GroupAppeals: A Python Package for Analyzing Social Group Appeals in Text},
  author={Dolinsky, Alona O. and Horne, Will and Huber, Lena Maria},
  year={2025},
  url={https://github.com/alonadoli/GroupAppeals}
}
```

### Research Papers

Please also cite the underlying research:

```bibtex
@misc{horne2025detecting,
  title={Detecting and Classifying Social Group Appeals using Language Models},
  author={Horne, Will and Dolinsky, Alona O. and Huber, Lena Maria},
  year={2025},
  url={https://osf.io/preprints/osf/fp2h3_v2}
}
```

---

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, Linux
- **Memory**: Minimum 4GB RAM, recommended 8GB+
- **Storage**: 2GB free space for models and cache
- **Network**: Internet connection for model downloads

### GPU Support

- **NVIDIA GPUs**: CUDA support available with compatible PyTorch installation
- **Apple Silicon**: MPS (Metal Performance Shaders) support for M1/M2/M3/M4 Macs
  - Requires macOS 12.3+ and PyTorch 1.12+
  - Automatic detection and optimization
- **CPU**: Always supported as fallback option

---

## License and Citation

### License

This project is licensed under the MIT License.

### Citation

If you use GroupAppeals in your research, please cite:

```bibtex
@misc{groupappeals2025,
  title={GroupAppeals: A Python Package for Analyzing Social Group Appeals in Party Manifestos},
  author={Dolinsky, Alona O. and Horne, Will and Huber, Lena Maria},
  year={2025},
  url={https://github.com/alonadoli/GroupAppeals}
}
```

### Research Papers

Please also cite the underlying research:

```bibtex
@misc{horne2025detecting,
  title={Detecting and Classifying Social Group Appeals using Language Models},
  author={Horne, Will and Dolinsky, Alona O. and Huber, Lena Maria},
  year={2025},
  url={https://osf.io/preprints/osf/fp2h3_v2}
}

@misc{dolinsky2025parties,
  title={Who do Parties Speak To? Introducing the PSoGA: A New Comprehensive Database of Parties' Social Group Appeals},
  author={Dolinsky, Alona O. and Huber, Lena Maria and Horne, Will},
  year={2025},
  url={https://osf.io/preprints/osf/64jwa_v1}
}
```