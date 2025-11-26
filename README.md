# Japanese Bar Exam QA - LLM Evaluation

## Overview

This project evaluates various Large Language Models (LLMs) on the **Japanese Bar Exam Question-Answer dataset** (JBE-QA). It provides a comprehensive evaluation framework for both closed-source models (Claude, GPT) and open-source models, supporting both few-shot and zero-shot prompting strategies.

## Key Features

- **Multi-Model Support**: Evaluate 20+ LLMs including Claude, GPT-4o, Qwen, Llama, and more
- **Few-shot & Zero-shot**: Test both learning paradigms with configurable exemplars
- **Flexible Infrastructure**: Easy to extend with new models and datasets

## Dataset

[**Japanese Bar Exam QA v2**](https://huggingface.co/datasets/nguyenthanhasia/japanese-bar-exam-qa)
- Legal multiple-choice questions from Japan's bar examination
- Binary classification: judge statements as correct (1) or incorrect (0)
- Splits: training, validation, and test sets
- 4 exemplars selected for few-shot learning

## Project Structure

```
project/
├── src/
│   ├── config.py                    # Configuration and constants
│   ├── evaluate_closed_model.py     # Claude & OpenAI evaluation
│   ├── evaluate_open_model.py       # Open-source LLM evaluation
│   └── test.py                      # Main evaluation script
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (for open-source models)
- API keys for Claude and OpenAI

### Setup

1. **Clone and install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set environment variables**:
```bash
export CLAUDE_API_KEY="your-claude-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export HF_TOKEN="your-huggingface-token"
```

3. **Verify configuration** in `src/config.py`:
```python
COMMIT = '37d89813b4cdf40ffb9945341b6c103ea19afc54'  # Dataset version
dataset_name = "nguyenthanhasia/japanese-bar-exam-qa-v2"
```

## Usage

### Quick Start: Evaluate a Single Model

```bash
cd src

# Zero-shot evaluation
python test_models.py --model_id claude-3-opus-20240229 --fewshot false

# Few-shot evaluation
python test_models.py --model_id gpt-4o-2024-11-20 --fewshot true

# With reasoning (Claude/o3 models)
python test_models.py --model_id claude-opus-4-1-20250805 --fewshot true --thinking true

# Few-shot evaluation (an open-weight model)
python test_models.py --model_id meta-llama/Llama-3.1-70B-Instruct --fewshot true
```

### Supported Models

#### Closed-source (API-based)
- Claude family: Claude 3 Opus/Sonnet/Haiku, Claude Opus 4.1, Claude Sonnet 4/4.5
- OpenAI: GPT-4o, GPT-4.1, GPT-5, o3, o4-mini
- With reasoning support for Opus, o3, o4 models

#### Open-source (Local inference)
- Qwen: Qwen2.5-32B/72B-Instruct, ABEJA-Qwen2.5-32b-Japanese
- Llama: Llama-3.1/3.3-70B-Instruct, Llama-3.1-Swallow-70B
- Google: Gemma-3-12b/27b-it
- Others: Athene-V2-Chat, LLM-jp-3.1 series

## Architecture

### Evaluation Pipeline

```
Dataset → Format Prompt → Call LLM API/Load a Model → Parse Output → Evaluate → DataFrame
```

### Key Components

#### `config.py`
- Environment configuration and API keys
- Prompt formatting function

#### `evaluate_closed_model.py`
- **Claude**: Uses Anthropic Batch API for batch processing
- **OpenAI**: Uses OpenAI Batch API with support for reasoning models
- Handles extended thinking mode

#### `evaluate_open_model.py`
- ModelLoader class wraps HuggingFace transformers
- Model-specific output parsing with regex patterns

#### `test_models.py`
- Entry point for all evaluations
- Automatic routing to appropriate evaluation function
- Aggregates results and prints accuracy/F1 scores

#### `instance_constuction_sample.ipynb`
- Sample code that shows the usage of `format_question` function (converts a dataset instance into a prompt)

## Output

### Results Format

Each evaluation produces a CSV with:
- `id`, `year`, `subject`, `subject_jp`: Question metadata
- `question`, `label`, `answer`: Question content and ground truth
- `prediction`: Model's prediction (0/1)
- `accuracy`: Whether prediction matches answer
- `in_exemplars`: Whether used in few-shot examples
- `split`: train/validation/test
- `model`: Model identifier
- `fewshot`: Whether few-shot was used
- `NotANumber`: Parsing failure indicator

## Configuration

### Exemplars
4 hard-coded exemplars for few-shot learning:
```python
exemplars = dataset.select([113, 1285, 1628, 2609])
```

### System Prompt (Japanese)
```
以下の法律に関する問題を解答せよ。理由や説明は不要。
「正しい」と判断した時に1を、「誤り」と判断したきに0を出力せよ。
出力は必ず1または0のいずれかの整数値のみとせよ。
```

## Requirements

See `requirements.txt`:
- torch >= 2.0
- transformers >= 4.30
- datasets >= 2.10
- pandas >= 1.3
- anthropic >= 0.7
- openai >= 1.0
- scikit-learn >= 1.0

## Citation

If you use this evaluation framework, please cite:

```bibtex
@misc{jbe_qa_2025,
  title={JBE-QA: Japanese Bar Exam QA Dataset for Assessing Legal Domain Knowledge},
  author={Zhihan Cao, Fumihito Nishino, Hiroaki Yamada, Nguyen Ha Thanh, Yusuke Miyao, Ken Satoh},
  year={2025},
  publisher={Arxiv},
}
```

## License

This code is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

The underlying content originates from official government sources that are in the public domain. Please refer to the original source for their terms of use.

## Contact

For questions or issues, please open an issue on GitHub or contact [cao.z.c8a7@m.isct.ac.jp]
