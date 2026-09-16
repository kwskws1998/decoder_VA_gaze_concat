## Installation

To run the project (Linux):

1. Create a virtual environment with python 3.11.8 and activate it  
2. Install the required dependencies from requirements.txt
3. Install the tokenizer aligner package:

```bash
pip install git+https://github.com/angelalopezcardona/tokenizer_aligner.git@v1.0.0
```
4. Install the eyetrackpy package:
```bash
pip install git+https://github.com/angelalopezcardona/eyetrackpy.git@v1.0.0
```

## Usage

Example: Training with OASST1 dataset using Meta-Llama-3-8B

```bash
python rlhf_rw/main.py \
  -d OpenAssistant/oasst1 \
  -m meta-llama/Meta-Llama-3-8B \
  --concat True \
  --seed 42
```

### Key Parameters

- `-d, --dataset_name`: Dataset to use for training. Currently, only OpenAssistant/oasst1 and nvidia/HelpSteer2 are supported.
- `-m, --model_name`: Base model to fine-tune. You can pass full model IDs `meta-llama/Meta-Llama-3-8B`, `meta-llama/Llama-3-8B-Instruct`
- `--concat`: Whether to concatenate prompt and response. True is of GazeConcat and False is of GazeAdd.
