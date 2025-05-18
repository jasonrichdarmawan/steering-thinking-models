# Understanding Reasoning in Thinking Language Models via Steering Vectors

Implementation of experiments and analysis for the paper ["Understanding Reasoning in Thinking Language Models via Steering Vectors"](https://openreview.net/forum?id=OwhVWNOBcz).

This repository provides tools for analyzing reasoning patterns in language models through steering vectors, allowing deeper insights into how LLMs process and execute different types of reasoning.

## Repository Structure

```
steering-thinking-models/
├── compare-base-reasoning/    # Compare reasoning capabilities between models
│   ├── compare_reasoning.py  # Core comparison implementation
│   └── run.sh               # Execution script
├── messages/                 # Message handling utilities
├── notebooks/               # Analysis notebooks
├── steering/                # Core steering implementation
│   ├── coefficient_study.py # Study steering coefficient effects
│   ├── evaluate_MATH.py    # Evaluate on MATH dataset
│   ├── evaluate_steering.py # General steering evaluation
│   └── evaluate_vectors.py # Vector evaluation utilities
├── train-steering-vectors/  # Training steering vectors
│   ├── cosine_sim.py      # Cosine similarity analysis
│   └── train_vectors.py   # Vector training implementation
├── utils/                  # Common utilities
└── vector-layer-attribution/ # Layer effect analysis
    └── analyze_layer_effects.py # Layer-wise effect analysis
```

## Requirements

- A workstation with 24GB VRAM (e.g., 1x NVIDIA 3090) for loading DeepSeek-R1-Distill-Llama-8B
- API keys from:
  - [OpenAI Platform](https://platform.openai.com/api-keys)
  - [OpenRouter](https://openrouter.com/settings/keys)
  - [NDIF](https://login.ndif.us/)

## Installation

1. Install Miniconda:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

2. Create and activate environment:
```bash
conda create --name steering-env python=3.11
conda activate steering-env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (.env file):
```
OPENAI_ORG_ID=<org id from https://platform.openai.com/settings/organization/general>
OPENAI_API_KEY=<api key from https://platform.openai.com/api-keys>
OPENROUTER_API_KEY=<api key from https://openrouter.com/settings/keys>
NDIF_API_KEY=<api key from https://login.ndif.us/>
```

## Using nnsight

This project uses nnsight for model analysis. Key integration points:

1. Model Loading:
```python
from nnsight import NNsight, LanguageModel
model = NNsight.from_pretrained("model_name")
```

2. Remote Execution:
```python
CONFIG.set_default_api_key(os.getenv("NDIF_API_KEY"))
with model.session(remote=True):
    # Your code here
```

3. Layer Tracing:
```python
with model.trace(input_ids) as tracer:
    # Capture layer outputs
    for layer_idx in range(model.config.num_hidden_layers):
        layer_outputs.append(model.model.layers[layer_idx].output[0].save())
```

## Key Features

1. Training Steering Vectors:
```bash
python train_vectors.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --n_samples 500 --max_tokens 1000 --batch_size 4 \
    --save_every 1 --load_from_json
```

2. Analyzing Layer Effects:
```bash
python analyze_layer_effects.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --n_examples 500 --load_in_8bit True
```

3. Evaluating Steering:
```bash
python evaluate_steering.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

## Migration to nnsight Remote

To migrate to using nnsight remotely:

1. Ensure API key is set in .env file:
```
NDIF_API_KEY=your_key_here
```

2. Configure remote execution:
```python
from nnsight import CONFIG
CONFIG.set_default_api_key(os.getenv("NDIF_API_KEY"))
```

3. Use remote session:
```python
with model.session(remote=True):
    # model operations here

# or 
with model.trace(remote=True) as tracer:
    ...
```

4. Key files requiring migration:
   - train_vectors.py
   - analyze_layer_effects.py
   - evaluate_steering.py
   - evaluate_vectors.py

## License

TBD
