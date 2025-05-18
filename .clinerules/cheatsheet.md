# Steering Models Cheatsheet

## Quick Setup

```bash
# Create environment
conda create --name steering-env python=3.11
conda activate steering-env

# Install dependencies
pip install -r requirements.txt

# Set up API keys in .env
echo "OPENAI_ORG_ID=your_org_id" > .env
echo "OPENAI_API_KEY=your_api_key" >> .env
echo "OPENROUTER_API_KEY=your_router_key" >> .env
echo "NDIF_API_KEY=your_ndif_key" >> .env
```

## Common Commands

### Training Steering Vectors

```bash
# Basic training
python train_vectors.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# Advanced training
python train_vectors.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --n_samples 500 \
    --max_tokens 1000 \
    --batch_size 4 \
    --save_every 1 \
    --load_from_json

# Update annotations only
python train_vectors.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --load_from_json \
    --update_annotation
```

### Analyzing Layer Effects

```bash
# Basic analysis
python analyze_layer_effects.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# Memory-optimized analysis
python analyze_layer_effects.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --n_examples 500 \
    --load_in_8bit True
```

### Evaluating Models

```bash
# Run MATH evaluation
./steering/run_math_eval.sh

# Run general steering evaluation
python steering/evaluate_steering.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

## nnsight Integration Points

### Model Loading
```python
from nnsight import NNsight, LanguageModel
model = NNsight.from_pretrained("model_name")
```

### Remote Execution
```python
# Configure API key
from nnsight import CONFIG
CONFIG.set_default_api_key(os.getenv("NDIF_API_KEY"))

# Use remote session
with model.session(remote=True):
    # Model operations here
```

### Layer Tracing
```python
# Capture layer outputs
with model.trace(input_ids) as tracer:
    for layer_idx in range(model.config.num_hidden_layers):
        outputs.append(model.model.layers[layer_idx].output[0].save())
```

## Troubleshooting

### Common Issues

1. Insufficient VRAM
   - Solution: Use `--load_in_8bit True` flag
   - Example: `python train_vectors.py --load_in_8bit True`

2. NaN Values in Results
   - Check batch size and reduce if needed
   - Verify input data normalization
   - Inspect layer gradients for anomalies

3. Remote Execution Failures
   - Verify NDIF API key is set correctly
   - Check network connection
   - Ensure model is supported by remote service

### Memory Management

```python
# Clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# Use gradient checkpointing
model.gradient_checkpointing_enable()
```

## Migration to nnsight

### Files to Modify
1. train_vectors.py
   - Update model loading
   - Add remote execution configuration
   - Modify layer tracing

2. analyze_layer_effects.py
   - Add CONFIG setup
   - Update session management
   - Modify activation capture

3. evaluate_steering.py & evaluate_vectors.py
   - Update model initialization
   - Add remote execution support
   - Modify tensor operations

### Key Changes
1. Model Loading:
   ```python
   # Before
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained(...)

   # After
   from nnsight import NNsight
   model = NNsight.from_pretrained(...)
   ```

2. Layer Access:
   ```python
   # Before
   hidden_states = outputs.hidden_states

   # After
   with model.trace(input_ids) as tracer:
       hidden_states = model.model.layers[layer_idx].output[0].save()
   ```

3. Remote Execution:
   ```python
   # Add at start of script
   CONFIG.set_default_api_key(os.getenv("NDIF_API_KEY"))

   # Wrap operations
   with model.session(remote=True):
       # Your code here
   ```
