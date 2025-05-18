# %%
"""
This script analyzes the effects of different reasoning behaviors on the layers of a language model.
It uses KL divergence as a metric to measure the impact of specific labeled sections of text on the model's 
hidden layers. The script also visualizes the results by plotting the layer effects for each label.
Modules and Libraries:
- argparse: For parsing command-line arguments.
- dotenv: For loading environment variables from a `.env` file.
- os: For file and directory operations.
- torch: For tensor operations and model computations.
- transformers: For loading and using pre-trained language models.
- nnsight: For interfacing with the NNsight API and analyzing language models.
- re: For regular expression operations.
- tqdm: For progress bars.
- matplotlib: For plotting results.
- numpy: For numerical operations.
- gc: For garbage collection.
- jaxtyping: For type annotations of tensors.
- einops: For tensor operations and reshaping.
Functions:
- find_label_positions(annotated_response, original_text, tokenizer, label):
    Parses annotations and finds token positions for a given label in the text.
- compute_kl_divergence_metric(logits):
    Computes the KL divergence between the predicted distribution and its detached version.
- analyze_layer_effects(model, tokenizer, text, label, feature_vectors, label_positions):
    Analyzes the effects of specific labels on the model's layers by computing gradients and activations.
- plot_layer_effects(layer_effects, model_name):
    Plots the layer effects for each label, showing the mean KL divergence and standard deviation across layers.
Command-line Arguments:
- --model: The name of the model to analyze (default: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B").
- --n_examples: The number of examples to analyze per label (default: 10).
- --load_in_8bit: Whether to load the model in 8-bit mode (default: False).
- --remote: Whether to run the analysis on the NNsight server (default: True).
Usage Example:
Run the script with the following command:
    python analyze_layer_effects.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --n_examples 500 --load_in_8bit True
Output:
- Results are saved in the `train-steering-vectors/results` directory.
- Figures showing the layer effects are saved as PNG files in the `results/figures` directory.
Notes:
- The script assumes the presence of a `.env` file containing the `NN_SIGHT_API_KEY`.
- The `utils.load_model_and_vectors` function is used to load the model and compute feature vectors.
- The script includes a TODO comment questioning the use of a `continue` statement in the `find_label_positions` function.
"""
# python vector-layer-attribution/analyze_layer_effects.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --n_examples 10 --load_in_8bit True --remote --plot_only True
import argparse
import dotenv
import os
from dotenv import load_dotenv, find_dotenv
env_file = find_dotenv(filename=".env", raise_error_if_not_found=False)
if not env_file:
    raise FileNotFoundError("Could not locate a .env file in any parent directory")
# 2. Load it *with* override so we ensure variables are set
load_dotenv(env_file, override=True)

# 3. Confirm it’s there
api_key = os.getenv("NN_SIGHT_API_KEY")
if api_key is None:
    raise RuntimeError(f"NN_SIGHT_API_KEY not found in {env_file!r}")
import nnsight

from nnsight import NNsight, LanguageModel, CONFIG
CONFIG.set_default_api_key(api_key)

from typing import Any
import torch
from torch import Tensor
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer

CONFIG.set_default_api_key(os.getenv("NN_SIGHT_API_KEY"))
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils

import gc
from jaxtyping import Float
from einops import einops

# Add argparse for model selection
parser = argparse.ArgumentParser(description="Analyze layer effects for different reasoning behaviors")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to analyze")
parser.add_argument("--n_examples", type=int, default=10,
                    help="Number of examples to analyze per label")
parser.add_argument("--load_in_8bit", type=bool, default=False,
                    help="Load the model in 8-bit mode")
parser.add_argument("--remote", action="store_true", default=True,
                    help="Run on nnsight server")
parser.add_argument("--plot_only", action="store_true", default=False,
                    help="Run on nnsight server")
args, _ = parser.parse_known_args()
REMOTE = args.remote
PLOT_ONLY = args.plot_only
# give eample run command
# python analyze_layer_effects --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --n_examples 500 --load_in_8bit True

# Create directories
RESULTS_FOLDER_PATH = "train-steering-vectors/results"
os.makedirs(f'{RESULTS_FOLDER_PATH}/vars', exist_ok=True)
os.makedirs(f'{RESULTS_FOLDER_PATH}/figures', exist_ok=True)

# %%
def find_label_positions(annotated_response: str, 
                         original_text: str, 
                         tokenizer: PreTrainedTokenizer, 
                         label: str):
    """Parse annotations and find token positions for each label"""
    label_positions: list[tuple[int, int]] = []
    pattern = f'\\["{label}"\\]([^\\[]+?)(?=\\[|$)'
    matches = re.finditer(pattern, annotated_response)
    thinking_tokens = tokenizer.encode(original_text)[1:]        
    
    for match in matches:

        text = match.group(1).strip()
        text_tokens = tokenizer.encode(text)[1:]

        if len(text_tokens) == 0:
            continue
        elif len(text_tokens) == 1:
            offset = 0
        elif len(text_tokens) > 1:
            offset = 1
            text_tokens = text_tokens[1:]
        
        for j in range(len(thinking_tokens) - len(text_tokens) + 1):
            if thinking_tokens[j:j + len(text_tokens)] == text_tokens:
                token_start = j-offset
                token_end = j + len(text_tokens)
                label_positions.append((token_start, token_end))
                continue
    
    return label_positions

def compute_kl_divergence_metric(logits: Float[Tensor, "batch seq_len vocab_size"]):
    """Compute KL divergence between predicted distribution and detached version"""
    probs = F.log_softmax(logits, dim=-1)
    detached_probs = F.log_softmax(logits.detach(), dim=-1)
    return F.kl_div(probs, detached_probs, reduction='batchmean')

# def analyze_layer_effects(model: LanguageModel, 
#                           tokenizer: PreTrainedTokenizer, 
#                           text: str, 
#                           label: str, 
#                           feature_vectors: dict[str, Float[Tensor, "num_hidden_layers hidden_size"]], 
#                           label_positions: list[tuple[int, int]]):
#     '''what is the return: 
#     - list of patching effects for each layer
#     type: list[float]'''
#     if len(label_positions) == 0:
#         return None

#     patching_effects = [0 for _ in range(model.config.num_hidden_layers)]

#     input_ids: Float[Tensor, "batch seq_len"] = tokenizer(text, return_tensors="pt").input_ids

#     for pos in label_positions:
#         start, end = pos

#         # 1) Collect _proxy_ slices inside the trace
#         layer_grad_slices: list[nnsight.intervention.graph.proxy.InterventionProxy] = []
#         with model.trace(input_ids[:, :end], remote=REMOTE):
#             # forward + backward as before...
#             logits = model.lm_head.output
#             value = compute_kl_divergence_metric(logits[0, start])
#             value.backward()

#             for layer_idx in range(model.config.num_hidden_layers):
#                 model.model.layers[layer_idx].output[0].requires_grad_(True)
#                 grad = model.model.layers[layer_idx].output[0].grad
#                 # take the slice
#                 slice_proxy = grad[0, start-1 : min(start, end-2)]
#                 # schedule it to be saved—but don't read it yet
#                 # saved_proxy = slice_proxy.save()
#                 # layer_grad_slices.append(saved_proxy)

#         # 2) Once we're _out_ of the trace, each proxy now has .value populated
#         layer_gradients: list[torch.Tensor] = []
#         for proxy in layer_grad_slices:
#             real_tensor = proxy.value.detach()   # now a bona-fide torch.Tensor
#             layer_gradients.append(real_tensor)

#         # you can now safely print, slice again, einsum, etc.
#         print(layer_gradients[0].shape, type(layer_gradients[0]))
#         # → torch.Size([s, d]) <class 'torch.Tensor'>
#         feature_activation = feature_vectors[label].to(torch.bfloat16)

#         for layer_idx in range(model.config.num_hidden_layers):
#             # Get activations and gradients for the entire labeled section
#             # gradients = layer_gradients[layer_idx][0, start-1:min(start, end-2)]
#             gradients = layer_gradients[layer_idx]
            
#             effect = einops.einsum(feature_activation[layer_idx], gradients, 'd, s d -> s').mean().abs()
            
#             patching_effects[layer_idx] += effect
            
#             # Clean up layer-specific tensors
#             del gradients
        
#         # Clean up batch-specific tensors
#         del layer_gradients
#         del feature_activation
#         # torch.cuda.empty_cache()
#         gc.collect()

#         patching_effects = [effect.item() for effect in patching_effects]

#     patching_effects = [effect / len(label_positions) for effect in patching_effects]

#     return patching_effects

def analyze_layer_effects(
    model: LanguageModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    label: str,
    feature_vectors: dict[str, Float[Tensor, "num_hidden_layers hidden_size"]],
    label_positions: list[tuple[int, int]],
):
    if not label_positions:
        return None

    # Prepare
    device = next(model.parameters()).device
    feat = feature_vectors[label].to(device).to(torch.bfloat16)
    nlayers = model.config.num_hidden_layers
    # accumulator for remote scalars
    saved_effects: list[list[nnsight.Proxy]] = [ [] for _ in range(nlayers) ]

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    # 1) Do everything inside one big trace per position
    for start, end in label_positions:
        with model.trace(input_ids[:, :end], remote=REMOTE) as tracer:
            # run forward + backward
            logits = model.lm_head.output
            loss = compute_kl_divergence_metric(logits[0, start])
            loss.backward()

            # for each layer, compute the dot-product → scalar
            for layer_idx in range(nlayers):
                model.model.layers[layer_idx].output[0].requires_grad_(True)
                grad_proxy = model.model.layers[layer_idx].output[0].grad   # shape [s, d]
                
                # slice the time-steps you care about
                grad_slice = grad_proxy[start - 1 : min(start, end - 2)]  # shape [s, d]

                # remote einsum + mean + abs → scalar proxy
                # note: if nnsight proxies support torch.einsum, otherwise do manual
                proj = torch.einsum("d,sd->s", feat[layer_idx], grad_slice)  # still proxy
                mean_abs = proj.abs().mean()                                 # proxy scalar

                # **SAVE** only the scalar into saved_effects
                saved_effects[layer_idx].append(mean_abs.save())

    # 2) After all traces, pull down only your scalars
    # this will batch-fetch under the hood, instead of thousands of big arrays
    patching_effects = []
    for layer_idx in range(nlayers):
        # get Python floats
        vals = [p.value.item() for p in saved_effects[layer_idx]]
        patching_effects.append(sum(vals) / len(vals))

    return patching_effects

def plot_layer_effects(layer_effects: dict[str, list[list[float]]], model_name: str):
    # Set up the figure with subplots
    n_labels = sum(1 for label, effects in layer_effects.items() if effects)
    axes: list[plt.Axes] = []
    fig, axes = plt.subplots(1, n_labels, figsize=(6*n_labels, 8), facecolor='white')
    
    # Handle the case where there's only one subplot
    if n_labels == 1:
        axes = [axes]
    
    # Color scheme
    colors = ['#2E86C1', '#E67E22', '#27AE60', '#C0392B']
    
    # Get model ID for title
    model_id = model_name.split('/')[-1]
    
    # Counter for valid labels
    valid_label_idx = 0
    
    for (label, effects), color in zip(layer_effects.items(), colors):
        if not effects:  # Skip if no effects for this label
            continue
            
        # Get current axis
        ax = axes[valid_label_idx]
        ax.set_facecolor('white')
        
        effects_array = np.array(effects)
        
        # Handle NaN values by replacing them with 0
        effects_array = np.nan_to_num(effects_array, nan=0.0)
        
        # Compute mean and std, ignoring NaN values
        mean_effects = np.nanmean(effects_array, axis=0)
        std_effects = np.nanstd(effects_array, axis=0)
        
        # Apply smoothing using convolution
        window_size = 1  # Increase coarseness by reducing window size
        kernel = np.ones(window_size) / window_size
        smoothed_effects = np.convolve(mean_effects, kernel, mode='valid')
        std_smoothed = np.convolve(std_effects, kernel, mode='valid')
        
        x = range(len(smoothed_effects))
        
        ax.fill_between(x, 
                        smoothed_effects - std_smoothed,
                        smoothed_effects + std_smoothed,
                        alpha=0.2, 
                        color=color)
        
        ax.plot(x, smoothed_effects, 
                color=color,
                linewidth=2.5,
                marker='o',
                markersize=4)
        
        # Set title and labels for each subplot
        ax.set_title("{}".format(label.replace('-', ' ').title()), 
                    fontsize=20, 
                    pad=10, 
                    color='black')
        
        ax.set_xlabel('Layer', fontsize=16, labelpad=10, color='black')
        
        # Only set y-label for the first subplot
        if valid_label_idx == 0:
            ax.set_ylabel('Mean KL-Divergence', fontsize=16, labelpad=10, color='black')
        
        ax.tick_params(axis='both', which='major', labelsize=14, colors='black')
        
        # Remove offset on x-axis
        ax.margins(x=0)
        
        # Add box and grid with stronger visibility
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)  # Make the box lines thicker
            spine.set_color('black')  # Set explicit color
        
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        
        # Enhanced grid settings
        ax.grid(True, 
                linestyle='--',      # Dashed lines
                alpha=0.4,           # More opaque
                color='gray',        # Gray color
                which='major')       # Show major grid lines
        
        valid_label_idx += 1
    
    # Add a common title for all subplots
    fig.suptitle(model_id, fontsize=24, y=0.98, color='black')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the suptitle
    
    model_id_lower = model_name.split('/')[-1].lower()
    
    plt.savefig(f'{RESULTS_FOLDER_PATH}/figures/layer_effects_{model_id_lower}_subplots.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.show()
    plt.close()

# %%
# Load model and data
model_name: str = args.model
REMOTE = args.remote
print(f"Loading model {model_name}...")
feature_vectors: dict[str, Float[Tensor, "num_hidden_layers hidden_size"]] = {}
model, tokenizer, feature_vectors = utils.load_model_and_vectors(
    load_in_8bit=args.load_in_8bit, 
    compute_features=True, 
    model_name=model_name,
    dispatch=False)

# %%
# Get model identifier for file naming
model_id = model_name.split('/')[-1].lower()
responses_path = f'{RESULTS_FOLDER_PATH}/vars/responses_{model_id}.json'

with open(responses_path, 'r') as f:
    results = json.load(f)

# %%
labels = [
    'backtracking',
    'uncertainty-estimation',
    'example-testing',
    'adding-knowledge',
]
n_examples: int = args.n_examples  # Number of examples to analyze per label

# Store results
layer_effects: dict[str, list[list[float]]] = {label: [] for label in labels}

if not PLOT_ONLY:
    # Analyze each label
    for label in labels:
        print(f"Analyzing label: {label}")
        for example in tqdm(results[:n_examples]):
            original_text: str = example['full_response']
            annotated_text: str = example['annotated_thinking']

            
            # Find token positions of labeled sentences
            label_positions: list[tuple[int, int]] = []
            for label_j in labels:
                if label_j != label:
                    label_positions.extend(find_label_positions(annotated_response=annotated_text, 
                                                                original_text=original_text,
                                                                tokenizer=tokenizer, 
                                                                label=label_j))

            if label_positions:  # Only process if we found labeled sentences
                effects = analyze_layer_effects(
                    model=model,
                    tokenizer=tokenizer,
                    text=original_text,
                    label=label,
                    feature_vectors=feature_vectors,
                    label_positions=label_positions
                )

                if effects:
                    layer_effects[label].append(effects)
        # save layer_effects, layer_effects is a list of float
        with open(f'{RESULTS_FOLDER_PATH}/vars/layer_effects_{label}_{model_id}.json', 'w') as f:
            json.dump(layer_effects, f, indent=4)

# %% Plot results
else:
    # read layer_effects from json file
    with open(f'{RESULTS_FOLDER_PATH}/vars/layer_effects_{model_id}.json', 'r') as f:
        layer_effects = json.load(f)
    plot_layer_effects(layer_effects=layer_effects, model_name=model_name)

# %%

# TODO: is this a bug?
# perhaps instead of `continue` statement, we want `break` statement?
# this function returns 
# [(0,4), (4,8)] for label="backtracking"
# [(0,4), (4,8)] for label="uncertainty-estimation"
#
# find_label_positions(
#     annotated_response="[\"backtracking\"]hello hello hello.[\"end-section\"][\"uncertainty-estimation\"]hello hello hello.[\"end-section\"]",
#     original_text="hello hello hello. hello hello hello.",
#     tokenizer=tokenizer,
#     label="uncertainty-estimation"
# )

# %%
