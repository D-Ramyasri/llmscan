import torch
import numpy as np
from scipy.stats import skew, kurtosis

def get_model_layers(model):
    """
    Get transformer layers regardless of model architecture.
    Supports: Llama, TinyLlama, Qwen, GPT2, SmolLM, etc.
    """
    # Try different model architectures in order
    patterns = [
        # Llama / TinyLlama / Qwen pattern
        (lambda m: hasattr(m, 'model') and hasattr(m.model, 'layers'), lambda m: m.model.layers),
        # GPT2 pattern
        (lambda m: hasattr(m, 'transformer') and hasattr(m.transformer, 'h'), lambda m: m.transformer.h),
        # Qwen alternative pattern
        (lambda m: hasattr(m, 'model') and hasattr(m.model, 'transformer') and hasattr(m.model.transformer, 'h'), 
         lambda m: m.model.transformer.h),
        # SmolLM pattern (similar to Llama)
        (lambda m: hasattr(m, 'model') and hasattr(m.model, 'decoder') and hasattr(m.model.decoder, 'layers'),
         lambda m: m.model.decoder.layers),
        # Direct layers attribute
        (lambda m: hasattr(m, 'layers'), lambda m: m.layers),
        # Fallback: search recursively
        (lambda m: True, lambda m: _find_layers_recursive(m))
    ]
    
    for condition, getter in patterns:
        try:
            if condition(model):
                layers = getter(model)
                if layers is not None and len(layers) > 0:
                    return layers
        except:
            continue
    
    raise ValueError(f"Cannot find transformer layers in model type: {type(model)}")

def _find_layers_recursive(model, max_depth=3):
    """Recursively search for a list of layers in the model"""
    import torch.nn as nn
    for name, child in model.named_children():
        if name in ['layers', 'h', 'decoder_layers', 'encoder_layers']:
            if isinstance(child, nn.ModuleList) and len(child) > 0:
                return child
        if hasattr(child, '__len__') and len(child) > 0:
            first = child[0] if len(child) > 0 else None
            if first is not None and hasattr(first, 'forward'):
                return child
    return None

def get_selected_layer_heads(model, num_layers, num_heads):
    """Select first, middle, last layers and heads per paper Appendix C.2"""
    if hasattr(model, 'config'):
        model_name = getattr(model.config, '_name_or_path', '')
    else:
        model_name = ''
    
    if '13b' in model_name.lower():
        selected_layers = [0, 19, 39]
    else:  # 7B, 8B models
        selected_layers = [0, num_layers // 2, num_layers - 1]
    
    selected_heads = [0, num_heads // 2, num_heads - 1]
    
    return [(l, h) for l in selected_layers for h in selected_heads]


def compute_token_causal_effects(prompt, model, tokenizer):
    """
    PAPER Definition 2: Replace each token with '-' and compute Euclidean distance
    between original and intervened attention scores.
    """
    device = next(model.parameters()).device
    
    # Get model dimensions
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    selected_pairs = get_selected_layer_heads(model, num_layers, num_heads)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs['input_ids'][0]
    seq_len = input_ids.shape[0]
    
    # Get token strings for output
    tokens = [tokenizer.decode([tid.item()]) for tid in input_ids]
    
    # Intervention token '-'
    intervention_ids = tokenizer("-", add_special_tokens=False)
    intervention_token_id = intervention_ids.input_ids[0] if intervention_ids.input_ids else tokenizer.unk_token_id
    
    # Step 1: Baseline forward pass - get attention scores from selected layers/heads
    with torch.no_grad():
        outputs = model(inputs['input_ids'], output_attentions=True)
    
    baseline_attentions = {}
    for layer_idx, head_idx in selected_pairs:
        attn = outputs.attentions[layer_idx][0, head_idx, :seq_len, :seq_len].detach().cpu()
        baseline_attentions[(layer_idx, head_idx)] = attn
    
    # Step 2: For each token, intervene and compute distance
    token_ces = []
    for i in range(seq_len):
        # Skip special tokens like BOS, EOS
        if input_ids[i].item() in tokenizer.all_special_ids:
            token_ces.append(0.0)
            continue
        
        # Create intervened input
        intervened_ids = input_ids.clone()
        intervened_ids[i] = intervention_token_id
        intervened_input = {'input_ids': intervened_ids.unsqueeze(0).to(device)}
        
        # Forward pass with intervention
        with torch.no_grad():
            interv_outputs = model(**intervened_input, output_attentions=True)
        
        # Compute distances for each selected layer/head
        distances = []
        for layer_idx, head_idx in selected_pairs:
            interv_attn = interv_outputs.attentions[layer_idx][0, head_idx, :seq_len, :seq_len].detach().cpu()
            dist = torch.norm(baseline_attentions[(layer_idx, head_idx)] - interv_attn, p=2).item()
            distances.append(dist)
        
        # Average across selected pairs (per paper)
        token_ces.append(float(np.mean(distances)))
    
    return token_ces, tokens

def compute_layer_causal_effects_skip(prompt, model, tokenizer):
    """
    PAPER Definition 3: Skip a layer and compute Euclidean distance between
    original and skipped output logits of the first token.
    """
    device = next(model.parameters()).device
    
    # Get number of layers
    num_layers = model.config.num_hidden_layers
    
    # Get layers using the new helper function
    layers = get_model_layers(model)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Step 1: Baseline - get logits for first token (position 0)
    with torch.no_grad():
        baseline_outputs = model(**inputs)
    baseline_logits = baseline_outputs.logits[0, 0, :].detach().cpu()
    
    # Step 2: For each layer, skip it and compute difference
    layer_ces = []
    
    def make_skip_hook():
        """Hook that bypasses the layer completely"""
        def skip_hook(module, input, output):
            hidden = input[0]
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return skip_hook
    
    for layer_idx in range(num_layers):
        handle = layers[layer_idx].register_forward_hook(make_skip_hook())
        
        with torch.no_grad():
            skipped_outputs = model(**inputs)
        
        handle.remove()
        
        skipped_logits = skipped_outputs.logits[0, 0, :].detach().cpu()
        ce = torch.norm(baseline_logits - skipped_logits, p=2).item()
        layer_ces.append(ce)
    
    return layer_ces

def compute_layer_causal_effects_all(prompt, model, tokenizer):
    """
    EXTENDED: Compute layer causal effects using MULTIPLE strategies:
    1. Skip (paper's method)
    2. Zero
    3. Scale
    4. Noise
    
    Returns a dictionary with all strategies.
    """
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    
    # Get layers using the new helper function
    layers = get_model_layers(model)
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Baseline logits
    with torch.no_grad():
        baseline_outputs = model(**inputs)
    baseline_logits = baseline_outputs.logits[0, 0, :].detach().cpu()
    
    results = {
        'skip': [],
        'zero': [],
        'scale': [],
        'noise': []
    }
    
    # Helper: generate with intervention hook
    def generate_with_hook(hook_fn, layer_idx):
        handle = None
        try:
            handle = layers[layer_idx].register_forward_hook(hook_fn)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.logits[0, 0, :].detach().cpu()
        finally:
            if handle:
                handle.remove()
    
    for layer_idx in range(num_layers):
        # 1. SKIP (paper's method)
        def skip_hook(module, input, output):
            hidden = input[0]
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        skipped_logits = generate_with_hook(skip_hook, layer_idx)
        results['skip'].append(torch.norm(baseline_logits - skipped_logits, p=2).item())
        
        # 2. ZERO
        def zero_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = torch.zeros_like(hidden)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        zeroed_logits = generate_with_hook(zero_hook, layer_idx)
        results['zero'].append(torch.norm(baseline_logits - zeroed_logits, p=2).item())
        
        # 3. SCALE (scale_factor = 0.5 as example)
        scale_factor = 0.5
        def scale_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden * scale_factor
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        scaled_logits = generate_with_hook(scale_hook, layer_idx)
        results['scale'].append(torch.norm(baseline_logits - scaled_logits, p=2).item())
        
        # 4. NOISE
        noise_scale = 0.1
        def noise_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            noise = torch.randn_like(hidden) * noise_scale
            hidden = hidden + noise
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        noised_logits = generate_with_hook(noise_hook, layer_idx)
        results['noise'].append(torch.norm(baseline_logits - noised_logits, p=2).item())
    
    return results

def extract_token_features(token_ces):
    """
    PAPER Section 3.2: Extract 5 statistical features from token CEs
    """
    token_ces = np.array(token_ces)
    
    if len(token_ces) == 0:
        return np.zeros(5)
    
    mean_val = np.mean(token_ces)
    std_val = np.std(token_ces)
    range_val = np.max(token_ces) - np.min(token_ces)
    skew_val = skew(token_ces) if len(token_ces) > 2 else 0.0
    kurt_val = kurtosis(token_ces) if len(token_ces) > 3 else 0.0
    
    return np.array([mean_val, std_val, range_val, skew_val, kurt_val])


def get_token_and_layer_maps(prompt, model, tokenizer, use_all_strategies=True):
    """
    Main function that returns:
    - Token features (5-dim)
    - Layer CEs (can be from skip only or all strategies)
    """
    # Compute token causal effects (PAPER method)
    token_ces, tokens = compute_token_causal_effects(prompt, model, tokenizer)
    token_features = extract_token_features(token_ces)
    
    if use_all_strategies:
        # Get layer CEs from all strategies
        layer_results = compute_layer_causal_effects_all(prompt, model, tokenizer)
        layer_ces = {
            'skip': np.array(layer_results['skip']),
            'zero': np.array(layer_results['zero']),
            'scale': np.array(layer_results['scale']),
            'noise': np.array(layer_results['noise'])
        }
    else:
        # PAPER only: skip strategy
        layer_ces = {
            'skip': np.array(compute_layer_causal_effects_skip(prompt, model, tokenizer))
        }
    
    return {
        'token_features': token_features,  # 5-dim array
        'layer_ces': layer_ces,  # dict of arrays
        'tokens': tokens,
        'token_ces_raw': token_ces,
        'selected_layer_heads': get_selected_layer_heads(model, 
                                                        model.config.num_hidden_layers, 
                                                        model.config.num_attention_heads)
    }
def get_head_level_attention(prompt, model, tokenizer, top_k=5):
    """Return the top-k attention heads (index, contribution) for the last layer."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    try:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)

        attns = outputs.attentions
        if not attns:
            return []

        # Get last layer attentions: tensor shape (batch, num_heads, seq, seq)
        last = attns[-1]
        if last.dim() == 4:
            last = last[0]  # remove batch dimension
        # last now (num_heads, seq, seq)
        head_contributions = []
        last_np = last.cpu().numpy()
        num_heads = last_np.shape[0]
        for head_idx in range(num_heads):
            head_attn = last_np[head_idx].copy()
            # zero diagonal (self-attention) to focus on cross-token influence
            np.fill_diagonal(head_attn, 0)
            contribution = float(head_attn.mean())
            head_contributions.append((int(head_idx), contribution))

        head_contributions.sort(key=lambda x: x[1], reverse=True)
        return head_contributions[:top_k]
    except Exception as e:
        print('Error computing head-level attention:', e)
        return []

def get_response_causal_maps(response_text: str, model, tokenizer):
    """
    Extract causal features from the MODEL'S RESPONSE
    """
    try:
        # DON'T add prefix - use raw response text
        # This keeps the token distribution similar to training data
        causal_result = get_token_and_layer_maps(response_text, model, tokenizer, use_all_strategies=False)
        
        return {
            'token_features': causal_result['token_features'],
            'layer_ces': causal_result['layer_ces']['skip'],
            'tokens': causal_result['tokens'],
            'token_ces_raw': causal_result['token_ces_raw'],
            'success': True
        }
    except Exception as e:
        print(f"Error getting response causal maps: {e}")
        return {
            'token_features': np.zeros(5),
            'layer_ces': np.zeros(32),
            'tokens': [],
            'token_ces_raw': np.zeros(10),
            'success': False,
            'error': str(e)
        }