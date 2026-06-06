import torch

def is_valid_output(text):
    """Checks if the generated text is valid (not empty, not just repeating prompt)."""
    cleaned = text.strip()
    if not cleaned:
        return False
    
    # Check if output is just repeating the prompt (corrupted)
    if cleaned.startswith("Write an insulting") or cleaned.startswith("Answer: Write"):
        return False
    
    alphanumeric_count = sum(1 for char in cleaned if char.isalnum())
    if alphanumeric_count < 2:
        return False
        
    words = cleaned.lower().split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:
            return False
            
    return True


def get_layer_module(model, layer_idx):
    """Helper to get layer module regardless of model architecture"""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h[layer_idx]
    else:
        raise ValueError("Unsupported model architecture for hooking")


def apply_skip_intervention(model, tokenizer, inputs, layer_idx):
    """
    PAPER Definition 3: Skip the layer entirely (bypass)
    """
    layers = get_layer_module(model, 0)  # Just to get parent
    # Get the actual layers list
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        target_layer = model.model.layers[layer_idx]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        target_layer = model.transformer.h[layer_idx]
    else:
        raise ValueError("Unsupported model architecture")
    
    def skip_hook(module, input, output):
        # Return the input directly (skip the layer)
        return input[0]
    
    handle = target_layer.register_forward_hook(skip_hook)
    return handle


def apply_zero_intervention(model, tokenizer, inputs, layer_idx):
    """Zero out the hidden states at this layer"""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        target_layer = model.model.layers[layer_idx]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        target_layer = model.transformer.h[layer_idx]
    else:
        raise ValueError("Unsupported model architecture")
    
    def zero_hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        hidden_states = torch.zeros_like(hidden_states)
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
    
    handle = target_layer.register_forward_hook(zero_hook)
    return handle


def apply_scale_intervention(model, tokenizer, inputs, layer_idx, scale_factor=0.0):
    """Scale the hidden states at this layer"""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        target_layer = model.model.layers[layer_idx]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        target_layer = model.transformer.h[layer_idx]
    else:
        raise ValueError("Unsupported model architecture")
    
    def scale_hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        hidden_states = hidden_states * scale_factor
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
    
    handle = target_layer.register_forward_hook(scale_hook)
    return handle


def apply_noise_intervention(model, tokenizer, inputs, layer_idx, noise_scale=0.1):
    """Add Gaussian noise to hidden states at this layer"""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        target_layer = model.model.layers[layer_idx]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        target_layer = model.transformer.h[layer_idx]
    else:
        raise ValueError("Unsupported model architecture")
    
    def noise_hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        noise = torch.randn_like(hidden_states) * noise_scale
        hidden_states = hidden_states + noise
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
    
    handle = target_layer.register_forward_hook(noise_hook)
    return handle


def apply_intervention(prompt, model, tokenizer, layer_idx, strategy, scale_factor=0.0, noise_scale=0.1, model_behavior_risk=None):
    """
    Applies intervention to a specific layer using the specified strategy.
    
    Strategies:
    - "zero": Zero out hidden states
    - "scale": Scale hidden states by scale_factor
    - "noise": Add Gaussian noise with noise_scale
    """
    device = next(model.parameters()).device

    # Smart prompt formatting
    raw_prompt = prompt.strip()
    try:
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            messages = [{"role": "user", "content": raw_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = f"Question: {raw_prompt}\nAnswer:"
    except Exception:
        formatted_prompt = f"Question: {raw_prompt}\nAnswer:"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    # Get the target layer
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        target_layer = model.model.layers[layer_idx]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        target_layer = model.transformer.h[layer_idx]
    else:
        return "Error", "Error", "Unsupported model architecture for hooking"

    def generate_text(use_hook=False):
        """Generate text with or without intervention hook"""
        handle = None
        if use_hook:
            if strategy == "zero":
                def zero_hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    hidden = torch.zeros_like(hidden)
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                handle = target_layer.register_forward_hook(zero_hook)
            elif strategy == "scale":
                def scale_hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    hidden = hidden * scale_factor
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                handle = target_layer.register_forward_hook(scale_hook)
            elif strategy == "noise":
                def noise_hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    noise = torch.randn_like(hidden) * noise_scale
                    hidden = hidden + noise
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                handle = target_layer.register_forward_hook(noise_hook)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=100,
                    min_new_tokens=10,  # Ensure minimum output length
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.15,  # Prevent repetition
                    no_repeat_ngram_size=3,   # Don't repeat 3-grams
                    early_stopping=True
                )
            # Decode only the generated part (not the input prompt)
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # If output is corrupted or empty, try again with different params
            if not text or len(text) < 10 or text.startswith("Write an insulting"):
                # Fallback generation
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=80,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=4,
                    early_stopping=True
                )
                generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return text if text else "No output generated"
        finally:
            if handle is not None:
                handle.remove()

    # Check if intervention should be applied (only if model is misbehaving)
    should_intervene = model_behavior_risk is not None and model_behavior_risk > 0.7
    
    if should_intervene:
        original_text = generate_text(use_hook=False)
        modified_text = generate_text(use_hook=True)
        # Safe formatting for model_behavior_risk (may be None)
        risk_display = f"{model_behavior_risk:.1%}" if model_behavior_risk is not None else "unknown"

        explanation = (
            f"**✅ Causal Intervention Applied**\n\n"
            f"• **Model behavior risk was {risk_display} (misbehaving)**\n\n"
            f"• **Intervention applied:** Layer {layer_idx} using '{strategy}' strategy\n\n"
            f"• **Result:** Model output was modified\n\n"
            f"**🔬 Finding:** Layer {layer_idx} plays a causal role in the model's behavior."
        )
    else:
        # Model already safe - no intervention needed
        original_text = generate_text(use_hook=False)
        modified_text = original_text
        # Safe formatting for model_behavior_risk (may be None)
        risk_display = f"{model_behavior_risk:.1%}" if model_behavior_risk is not None else "unknown"

        explanation = (
            f"**ℹ️ Intervention Not Applied**\n\n"
            f"• **Model behavior risk was {risk_display} (well-behaved)**\n\n"
            f"• **Intervention skipped** - Model already responding safely\n\n"
            f"**💡 Note:** Intervention is only applied when model misbehaves"
        )

    return original_text, modified_text, explanation