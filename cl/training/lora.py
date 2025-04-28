import peft, torch.nn as nn
def apply_lora(unet, r=4, alpha=16):
    # Debug: Print model structure
#     for name, module in unet.named_modules():
#         if "to_q" in name or "to_k" in name or "to_v" in name or "to_out" in name:
#             print(f"Found target: {name}, type: {type(module)}")
        
    targets = ["to_q", "to_k", "to_v", "to_out.0"]  # Cubediff inflated-attn names
    cfg = peft.LoraConfig(
            r=r, lora_alpha=alpha,
            target_modules=targets,
            bias="none", task_type="FEATURE_EXTRACTION")
    return peft.get_peft_model(unet, cfg)
