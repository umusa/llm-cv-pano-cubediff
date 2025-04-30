import peft
def apply_lora(unet, r=4, alpha=16):
    """
    Inject LoRA adapters in every Q / K / V / out projection of the UNet
    (self-attention and cross-attention).  Uses PEFT’s built-in patcher,
    so no hand-written wrappers → minimum VRAM.
    """
    # Debug: Print model structure
    # for name, module in unet.named_modules():
    #     if "to_q" in name or "to_k" in name or "to_v" in name or "to_out" in name:
    #         print(f"Found target: {name}, type: {type(module)}")
    
    # targets = ["to_q", "to_k", "to_v", "to_out.0"]  # Cubediff inflated-attn names
    targets = [
        "to_q", "to_k", "to_v", "to_out.0",        # diffusers ≤0.30 names
        "q_proj", "k_proj", "v_proj", "out_proj",  # diffusers ≥0.31 names
    ]
    cfg = peft.LoraConfig(
            r=r, lora_alpha=alpha,
            target_modules=targets,
            bias="none", task_type="FEATURE_EXTRACTION")
    return peft.get_peft_model(unet, cfg)

from peft.tuners.lora import LoraLayer


# import peft, torch.nn as nn                    # make sure peft is imported
# # from peft.tuners.lora import Linear as LoRALinear 

# def patch_qkv_out(module: nn.Module, r: int, alpha: int):
#     """
#     LoRA-wrap any module that carries {to_q,to_k,to_v,to_out} *or*
#     {q_proj,k_proj,v_proj,out_proj}.
#     Works for both Self- and Cross-attention in all Diffusers versions.
#     """
#     names = [("to_q", "to_k", "to_v", "to_out"),
#              ("q_proj", "k_proj", "v_proj", "out_proj")]

#     for keys in names:
#         if all(hasattr(module, k) for k in keys):
#             break
#     else:
#         return                                           # nothing to do

#     for k in keys:
#         base = getattr(module, k)
#         if isinstance(base, peft.tuners.lora.LoraLayer):
#             continue                                     # already wrapped
#         # cfg   = peft.LoraConfig(r=r, lora_alpha=alpha,
#         #                        target_modules=[],
#         #                        bias="none",
#         #                        task_type="FEATURE_EXTRACTION")
        
#         # print(f"lora.py - patch_qkv_out - cfg is {cfg}")
#         # wrapped = peft.tuners.lora.Linear(base.in_features,
#         #                                   base.out_features,
#         #                                   bias=base.bias is not None)
#         # wrapped.weight.data.copy_(base.weight.data)
#         # if base.bias is not None:
#         #     wrapped.bias.data.copy_(base.bias.data)
#         # wrapped = peft.tuners.lora.LoraModel(wrapped, cfg)
#         # setattr(module, k, wrapped)

#         base = getattr(module, k)

#         # only wrap real Linear layers (skip ModuleList / None)
#         if not isinstance(base, nn.Linear):
#             continue
#         if isinstance(base, peft.tuners.lora.LoraLayer):
#             continue                                # already done

#         wrapped = peft.tuners.lora.Linear(          # correct PEFT API
#             base,                                   # ← existing layer
#             adapter_name="default",
#             r=r,
#             lora_alpha=alpha,
#             lora_dropout=0.0,
#             fan_in_fan_out=False,
#             lora_bias=False,
#         )
#         setattr(module, k, wrapped)

#         # wrapped.weight.data.copy_(base.weight.data)
#         # if base.bias is not None:
#             # wrapped.bias.data.copy_(base.bias.data)
#         # setattr(module, k, wrapped)


# def apply_lora(unet, r=4, alpha=16):    
#     print(f"lora.py = r is {r}")
#     # for name, module in unet.named_modules():
#     #     if isinstance(module, nn.MultiheadAttention) or isinstance(module, CrossAttention):
#     #         patch_qkv_out(module, r, alpha)
#     for _, module in unet.named_modules():
#         patch_qkv_out(module, r, alpha)        # will patch both self & cross
#     # after LoRA-patching every projection, hand the model back
#     return unet
