#!/usr/bin/env python
"""
Fix for xFormers attention parameter mismatch in diffusers v0.33.1
"""
import torch
from functools import wraps

def fix_xformers_attention():
    """Fix the parameter mismatch in memory efficient attention for diffusers 0.33.1"""
    try:
        # First try to find and patch the set_use_memory_efficient_attention_xformers method
        from diffusers.models.attention_processor import Attention
        
        if hasattr(Attention, "set_use_memory_efficient_attention_xformers"):
            original_method = Attention.set_use_memory_efficient_attention_xformers
            
            @wraps(original_method)
            def patched_setter(self, use_memory_efficient_attention_xformers=True, *args, **kwargs):
                """Handle both 2-arg and 3-arg calls by discarding extra args"""
                try:
                    # Try calling with just the first required argument
                    return original_method(self, use_memory_efficient_attention_xformers)
                except Exception as e:
                    print(f"Warning: First attempt to set memory efficient attention failed: {e}")
                    try:
                        # If that fails, try with the legacy signature
                        return original_method(self, use_memory_efficient_attention_xformers, None)
                    except Exception as e2:
                        print(f"Error: Could not set memory efficient attention: {e2}")
                        # Return the original output to avoid breaking the chain
                        return self
            
            # Apply the patch
            Attention.set_use_memory_efficient_attention_xformers = patched_setter
            print("✅ Patched Attention.set_use_memory_efficient_attention_xformers")
            
            # Now directly patch UNet2DConditionModel.enable_xformers_memory_efficient_attention
            from diffusers import UNet2DConditionModel
            if hasattr(UNet2DConditionModel, "enable_xformers_memory_efficient_attention"):
                orig_enable = UNet2DConditionModel.enable_xformers_memory_efficient_attention
                
                @wraps(orig_enable)
                def safe_enable_xformers(self, attention_op=None):
                    """Safely enable xformers memory efficient attention"""
                    try:
                        # Attempt standard call
                        return orig_enable(self)
                    except TypeError:
                        # If that fails, try manually calling set_use on each attention module
                        for module in self.modules():
                            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                                try:
                                    module.set_use_memory_efficient_attention_xformers(True)
                                except Exception as e:
                                    print(f"Warning: Could not set memory efficient attention on module: {e}")
                        return self
                
                UNet2DConditionModel.enable_xformers_memory_efficient_attention = safe_enable_xformers
                print("✅ Patched UNet2DConditionModel.enable_xformers_memory_efficient_attention")
            
            # Also ensure tensor dtypes are consistent
            def ensure_consistent_dtypes(model):
                dtype = torch.bfloat16
                for module in model.modules():
                    if hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v"):
                        module.to_q = module.to_q.to(dtype=dtype)
                        module.to_k = module.to_k.to(dtype=dtype)
                        module.to_v = module.to_v.to(dtype=dtype)
                
            # Add the helper method to UNet2DConditionModel
            UNet2DConditionModel.ensure_consistent_dtypes = ensure_consistent_dtypes
            print("✅ Added dtype consistency helper to UNet2DConditionModel")
            
            return True
        else:
            print("⚠️ Could not find set_use_memory_efficient_attention_xformers method")
            return False
            
    except ImportError as e:
        print(f"⚠️ Could not import required modules: {e}")
        return False
    except Exception as e:
        print(f"❌ Error fixing xFormers: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_xformers_attention()
    print(f"xFormers fix {'succeeded' if success else 'failed'}")
else:
    # Apply fix when imported
    fix_xformers_attention()