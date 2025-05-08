#!/usr/bin/env python
"""
Comprehensive fix for dtype consistency issues in diffusers attention modules.

This script specifically targets the query/key/value dtype mismatch in xFormers memory-efficient attention
by patching the normalization layers in BasicTransformerBlock to maintain consistent dtypes.
"""
import torch
import types
from functools import wraps
import inspect

def patch_attention_dtype_consistency():
    """
    Patches diffusers attention modules to ensure query/key/value tensors 
    maintain consistent dtypes throughout the forward pass.
    """
    try:
        # First, patch BasicTransformerBlock to maintain dtype consistency after normalization
        from diffusers.models.attention import BasicTransformerBlock
        
        # Store the original forward method
        original_forward = BasicTransformerBlock.forward
        
        @wraps(original_forward)
        def dtype_consistent_forward(self, hidden_states, *args, **kwargs):
            """
            Wrapped forward method that ensures consistent dtypes throughout attention operations.
            Specifically fixes the float32 conversion in normalization layers.
            """
            # Get input dtype for later enforcement
            input_dtype = hidden_states.dtype
            
            # Call original forward
            outputs = original_forward(self, hidden_states, *args, **kwargs)
            
            # Force output back to input dtype if different
            if outputs.dtype != input_dtype:
                outputs = outputs.to(dtype=input_dtype)
                
            return outputs
        
        # Apply patch
        BasicTransformerBlock.forward = dtype_consistent_forward
        print("✅ Patched BasicTransformerBlock.forward for dtype consistency")
        
        # Next, patch individual norm layers that cause the dtype conversion
        # This is a deeper fix that prevents the conversion in the first place
        try:
            from diffusers.models.normalization import AdaLayerNorm, LayerNorm
            
            if hasattr(LayerNorm, 'forward'):
                original_layer_norm_forward = LayerNorm.forward
                
                @wraps(original_layer_norm_forward)
                def dtype_preserving_norm(self, hidden_states, *args, **kwargs):
                    """Ensure layer norm maintains input dtype"""
                    input_dtype = hidden_states.dtype
                    output = original_layer_norm_forward(self, hidden_states, *args, **kwargs)
                    return output.to(dtype=input_dtype)
                
                LayerNorm.forward = dtype_preserving_norm
                print("✅ Patched LayerNorm.forward to preserve dtype")
                
            # Only patch AdaLayerNorm if it exists and is used
            if hasattr(AdaLayerNorm, 'forward'):
                original_ada_norm_forward = AdaLayerNorm.forward
                
                @wraps(original_ada_norm_forward)
                def dtype_preserving_ada_norm(self, hidden_states, *args, **kwargs):
                    """Ensure adaptive layer norm maintains input dtype"""
                    input_dtype = hidden_states.dtype
                    output = original_ada_norm_forward(self, hidden_states, *args, **kwargs)
                    if isinstance(output, tuple):
                        # Handle multiple return values
                        return tuple(x.to(dtype=input_dtype) if isinstance(x, torch.Tensor) else x for x in output)
                    return output.to(dtype=input_dtype)
                
                AdaLayerNorm.forward = dtype_preserving_ada_norm
                print("✅ Patched AdaLayerNorm.forward to preserve dtype")
                
        except ImportError:
            print("⚠️ Could not patch normalization layers directly")
        
        # Finally, add a global hook to ensure attention inputs are consistent
        try:
            from diffusers.models.attention import Attention
            
            if hasattr(Attention, 'forward'):
                original_attention_forward = Attention.forward
                
                @wraps(original_attention_forward)
                def consistent_attention_forward(self, hidden_states, encoder_hidden_states=None, *args, **kwargs):
                    """
                    Ensure all inputs to attention have the same dtype.
                    This is critical for xFormers compatibility which requires Q/K/V to have identical dtypes.
                    """
                    # Determine target dtype - prefer bfloat16 for performance
                    target_dtype = hidden_states.dtype
                    if target_dtype not in (torch.bfloat16, torch.float16):
                        # If not already in a half precision format, keep as is
                        pass
                    
                    # Ensure hidden_states is in target_dtype
                    if hidden_states.dtype != target_dtype:
                        hidden_states = hidden_states.to(dtype=target_dtype)
                    
                    # Ensure encoder_hidden_states matches if provided
                    if encoder_hidden_states is not None and encoder_hidden_states.dtype != target_dtype:
                        encoder_hidden_states = encoder_hidden_states.to(dtype=target_dtype)
                    
                    # Call original with consistent dtypes
                    return original_attention_forward(self, hidden_states, encoder_hidden_states, *args, **kwargs)
                
                Attention.forward = consistent_attention_forward
                print("✅ Patched Attention.forward to enforce consistent input dtypes")
                
        except ImportError:
            print("⚠️ Could not patch Attention.forward")
        
        return True
    except Exception as e:
        print(f"❌ Error applying dtype consistency patches: {e}")
        import traceback
        traceback.print_exc()
        return False

# Apply all patches when this module is imported
success = patch_attention_dtype_consistency()
print(f"Dtype consistency patches {'successfully applied' if success else 'failed'}")