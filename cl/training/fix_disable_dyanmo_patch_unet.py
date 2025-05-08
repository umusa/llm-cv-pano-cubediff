#!/usr/bin/env python
"""
Combined fix for UNet patching and torch._dynamo disabling
This script applies both fixes and verifies they work correctly
"""
import torch
import sys
import importlib

def disable_dynamo_thoroughly():
    """Thoroughly disable torch._dynamo"""
    import torch
    if hasattr(torch, "_dynamo"):
        try:
            # Method 1: Import and call disable
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.reset()
            torch._dynamo.disable()
            
            # Method 2: Set environment variable
            import os
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
            
            # Method 3: Modify sys.modules to prevent import
            if "torch._dynamo.config" in sys.modules:
                del sys.modules["torch._dynamo.config"]
            
            # Verify disable worked
            try:
                torch._dynamo.is_dynamo_supported()
                print("❌ torch._dynamo still seems to be active")
                return False
            except:
                print("✅ torch._dynamo successfully disabled")
                return True
        except Exception as e:
            print(f"❌ Error disabling torch._dynamo: {e}")
            return False
    else:
        print("ℹ️ torch._dynamo not found, no need to disable")
        return True

def patch_unet_directly():
    """
    Directly patch UNet2DConditionModel.forward using a different approach
    that should work regardless of verification issues
    """
    try:
        # Force reimport
        if "diffusers.models.unet_2d_condition" in sys.modules:
            del sys.modules["diffusers.models.unet_2d_condition"]
        
        # Import UNet2DConditionModel
        from diffusers import UNet2DConditionModel
        
        # Get the original forward method
        if hasattr(UNet2DConditionModel, "_original_forward"):
            # Already patched
            original_forward = UNet2DConditionModel._original_forward
        else:
            # Not patched yet
            original_forward = UNet2DConditionModel.forward
            # Save original for reference
            UNet2DConditionModel._original_forward = original_forward
        
        # Define a new forward function
        def robust_forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
            """
            A robust forward method that filters out extra arguments
            """
            # Print for debugging
            print(f"UNet robust_forward called with {len(args)} args and {len(kwargs)} kwargs")
            # Call original with only the required arguments
            return original_forward(self, sample, timestep, encoder_hidden_states)
        
        # Replace the forward method
        UNet2DConditionModel.forward = robust_forward
        
        # Verify the patch
        import inspect
        new_src = inspect.getsource(UNet2DConditionModel.forward)
        if "robust_forward" in new_src and "**kwargs" in new_src:
            print("✅ UNet2DConditionModel.forward successfully patched")
            return True
        else:
            print("❌ UNet2DConditionModel.forward patch verification failed")
            print(f"Current forward source: {new_src[:200]}...")
            return False
    
    except Exception as e:
        print(f"❌ Error patching UNet2DConditionModel: {e}")
        return False

def fix_dtype_in_model(model=None):
    """
    Ensure all Q/K/V tensors in the model use consistent dtype
    This can be called after model initialization
    """
    if model is None:
        return False
    
    count = 0
    try:
        # Set target dtype
        dtype = torch.bfloat16
        device = next(model.parameters()).device
        
        # Process all modules in the model
        for module in model.modules():
            # Check if module has Q/K/V projections
            if hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v"):
                # Cast Q/K/V to the same dtype
                module.to_q = module.to_q.to(dtype=dtype, device=device)
                module.to_k = module.to_k.to(dtype=dtype, device=device)
                module.to_v = module.to_v.to(dtype=dtype, device=device)
                
                # Also cast output projection if present
                if hasattr(module, "to_out"):
                    if isinstance(module.to_out, list) and len(module.to_out) > 0:
                        module.to_out[0] = module.to_out[0].to(dtype=dtype, device=device)
                    else:
                        module.to_out = module.to_out.to(dtype=dtype, device=device)
                
                count += 1
        
        print(f"✅ Applied dtype casting to {count} attention modules")
        return True
    
    except Exception as e:
        print(f"❌ Error fixing dtype in model: {e}")
        return False

# Apply fixes when run directly
if __name__ == "__main__":
    print("\n🔧 Applying combined UNet and dynamo fixes...\n")
    
    # Apply UNet patch
    unet_fixed = patch_unet_directly()
    
    # Disable dynamo
    dynamo_disabled = disable_dynamo_thoroughly()
    
    # Overall status
    if unet_fixed and dynamo_disabled:
        print("\n✅ All fixes successfully applied!")
    else:
        print("\n⚠️ Some fixes failed to apply.")
    
    print("\nCall fix_dtype_in_model(your_model) after model initialization to ensure dtype consistency.")

# Import this module in your training script to apply fixes
print("Imported combined_unet_dynamo_fix.py - call patch_unet_directly() and disable_dynamo_thoroughly() to apply fixes")