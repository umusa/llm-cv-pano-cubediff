#!/usr/bin/env python
"""
Verification module for UNet and xFormers compatibility patches

This module can be imported to verify if all necessary patches are applied correctly
to ensure compatibility between UNet, PEFT-LoRA, and xFormers memory-efficient attention.
"""
import torch
import importlib
import sys
import os
from pathlib import Path
import inspect
import warnings

def check_unet_forward_patched():
    """Check if UNet2DConditionModel.forward is patched to filter extra parameters"""
    try:
        from diffusers import UNet2DConditionModel
        
        # Check for patched UNet forward
        forward_src = inspect.getsource(UNet2DConditionModel.forward)
        is_patched = "**kwargs" in forward_src and "ignore" in forward_src
        
        if is_patched:
            print("✅ UNet2DConditionModel.forward is correctly patched")
        else:
            print("❌ UNet2DConditionModel.forward is NOT patched")
        
        return is_patched
    except Exception as e:
        print(f"Error checking UNet2DConditionModel: {e}")
        return False

def check_attention_xformers_patched():
    """Check if Attention.set_use_memory_efficient_attention_xformers is patched"""
    try:
        from diffusers.models.attention_processor import Attention
        
        # Check if method exists
        has_method = hasattr(Attention, "set_use_memory_efficient_attention_xformers")
        
        if not has_method:
            print("❌ Attention.set_use_memory_efficient_attention_xformers method does not exist")
            return False
            
        # Try to check if it's patched
        try:
            # Get source code to check for our patch
            method_src = inspect.getsource(Attention.set_use_memory_efficient_attention_xformers)
            is_patched = "dtype" in method_src or "to(" in method_src
            
            if is_patched:
                print("✅ Attention.set_use_memory_efficient_attention_xformers is correctly patched")
            else:
                print("❌ Attention.set_use_memory_efficient_attention_xformers is NOT patched")
            
            return is_patched
        except Exception as e:
            print(f"Warning: Could not check Attention method source: {e}")
            return False
    except Exception as e:
        print(f"Error checking Attention processor: {e}")
        return False

def check_peft_forward_patched():
    """Check if PeftModel.forward is patched to filter extra parameters"""
    try:
        from peft.peft_model import PeftModel
        
        # Check for patched PeftModel forward
        forward_src = inspect.getsource(PeftModel.forward)
        is_patched = "**kwargs" in forward_src
        
        if is_patched:
            print("✅ PeftModel.forward is correctly patched")
        else:
            print("❌ PeftModel.forward is NOT patched")
        
        return is_patched
    except Exception as e:
        print(f"Error checking PeftModel: {e}")
        return False

def check_cubediff_dtype_casting():
    """Check if CubeDiffModel.forward has proper dtype casting"""
    try:
        # Try to import from local modules
        sys.path.append(str(Path.cwd()))
        from cl.model.architecture import CubeDiffModel
        
        # Check if forward method has dtype casting
        forward_src = inspect.getsource(CubeDiffModel.forward)
        has_dtype_casting = "dtype" in forward_src and "to(" in forward_src
        
        if has_dtype_casting:
            print("✅ CubeDiffModel.forward has proper dtype casting")
        else:
            print("❌ CubeDiffModel.forward does NOT have dtype casting")
        
        return has_dtype_casting
    except Exception as e:
        print(f"Error checking CubeDiffModel: {e}")
        return False

def check_xformers_configuration():
    """Check if xFormers is available and properly configured"""
    results = {}
    
    try:
        import xformers
        results["available"] = True
        print("✅ xformers is available")
        
        # Check if xformers is configured for mixed precision
        if "XFORMERS_FORCE_MIXED_PREC" in os.environ:
            results["mixed_prec_configured"] = True
            print(f"✅ xformers mixed precision configured: {os.environ['XFORMERS_FORCE_MIXED_PREC']}")
        else:
            results["mixed_prec_configured"] = False
            print("❌ xformers mixed precision is NOT configured")
    except ImportError:
        print("❌ xformers is NOT available")
        results["available"] = False
        results["mixed_prec_configured"] = False
    
    return results

def check_torch_config():
    """Check if PyTorch is configured optimally"""
    import torch
    results = {}
    
    # Check TF32 support
    results["tf32_allowed"] = torch.backends.cuda.matmul.allow_tf32
    if results["tf32_allowed"]:
        print("✅ TF32 is enabled for faster computation")
    else:
        print("❌ TF32 is not enabled")
    
    # Check CUDNN config
    results["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    results["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    
    if results["cudnn_benchmark"]:
        print("✅ cuDNN benchmark mode is enabled for faster training")
    else:
        print("❌ cuDNN benchmark mode is not enabled")
    
    # Check if dynamo is disabled (helps with compatibility)
    if hasattr(torch, "_dynamo"):
        dynamo_disabled = False
        try:
            import torch._dynamo
            dynamo_disabled = torch._dynamo.is_dynamo_supported() == False
        except:
            pass
        
        results["dynamo_disabled"] = dynamo_disabled
        if dynamo_disabled:
            print("✅ torch._dynamo is disabled for better compatibility")
        else:
            print("❌ torch._dynamo is not disabled")
    else:
        results["dynamo_disabled"] = True
    
    return results

def apply_emergency_fixes():
    """Apply emergency fixes to make things work even if patches failed"""
    print("\n🔧 Applying emergency fixes...\n")
    
    # 1. Configure environment variables for better compatibility
    os.environ["XFORMERS_FORCE_MIXED_PREC"] = "bfloat16"
    print("✅ Set XFORMERS_FORCE_MIXED_PREC=bfloat16")
    
    # 2. Disable dynamo for better compatibility
    if hasattr(torch, "_dynamo"):
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.disable()
            print("✅ Disabled torch._dynamo")
        except Exception as e:
            print(f"❌ Failed to disable torch._dynamo: {e}")
    
    # 3. Optimize CUDA memory allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    print("✅ Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
    
    # 4. Ensure TF32 is enabled for faster computation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✅ Enabled TF32 for faster computation")
    
    # 5. Set more stable cuDNN settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("✅ Configured cuDNN for stability")
    
    print("\n✅ Emergency fixes applied!\n")

def verify_patching():
    """Verify all necessary patches are applied correctly"""
    print("\n🔍 Verifying UNet and xFormers compatibility...\n")
    
    results = {
        "unet_forward_patched": check_unet_forward_patched(),
        "attention_xformers_patched": check_attention_xformers_patched(),
        "peft_forward_patched": check_peft_forward_patched(),
        "cubediff_dtype_casting": check_cubediff_dtype_casting(),
    }
    
    # Check xFormers configuration
    xformers_results = check_xformers_configuration()
    results.update({f"xformers_{k}": v for k, v in xformers_results.items()})
    
    # Check PyTorch configuration
    torch_results = check_torch_config()
    results.update({f"torch_{k}": v for k, v in torch_results.items()})
    
    # Summarize results
    success = all(results.values())
    print("\n" + "="*50)
    if success:
        print("✅ All patches are correctly applied!")
    else:
        print("⚠️ Some patches are missing or incorrect:")
        for name, status in results.items():
            if not status:
                print(f"  ❌ {name}")
        
        # Offer to apply emergency fixes
        print("\n⚠️ Would you like to apply emergency fixes? (y/n)")
        try:
            response = input().strip().lower()
            if response == 'y':
                apply_emergency_fixes()
        except:
            # Non-interactive environment
            warnings.warn("Non-interactive environment detected. Emergency fixes will not be applied automatically.")
    
    print("="*50 + "\n")
    return results

if __name__ == "__main__":
    results = verify_patching()