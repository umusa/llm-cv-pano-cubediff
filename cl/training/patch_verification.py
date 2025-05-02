#!/usr/bin/env python
"""
Patch verification utility - checks if PEFT and UNet models are properly patched
to filter out problematic parameters.
"""
import inspect
import types

def verify_patching():
    """
    Check if the critical classes have been properly patched with parameter filtering.
    Returns a dictionary with the status of each class.
    """
    results = {
        "UNet2DConditionModel": False,
        "PeftModel": False,
        "LoraModel": False
    }
    
    # Check UNet
    try:
        from diffusers import UNet2DConditionModel
        
        # Get source code to check for filtering
        source = inspect.getsource(UNet2DConditionModel.forward)
        if "filtered_kwargs" in source or "filter" in source:
            print("✅ UNet2DConditionModel.forward is patched")
            results["UNet2DConditionModel"] = True
        else:
            print("⚠️ UNet2DConditionModel.forward does not appear to be patched")
    except Exception as e:
        print(f"❌ Could not check UNet2DConditionModel: {e}")
    
    # Check PeftModel
    try:
        from peft.peft_model import PeftModel
        
        # Get source code to check for filtering
        source = inspect.getsource(PeftModel.forward)
        if "filtered_kwargs" in source or "filter" in source:
            print("✅ PeftModel.forward is patched")
            results["PeftModel"] = True
        else:
            print("⚠️ PeftModel.forward does not appear to be patched")
    except Exception as e:
        print(f"❌ Could not check PeftModel: {e}")
    
    # Check LoraModel
    try:
        from peft.tuners.lora import LoraModel
        
        # Get source code to check for filtering
        source = inspect.getsource(LoraModel.forward)
        if "filtered_kwargs" in source or "filter" in source:
            print("✅ LoraModel.forward is patched")
            results["LoraModel"] = True
        else:
            print("⚠️ LoraModel.forward does not appear to be patched")
    except Exception as e:
        print(f"❌ Could not check LoraModel: {e}")
    
    # Final status
    if all(results.values()):
        print("✅ All critical classes are properly patched")
    else:
        print("⚠️ Not all classes are patched - training may encounter errors")
    
    return results

def test_forward_call():
    """
    Test a simple forward call to see if parameters are filtered correctly.
    """
    try:
        import torch
        from diffusers import UNet2DConditionModel
        
        # Create a minimal UNet
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            torch_dtype=torch.float16
        )
        
        # Create minimal inputs
        sample = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="cuda")
        timestep = torch.tensor([1], device="cuda")
        encoder_hidden_states = torch.randn(1, 77, 768, dtype=torch.float16, device="cuda")
        
        # Try to call with problematic parameters
        print("Testing forward call with problematic parameters...")
        output = unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            input_ids=torch.tensor([[1, 2, 3]], device="cuda"),  # Problematic parameter
            attention_mask=torch.tensor([[1, 1, 1]], device="cuda")  # Problematic parameter
        )
        
        print("✅ Forward call succeeded - parameters are being filtered correctly")
        return True
    except Exception as e:
        print(f"❌ Forward call failed: {e}")
        return False

if __name__ == "__main__":
    print("Verifying PEFT and UNet patching status...")
    results = verify_patching()
    
    # Only run the forward test if UNet is patched
    if results["UNet2DConditionModel"]:
        print("\nTesting patched forward call...")
        test_forward_call()