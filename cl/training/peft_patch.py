#!/usr/bin/env python
"""
Revised PEFT/UNet patching to filter out unwanted parameters
without impacting training performance.

This handles different PEFT versions and avoids patching non-existent methods.
"""
import types
from functools import wraps

# Parameters that we need to filter out
FILTER_PARAMS = [
    'input_ids', 'attention_mask', 'inputs_embeds',
    'output_hidden_states', 'output_attentions', 'return_dict'
]

def create_filtered_forward(original_forward, filter_params=None):
    """Create a forward function that filters out specified parameters"""
    if filter_params is None:
        filter_params = FILTER_PARAMS
        
    @wraps(original_forward)
    def filtered_forward(self, *args, **kwargs):
        # Filter out problematic parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in filter_params}
        
        # Uncomment for debugging - but disable in production for speed
        # removed = set(kwargs.keys()) - set(filtered_kwargs.keys())
        # if removed:
        #     print(f"Filtered {removed} from {self.__class__.__name__}.forward")
        
        return original_forward(self, *args, **filtered_kwargs)
    
    return filtered_forward

def patch_peft_unet_classes():
    """
    Patch the specific PEFT and UNet classes that need parameter filtering.
    This checks if methods exist before patching them.
    """
    try:
        # 1. UNet patching
        from diffusers import UNet2DConditionModel
        if hasattr(UNet2DConditionModel, 'forward'):
            original_unet_forward = UNet2DConditionModel.forward
            UNet2DConditionModel.forward = create_filtered_forward(original_unet_forward)
            print("✅ Patched UNet2DConditionModel.forward")
        else:
            print("⚠️ UNet2DConditionModel has no forward method")

        # 2. PEFT Model patching
        try:
            from peft.peft_model import PeftModel
            if hasattr(PeftModel, 'forward'):
                original_peft_forward = PeftModel.forward
                PeftModel.forward = create_filtered_forward(original_peft_forward)
                print("✅ Patched PeftModel.forward")
            else:
                print("⚠️ PeftModel has no forward method")
        except (ImportError, AttributeError):
            print("⚠️ PeftModel not found, skipping")
        
        # 3. LoraModel patching
        try:
            from peft.tuners.lora import LoraModel
            if hasattr(LoraModel, 'forward'):
                original_lora_forward = LoraModel.forward
                LoraModel.forward = create_filtered_forward(original_lora_forward)
                print("✅ Patched LoraModel.forward")
            else:
                print("⚠️ LoraModel has no forward method")
        except (ImportError, AttributeError):
            print("⚠️ LoraModel not found, skipping")
        
        # 4. Skip BaseTunerLayer patching as it may not have a forward method
        
        # 5. PeftModelForFeatureExtraction patching (conditional)
        try:
            from peft.peft_model import PeftModelForFeatureExtraction
            if hasattr(PeftModelForFeatureExtraction, 'forward'):
                original_peft_feature_forward = PeftModelForFeatureExtraction.forward
                PeftModelForFeatureExtraction.forward = create_filtered_forward(original_peft_feature_forward)
                print("✅ Patched PeftModelForFeatureExtraction.forward")
            else:
                print("⚠️ PeftModelForFeatureExtraction has no forward method")
        except (ImportError, AttributeError):
            print("⚠️ PeftModelForFeatureExtraction not found, skipping")
            
        print("✅ PEFT and UNet classes patched successfully")
        return True
    except Exception as e:
        print(f"❌ Error patching classes: {e}")
        import traceback
        traceback.print_exc()
        return False

# Auto-execute when the script is imported
__patched = patch_peft_unet_classes()