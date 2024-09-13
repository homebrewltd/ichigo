import torch
from collections import OrderedDict

# Load the original model
original_model_data = torch.load("whisper-vq-stoks-v3-7lang.model")
original_state_dict = original_model_data['state_dict']

# Create a new OrderedDict to store the renamed layers
new_state_dict = OrderedDict()

# Iterate through the state dict items
for key, value in original_state_dict.items():
    # Check if the key needs to be modified
    if key.startswith('rq.layers.0.'):
        if "_codebook" in key:
            # skip the codebook layers
            new_key = key
        else:
            new_key = key.replace('rq.layers.0.', 'rq.')
    else:
        new_key = key
    new_state_dict[new_key] = value

# Update the state_dict in the model_data
model_data = original_model_data.copy()
model_data['state_dict'] = new_state_dict

# Verification steps
print("Verification:")
print(f"Original number of layers: {len(original_state_dict)}")
print(f"New number of layers: {len(new_state_dict)}")

# Check if weights are unchanged
for key in original_state_dict.keys():
    new_key = key.replace('rq.layers.0.', 'rq.') if key.startswith('rq.layers.0.') and "_codebook" not in key else key
    if not torch.equal(original_state_dict[key], new_state_dict[new_key]):
        print(f"Warning: Weights changed for layer {key}")

# Print out the keys to verify the changes
print("\nKeys in state_dict after modifications:")
for key in new_state_dict.keys():
    print(key)

# Save the modified model data
torch.save(model_data, 'whisper-vq-stoks-v3-7lang-fixed.model')
print("\nModified model saved as: whisper-vq-stoks-v3-7lang-fixed.model")