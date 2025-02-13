import torch

# Path to the original checkpoint file
checkpoint_path = "checkpoint.pth"  # Update the path if required
output_path = "model_bin.pth"       # Output file for model parameters

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Extract only the model parameters
if 'model_state_dict' in checkpoint:
    model_params = checkpoint['model_state_dict']
else:
    raise KeyError("'model_state_dict' not found in the checkpoint.")

# Save the model parameters to a new file
torch.save(model_params, output_path)

print(f"Model parameters saved to {output_path}.")
