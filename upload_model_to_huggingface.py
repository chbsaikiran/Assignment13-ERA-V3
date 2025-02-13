
from huggingface_hub import HfApi

# Replace 'your_model_name' and 'your_username' with appropriate values
repo_id = "chbsaikiran/smollm2_sai_105M"
api = HfApi()

api.create_repo(repo_id=repo_id, exist_ok=True)

# Upload model
api.upload_file(
    path_or_fileobj="model_bin.pth",
    path_in_repo="model_bin.pth",
    repo_id=repo_id,
    repo_type="model"
)

print(f"Model uploaded to: https://huggingface.co/{repo_id}")
