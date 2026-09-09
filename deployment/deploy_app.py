
import os
from huggingface_hub import HfApi

# These variables will be used by GitHub Actions
HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_REPO_ID = "senthil31/tourism-product-prediction-app-space"
DEPLOYMENT_FOLDER = "tourism_project/deployment"

if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable is not set. Please add it to your GitHub Secrets.")
else:
    api = HfApi()
    print(f"Uploading folder '{DEPLOYMENT_FOLDER}' to Space '{SPACE_REPO_ID}'...")
    
    api.upload_folder(
        folder_path=DEPLOYMENT_FOLDER,
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        token=HF_TOKEN
    )
    print("App deployment to Hugging Face Spaces completed successfully.")
