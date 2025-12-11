# sacuploadmodel.py
import os
from huggingface_hub import HfApi, upload_file, login

# ---------------------------
# SETTINGS
# ---------------------------
# Use environment variable for token (set HF_TOKEN in your environment)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set HF_TOKEN environment variable with your Hugging Face token")
  
# Get the directory where this script is located, then go up one level to find models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")
USERNAME = "MohamedMaher003"

# ---------------------------
# LOGIN
# ---------------------------
login(HF_TOKEN)
api = HfApi()

# ---------------------------
# SCAN MODELS DIRECTORY
# ---------------------------
# List of model files to upload (matching your saved .pth files)
MODEL_FILES = [
    "sac_model_actor.pth",
    "sac_model_critic.pth",
    "sac_model_critic_target.pth",
    "sac_model_encoder.pth",
    "README.md"  # Model card
]

# Specify which environment to upload (set to None to upload all)
TARGET_ENV = "LunarLander-v3"

for env_name in os.listdir(MODELS_DIR):
    # Skip if not the target environment
    if TARGET_ENV and env_name != TARGET_ENV:
        continue
        
    env_path = os.path.join(MODELS_DIR, env_name)
    
    if not os.path.isdir(env_path):
        continue
    
    # Check if at least the actor model exists
    actor_file = os.path.join(env_path, "sac_model_actor.pth")
    
    if os.path.exists(actor_file):
        repo_id = f"{USERNAME}/{env_name}-SAC"

        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"Uploading models for {env_name} to Hugging Face repo {repo_id}...")

        # Upload all model files
        for model_filename in MODEL_FILES:
            model_file = os.path.join(env_path, model_filename)
            if os.path.exists(model_file):
                upload_file(
                    path_or_fileobj=model_file,
                    path_in_repo=model_filename,
                    repo_id=repo_id,
                    repo_type="model",
                    token=HF_TOKEN
                )
                print(f"  ✅ Uploaded {model_filename}")
            else:
                print(f"  ⚠️ {model_filename} not found, skipping.")
        
        print(f"✅ Finished uploading {env_name} model!\n")
    else:
        print(f"⚠️ Model file not found for {env_name}, skipping.")
