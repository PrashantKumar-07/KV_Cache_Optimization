import os
from huggingface_hub import snapshot_download

cache_dir = '/data/nishant/Nishant/Prashant/Project/hf_cache'
local_dir = '/data/nishant/Nishant/Prashant/Project/models/mistral-7b-instruct'

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(local_dir, exist_ok=True)

print("Starting download...")
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    local_dir=local_dir,
    cache_dir=cache_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)
print("Download finished!")
