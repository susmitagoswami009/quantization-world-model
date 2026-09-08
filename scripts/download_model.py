"""
Downloads the Qwen2-VL-2B-Instruct backbone from Hugging Face to E:\\quantization-world-model\\models\\hf_source.

Run once, before GGUF conversion. Requires HUGGINGFACE_HUB env token only if the
model repo is gated (check on first run — script will error clearly if so).
"""

import os
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEST = r"E:\quantization-world-model\models\hf_source"

if __name__ == "__main__":
    os.makedirs(DEST, exist_ok=True)
    path = snapshot_download(repo_id=MODEL_ID, local_dir=DEST)
    print(f"Downloaded {MODEL_ID} to {path}")
