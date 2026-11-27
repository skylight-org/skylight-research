"""Download Inkling-2-Mini + clone skylight-research onto a Modal Volume.

Runs on a CPU-only container (no GPU cost). Nothing is stored locally.

    modal run setup_volume.py
"""

import modal

app = modal.App("inkling-setup")
vol = modal.Volume.from_name("inkling-workspace", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("huggingface_hub[hf_xet]>=0.34")
)

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
REPO_URL = "https://github.com/skylight-org/skylight-research.git"


@app.function(
    image=image,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-token")],
    cpu=4,
    memory=8192,
    timeout=60 * 60,  # hard 1h ceiling (~$0.25 worst case)
)
def setup():
    import os
    import subprocess

    from huggingface_hub import snapshot_download

    os.makedirs("/vol/repos", exist_ok=True)
    repo_dir = "/vol/repos/skylight-research"
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", REPO_URL, repo_dir], check=True)
    else:
        subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
    vol.commit()

    path = snapshot_download(
        MODEL_ID,
        local_dir="/vol/models/Inkling-2-Mini",
        token=os.environ["HF_TOKEN"],
        max_workers=16,
    )
    vol.commit()
    out = subprocess.run(["du", "-sh", path, repo_dir], capture_output=True, text=True)
    print(out.stdout)


@app.local_entrypoint()
def main():
    setup.remote()
