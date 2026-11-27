"""Run a shell command in a cheap CPU container with the workspace volume mounted.

Clones skylight_kernels onto the volume on first use (token from the `github-readonly`
Modal secret, stripped from the remote URL afterwards so it is never stored on disk).

    modal run repo_shell.py --cmd "ls /vol/repos"
"""

import modal

app = modal.App("inkling-repo-shell")
vol = modal.Volume.from_name("inkling-workspace")
image = modal.Image.debian_slim(python_version="3.11").apt_install("git", "tree")

KERNELS_DIR = "/vol/repos/skylight_kernels"
KERNELS_REPO = "github.com/skylight-org/skylight_kernels.git"


@app.function(image=image, volumes={"/vol": vol}, cpu=1, memory=2048, timeout=600,
              secrets=[modal.Secret.from_name("github-readonly")])
def sh(cmd: str) -> str:
    import os
    import subprocess

    if not os.path.exists(KERNELS_DIR):
        tok = os.environ["GITHUB_TOKEN"]
        subprocess.run(["git", "clone", "--quiet", f"https://x-access-token:{tok}@{KERNELS_REPO}",
                        KERNELS_DIR], check=True)
        subprocess.run(["git", "-C", KERNELS_DIR, "remote", "set-url", "origin",
                        f"https://{KERNELS_REPO}"], check=True)
        vol.commit()
    out = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    return out.stdout + out.stderr


@app.local_entrypoint()
def main(cmd: str = "ls"):
    print(sh.remote(cmd))
