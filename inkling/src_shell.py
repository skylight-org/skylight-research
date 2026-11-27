"""CPU shell inside the GPU image (transformers/torch installed), for reading library source.

    modal run src_shell.py --cmd "python -c 'import transformers; print(transformers.__file__)'"
"""
import modal

from paged_modal import image as _img

image = _img.add_local_python_source("paged_modal")

app = modal.App("inkling-src-shell")
vol = modal.Volume.from_name("inkling-workspace")


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192, timeout=900)
def sh(cmd: str) -> str:
    import subprocess

    out = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    return out.stdout[-60000:] + out.stderr[-4000:]


@app.local_entrypoint()
def main(cmd: str = "ls"):
    print(sh.remote(cmd))
