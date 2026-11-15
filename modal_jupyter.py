"""Start GPU Jupyter on Modal: python modal_jupyter.py."""

import argparse
import secrets
import sys

import modal


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--stop", metavar="SANDBOX_ID")
    args = parser.parse_args()

    if args.stop:
        modal.Sandbox.from_id(args.stop).terminate()
        print("Sandbox stopped.")
        return
    if not 0 < args.hours <= 24:
        parser.error("--hours must be greater than 0 and at most 24")

    # Upload the notebook and dataset to this volume before starting.
    volume = modal.Volume.from_name("chexpert-workspace")
    app = modal.App.lookup("chexpert-jupyter", create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "jupyterlab", "torch", "torchvision", "numpy", "pandas", "pillow",
            "matplotlib", "scikit-learn", "tqdm",
        )
    )
    token = secrets.token_urlsafe(32)
    print(f"Building/loading image and requesting {args.gpu} GPU...", flush=True)
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "jupyter", "lab", "--no-browser", "--allow-root",
            "--ip=0.0.0.0", "--port=8888",
            "--ServerApp.root_dir=/workspace",
            "--ServerApp.allow_remote_access=True",
            app=app,
            image=image,
            gpu=args.gpu,
            cpu=4,
            memory=32768,
            timeout=int(args.hours * 3600),
            workdir="/workspace",
            volumes={"/workspace": volume},
            secrets=[modal.Secret.from_dict({"JUPYTER_TOKEN": token})],
            env={
                "WORKSPACE_DIR": "/workspace",
                "PROJECT_DIR": "/workspace/cv-contrastive-learning",
                "DATA_DIR": "/workspace/CheXpert-v1.0-small",
            },
            encrypted_ports=[8888],
        )
        print(f"Sandbox created: {sandbox.object_id}", flush=True)
        print(f"Stop: python modal_jupyter.py --stop {sandbox.object_id}", flush=True)
        print("Waiting for tunnel metadata (up to 60 seconds)...", flush=True)
        try:
            tunnel = sandbox.tunnels(timeout=60)[8888]
        except Exception:
            print(
                "Could not obtain the Jupyter URL. Check this sandbox in the Modal "
                "dashboard before starting another GPU session.",
                flush=True,
            )
            raise

    print(f"Sandbox ID: {sandbox.object_id}")
    print(f"Open Jupyter (allow a moment for startup):\n{tunnel.url}/lab?token={token}")
    print(f"GPU: {args.gpu}; maximum session length: {args.hours:g} hours.")
    print("GPU billing continues after this script exits or you close the browser.")
    print("Before stopping, save the notebook and run in a notebook cell: !sync /workspace")
    print(f"Stop: python modal_jupyter.py --stop {sandbox.object_id}")


if __name__ == "__main__":
    main()
