"""Create or update the Hugging Face Space that hosts the live demo.

Needs two environment variables (never pass them on the command line):
    HF_TOKEN      Hugging Face token with write access
    GROQ_API_KEY  stored as an encrypted Space secret, never uploaded as a file

    python scripts/deploy_space.py --space <hf-username>/document-chatbot
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ["app.py", "requirements.txt", ".streamlit", "chatbot", "sample_docs", "LICENSE"]
IGNORE = ["**/__pycache__/**", "*.pyc", "**/secrets.toml"]


def stage(target: Path) -> None:
    for name in INCLUDE:
        src = ROOT / name
        if src.is_dir():
            # .streamlit/secrets.toml holds the sign-in client secret; on a Space it must be a Space secret.
            shutil.copytree(src, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "secrets.toml"))
        else:
            shutil.copy2(src, target / name)
    for name in ("Dockerfile", "README.md"):
        shutil.copy2(ROOT / "deploy" / "space" / name, target / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="<hf-username>/<space-name>")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token, groq_key = os.getenv("HF_TOKEN"), os.getenv("GROQ_API_KEY")
    missing = [n for n, v in (("HF_TOKEN", token), ("GROQ_API_KEY", groq_key)) if not v]
    if missing:
        print(f"Set {', '.join(missing)} in the environment first.")
        return 1

    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.add_space_secret(args.space, "GROQ_API_KEY", groq_key)

    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=tmp, ignore_patterns=IGNORE,
                          commit_message="Deploy live demo")

    print(f"Deployed: https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
