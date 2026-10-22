"""Upload trained artifacts to a Hugging Face model repo so the hosted app can download them.
Requires `huggingface-cli login` or HF_TOKEN in the environment."""
import argparse

from huggingface_hub import HfApi

from traceverify import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", help="e.g. your-username/traceverify-artifacts")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(config.ARTIFACT_DIR), repo_id=args.repo_id, repo_type="model")
    print("uploaded. Set TV_ARTIFACT_REPO =", args.repo_id, "in your deployment.")


if __name__ == "__main__":
    main()
