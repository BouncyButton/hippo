import argparse
from pathlib import Path

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a .pth checkpoint to Weights & Biases as an artifact.")
    parser.add_argument("--file", required=True, help="Path to the .pth file to upload.")
    parser.add_argument("--name", default=None, help="Artifact name. Defaults to the checkpoint stem.")
    parser.add_argument("--type", default="model", help="Artifact type. Defaults to 'model'.")
    parser.add_argument("--project", default="hippopotamus-project", help="W&B project name.")
    parser.add_argument("--entity", default="hippopotamus", help="W&B entity/account.")
    parser.add_argument("--run-name", default=None, help="Optional W&B run name.")
    parser.add_argument(
        "--alias",
        action="append",
        default=None,
        help="Artifact alias. Repeat to add multiple aliases, e.g. --alias latest --alias fold1.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.file).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix != ".pth":
        raise ValueError(f"expected a .pth file, got: {checkpoint_path}")

    artifact_name = args.name or checkpoint_path.stem
    aliases = args.alias or ["latest"]

    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name, job_type="artifact-upload")
    artifact = wandb.Artifact(name=artifact_name, type=args.type)
    artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
    run.log_artifact(artifact, aliases=aliases)
    run.finish()

    print(f"uploaded {checkpoint_path} as artifact '{artifact_name}' with aliases {aliases}")


if __name__ == "__main__":
    main()
