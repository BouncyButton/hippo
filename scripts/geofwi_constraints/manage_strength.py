"""Use the verified transfer/submission flow for the separate strength stage."""
import argparse

from scripts.geofwi_constraints import prepare_strength as prepared
from scripts.geofwi_constraints import transfer, launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("transfer", "launch"))
    args = parser.parse_args()
    module = transfer if args.action == "transfer" else launch
    for key in ("ROOT", "NAME", "REPORT", "REMOTE"):
        setattr(module, key, getattr(prepared, key))
    module.main()


if __name__ == "__main__":
    main()
