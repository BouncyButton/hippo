"""Install the verified public MedSAM3 source files in the local user cache."""
import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from portable import CACHE, PROTOCOLS, read, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    spec = read(PROTOCOLS / "upstream.json")
    cache = args.cache.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / ("medsam3-" + spec["commit"])
    if not destination.exists():
        with TemporaryDirectory(dir=cache) as temp:
            checkout = Path(temp) / "upstream"
            checkout.mkdir()
            def git(*args, **kwargs):
                subprocess.run(["git", "-C", str(checkout), *args], check=True, **kwargs)
            git("init", "--quiet")
            git("remote", "add", "origin", "https://github.com/Joey-S-Liu/MedSAM3.git")
            git("sparse-checkout", "set", "--no-cone", "--stdin",
                input="".join("/" + path + "\n" for path in spec["files"]), text=True)
            git("fetch", "--quiet", "--filter=blob:none", "--depth=1", "origin", spec["commit"])
            git("checkout", "--quiet", "--detach", "FETCH_HEAD")
            for name, checksum in spec["files"].items():
                verify(checkout / name, checksum)
            checkout.rename(destination)
    for name, checksum in spec["files"].items():
        verify(destination / name, checksum)
    print(destination)


if __name__ == "__main__":
    main()
