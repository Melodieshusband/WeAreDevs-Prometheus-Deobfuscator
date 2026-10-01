import argparse
import sys
from pathlib import Path

from .pipeline import run


def main() -> None:
    sys.setrecursionlimit(20000)
    parser = argparse.ArgumentParser(
        prog="wearedevs-deobfuscator",
        description="Static deobfuscator for WeAreDevs / Prometheus-style Lua scripts.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="obfuscated .lua files")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output"), help="output directory")
    parser.add_argument("--no-blocks", action="store_true", help="skip the VM block report")
    args = parser.parse_args()
    for path in args.inputs:
        for written in run(path, args.output_dir, with_blocks=not args.no_blocks):
            print(written)


if __name__ == "__main__":
    main()
