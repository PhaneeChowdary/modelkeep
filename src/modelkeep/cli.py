"""Command line: modelkeep check | fix | verify | describe."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import CURRENT, REPAIRABLE, CannotRepair, UnsafeFile, __version__, check, describe, find_models, fix, verify


def _files(paths: list[str]) -> list[str]:
    """A folder stands for all model files below it."""
    found = []
    for path in paths:
        found += [str(file) for file in find_models(path)] if Path(path).is_dir() else [path]
    return found


def _check(args) -> int:
    worst, reports = 0, []
    for file in _files(args.paths):
        try:
            report = check(file, trust=args.trust)
        except OSError as error:
            print(f"{file}\n  Cannot read the file: {error}", file=sys.stderr)
            worst = 2
            continue
        reports.append(report)
        worst = max(worst, 0 if report.status == CURRENT else 1 if report.status == REPAIRABLE else 2)
        if not args.json:
            print(report)
            if report.status == REPAIRABLE:
                print(f"  Next         modelkeep fix {file}")
    if args.json:
        print(json.dumps([report.as_dict() for report in reports], indent=1))
    return worst


def _fix(args) -> int:
    try:
        report = fix(args.file, args.output, trust=args.trust, overwrite=args.overwrite)
    except (CannotRepair, UnsafeFile, OSError, RuntimeError) as error:
        print(f"modelkeep: {error}", file=sys.stderr)
        return 2
    print(report)
    if not report.proof:
        print("  Saved. Test the new file with your data before you replace the old one.")
    return 0


def _verify(args) -> int:
    worst = 0
    for file in _files(args.paths):
        try:
            verdict = verify(file, tolerance=args.tolerance, trust=args.trust)
        except (OSError, RuntimeError) as error:
            print(f"{file}\n  Cannot check the file: {error}", file=sys.stderr)
            worst = 2
            continue
        print(verdict)
        worst = max(worst, 0 if verdict.ok else 1 if verdict.ok is None else 2)
    return worst


def _describe(args) -> int:
    try:
        print(describe(args.file, trust=args.trust))
    except OSError as error:
        print(f"modelkeep: {error}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modelkeep", description="Keep scikit-learn model files in a usable condition.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("check", help="say what is in a model file and if it loads")
    command.add_argument("paths", nargs="+", help="model files, or folders to search")
    command.add_argument("--json", action="store_true", help="print the reports as JSON")
    command.set_defaults(run=_check)

    command = commands.add_parser("fix", help="repair a model file and save it for the installed version")
    command.add_argument("file")
    command.add_argument("-o", "--output", help="name of the new file; the default adds .fixed to the name")
    command.add_argument("--overwrite", action="store_true", help="replace the new file if it exists")
    command.set_defaults(run=_fix)

    command = commands.add_parser("verify", help="compare the results of a model with its proof card")
    command.add_argument("paths", nargs="+", help="model files, or folders to search")
    command.add_argument("--tolerance", type=float, default=1e-8, help="largest accepted difference; default 1e-8")
    command.set_defaults(run=_verify)

    command = commands.add_parser("describe", help="show the parts of a model and the input that it expects")
    command.add_argument("file")
    command.set_defaults(run=_describe)

    for command in commands.choices.values():
        command.add_argument("--trust", action="store_true", help="also load code that is not from scikit-learn")
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
