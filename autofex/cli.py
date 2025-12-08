"""
Command-line interface for AutoFE-X.
"""

import sys
import argparse
from typing import Optional

try:
    from . import __version__
except ImportError:
    __version__ = "unknown"


def version_command() -> None:
    """Print the version and exit."""
    print(f"autofex {__version__}")
    sys.exit(0)


def main(args: Optional[list] = None) -> None:
    """
    Main CLI entry point.

    Args:
        args: Command-line arguments (defaults to sys.argv)
    """
    parser = argparse.ArgumentParser(
        prog="autofex",
        description="AutoFE-X: Automated Feature Engineering + Data Profiling + Leakage Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    # Parse arguments
    parsed_args = parser.parse_args(args)

    if parsed_args.version:
        version_command()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

