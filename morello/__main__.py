"""Lets `python -m morello` run the command line."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
