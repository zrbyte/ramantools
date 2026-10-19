"""``python -m ramantools`` entry point — delegates to :func:`ramantools.cli.main`."""
import sys  # exit status propagation

from .cli import main

# ``main`` returns the process exit status so it can be tested without
# a subprocess; here we hand that status to the interpreter.
sys.exit(main())
