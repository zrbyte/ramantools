"""Logging plumbing for the ramantools package.

The package historically reported progress with bare ``print`` calls,
which is fine in a notebook but invisible to anything that captures
output through the :mod:`logging` module (agents, batch scripts, test
harnesses). This module gives every message a logger to flow through
while keeping the default notebook experience unchanged:

* ``logger`` is the package logger (``logging.getLogger('ramantools')``).
  A :class:`logging.NullHandler` is attached so that, as the logging
  HOWTO recommends for libraries, importing ramantools never prints
  "No handlers could be found" warnings and never configures the root
  logger on the user's behalf.
* :func:`_say` is the single choke point for user-facing status text.
  It emits the message through ``logger`` **and** prints it to stdout,
  unless the caller has wired up logging, in which case stdout is left
  alone so the message is not shown twice.
* Pipeline methods additionally call ``logger.debug`` with their
  parameters. Those lines are silent by default and appear only when
  the user opts in, e.g. ``logging.basicConfig(level=logging.DEBUG)``.

Nothing here is public API; the one documented hook is the logger name
``'ramantools'``.
"""
import logging

# Package-wide logger. Child loggers are not used; one name keeps the
# user-facing configuration a single line.
logger = logging.getLogger('ramantools')
# NullHandler: swallow records when the application has not configured
# logging, instead of letting Python's "last resort" handler print
# WARNING+ records to stderr with no formatting.
logger.addHandler(logging.NullHandler())


def _logging_configured() -> bool:
        """Return ``True`` when a real handler would receive our records.

        Walks from the ``ramantools`` logger up through its ancestors
        (respecting ``propagate``) and looks for any handler that is not
        the :class:`logging.NullHandler` we attach ourselves. A ``True``
        here means the user has set up logging (``basicConfig`` or a
        custom handler), so :func:`_say` should not duplicate the message
        on stdout.
        """
        current = logger
        while current is not None:
                for handler in current.handlers:
                        if not isinstance(handler, logging.NullHandler):
                                return True
                # Stop climbing when propagation is switched off: records
                # would never reach the ancestors' handlers anyway.
                if not current.propagate:
                        break
                current = current.parent
        return False


def _say(message: str, level: int = logging.INFO) -> None:
        """Report ``message`` to the user exactly once.

        The record always goes to ``logger`` at ``level`` so a configured
        application captures it. Whether it is *also* printed follows
        three rules, checked in order:

        1. The user silenced the package explicitly
           (``logging.getLogger('ramantools').setLevel(logging.WARNING)``
           or higher): nothing is printed.
        2. A real handler is configured somewhere up the logger chain
           *and* the record passes the level filter: it was delivered
           through logging, so stdout is left alone (no double output).
        3. Otherwise — no logging set up at all, or set up at a level
           that drops INFO (``logging.basicConfig()`` with its WARNING
           default) — the message is printed, reproducing the historical
           ``print`` behaviour so it never silently disappears.
        """
        logger.log(level, message)
        # Rule 1: an explicit level on *our* logger is the user's opt-out.
        if logger.level != logging.NOTSET and not logger.isEnabledFor(level):
                return
        # Rule 2: delivered through a configured handler → done.
        if _logging_configured() and logger.isEnabledFor(level):
                return
        # Rule 3: fall back to stdout.
        print(message)
