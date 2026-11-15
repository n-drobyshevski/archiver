"""`python -m archiver`: the same as the `archiver` command.

Needed where the generated `archiver` launcher cannot run, such as Windows with Smart App Control.
"""

from archiver.cli import app

app(prog_name="archiver")
