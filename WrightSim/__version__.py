"""Define WrightSim version."""

import pathlib

__here__ = pathlib.Path(__file__).parent

__all__ = ["__version__", "__branch__"]

# read from VERSION file
with __here__ / "VERSION" as f:
    __version__ = f.read_text().strip()

# add git branch, if appropriate
p = __here__.parent / ".git" / "HEAD"
if p.is_file():
    with p.open() as f:
        __branch__ = f.readline().rstrip().split(r"/")[-1]
    if __branch__ != "master":
        __version__ += "-" + __branch__
else:
    __branch__ = None
