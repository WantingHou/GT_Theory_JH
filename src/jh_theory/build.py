"""End-to-end builder for JH Figures 2 through 7."""

from __future__ import annotations

from pathlib import Path

from . import fig02, fig03, fig04, fig05, fig06, fig07

BUILDERS = (fig02.build, fig03.build, fig04.build, fig05.build, fig06.build, fig07.build)


def build_all(output: Path | str) -> tuple[Path, ...]:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    expected = {f"Fig{number:02d}.pdf" for number in range(2, 8)}
    existing = {path.name for path in directory.iterdir() if path.is_file()}
    collision = expected & existing
    if collision:
        raise FileExistsError(f"refusing to overwrite: {sorted(collision)}")
    paths = tuple(builder(directory) for builder in BUILDERS)
    if {path.name for path in paths} != expected:
        raise RuntimeError("figure inventory differs from Figures 2 through 7")
    return paths
