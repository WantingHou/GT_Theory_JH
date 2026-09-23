from __future__ import annotations

from pathlib import Path

from jh_theory import build_all


def test_build_all_figures(tmp_path: Path) -> None:
    paths = build_all(tmp_path)
    assert {path.name for path in paths} == {f"Fig{number:02d}.pdf" for number in range(2, 8)}
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 10_000
        assert path.read_bytes().startswith(b"%PDF-")


def test_builder_refuses_to_overwrite(tmp_path: Path) -> None:
    (tmp_path / "Fig02.pdf").write_bytes(b"existing")
    try:
        build_all(tmp_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("builder overwrote an existing figure")
