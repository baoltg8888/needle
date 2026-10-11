"""Convert the Markdown chapters in ../chapters to LaTeX in ./chapters.

Usage (from the repo root):
    python docs/thesis/latex/build.py
Needs pandoc >= 3.2 on PATH, or `pip install pypandoc_binary`.
Then compile main.tex (Overleaf: upload this folder plus ../figures as figures/).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "chapters"
OUTPUT = HERE / "chapters"


def pandoc_binary() -> str:
    found = shutil.which("pandoc")
    if found:
        return found
    import pypandoc  # type: ignore[import-not-found]

    return pypandoc.get_pandoc_path()


def main() -> None:
    pandoc = pandoc_binary()
    OUTPUT.mkdir(exist_ok=True)
    for markdown in sorted(SOURCE.glob("*.md")):
        target = OUTPUT / f"{markdown.stem}.tex"
        subprocess.run(
            [
                pandoc,
                str(markdown),
                "--from=markdown-implicit_figures",
                "--to=latex",
                "--top-level-division=chapter",
                "--syntax-highlighting=none",
                f"--lua-filter={HERE / 'thesis.lua'}",
                f"--output={target}",
            ],
            check=True,
        )
        print(f"{markdown.name} -> {target.relative_to(HERE)}")
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    for pdf in (HERE.parent / "figures").glob("*.pdf"):
        shutil.copy2(pdf, figures / pdf.name)
    print(f"copied figures -> {figures.relative_to(HERE)}")


if __name__ == "__main__":
    main()
