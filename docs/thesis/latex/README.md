# Thesis LaTeX build

The Markdown in `../chapters/` is the source. Everything in `chapters/` and `figures/` here is generated, so don't edit it by hand.

```bash
python docs/thesis/latex/build.py          # Markdown -> chapters/*.tex, copies figure PDFs
cd docs/thesis/latex && tectonic main.tex  # or latexmk -xelatex / pdflatex, or Overleaf
```

- `build.py` needs pandoc 3.2 or newer, either on PATH or via `pip install pypandoc_binary`.
- `thesis.lua` does these conversions:
  - strips the hand-written numbers (`## 3.4 …`) so LaTeX numbers sections itself
  - turns `**Figure 4.1.** …` and `**Table 4.1.** …` paragraphs into real captions
  - turns "Figure 4.1", "Table 4.1" and "Section 3.4" in the text into `\ref`
  - replaces Hangul in quoted HotpotQA passages with `[Hangul]`, because the default fonts have no Korean glyphs
- **Overleaf:** upload this folder, then compile with XeLaTeX or pdfLaTeX.
- **University template:** replace the preamble of `main.tex` with the template's, and keep the `\input{chapters/...}` lines.
- **Known gap:** `\ref{sec:5}` stays undefined until Chapter 5 is written.
