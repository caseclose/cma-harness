# CMA-Harness — Project Page

Static project page for **Cognitive-structured Multimodal Agent for
Multimodal Understanding, Generation, and Editing** (CMA-Harness).

Live: <https://caseclose.github.io/cma-harness/>

## Contents

- `index.html` — landing page (hero, demo gallery, method + pipeline, benchmark, results, deployment, BibTeX)
- `styles.css` — theming (light / dark) and layout
- `assets/`
  - `paper.pdf` — compiled paper
  - `pipeline.png`, `data-pipeline.png`, `showcase.png`, `qualitative-case.png`, `case-preview.png` — figures
  - `demos/demo-{1..8}.mp4` + `demo-{1..8}.jpg` — 8 demo videos with fancy posters
  - `demos/_make_posters.py` — helper to regenerate poster frames from source videos
- `.nojekyll` — disable Jekyll so files under `_...` are served as-is

## Publish on GitHub Pages

1. Push this repo to `https://github.com/caseclose/cma-harness`.
2. Repo → **Settings → Pages**.
3. **Source**: *Deploy from a branch* · **Branch**: `main` · **Folder**: `/ (root)`.
4. First deploy takes ~30s; visit `https://caseclose.github.io/cma-harness/`.
