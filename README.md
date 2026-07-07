# Cognitive-structured Multimodal Agent

**for Multimodal Understanding, Generation, and Editing**

> Long-horizon multimodal memory, retrieval, generation, and editing.

[**Feng Wang**](https://github.com/caseclose)¹\*, Canmiao Fu², Zhipeng Huang², Chen Li², Jing LYU², Ge Li¹

¹Peking University  ²WeChat Vision, Tencent Inc.

<sub>\*Work done during an internship at WeChat Vision, Tencent Inc.</sub>

<p>
  <a href="https://caseclose.github.io/cma-harness/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-4f8cff?style=flat-square"></a>
  <a href="https://caseclose.github.io/cma-harness/#demo-gallery"><img alt="Demo" src="https://img.shields.io/badge/Demo-Gallery-8b5cf6?style=flat-square"></a>
  <a href="https://caseclose.github.io/cma-harness/#benchmark"><img alt="Benchmark" src="https://img.shields.io/badge/M2CA--Bench-2%2C000%20turns-22d3ee?style=flat-square"></a>
  <a href="https://caseclose.github.io/cma-harness/#bibtex"><img alt="BibTeX" src="https://img.shields.io/badge/Cite-BibTeX-13b8a6?style=flat-square"></a>
</p>

**🌐 Live project page → https://caseclose.github.io/cma-harness/**

---

## TL;DR

We introduce a **memory-centric multimodal agent** that externalizes visual history into
**Episodic Visual Memory**, selectively retrieves relevant visual episodes, and plans
understanding, generation, editing, and composition actions through a
**Multimodal Executive Controller (MEC)**.

## Key Results

| Metric | Value | What it measures |
| --- | --- | --- |
| **91.4%** | Retrieval accuracy | English retrieval over 20-turn sessions (All) |
| **89.4%** | Retrieval accuracy | Long subset (turns 11–20) |
| **82.0%** | Retrieval accuracy | Hard subset (`very_hard` @ turns 11–20) |
| **12.7 s** | Per-turn runtime | ~½ the 32B all-context baseline |
| **8.53 / 10** | Gemini quality score | Chinese overall generation quality |

## Method

A cognitive structure for long-horizon multimodal interaction:

- **Structured visual memory** — incoming and generated images are compressed into
  captions, tags, thumbnails, and metadata, so visual evidence persists without
  repeatedly occupying the model context window.
- **Selective cross-turn retrieval** — the *Cognitive Retrieval Engine* selects only
  the visual episodes relevant to the current user turn, improving grounding while
  reducing visual-token overhead.
- **Executive task control** — the *Multimodal Executive Controller* infers whether a
  turn requires understanding, generation, editing, composition, or pure chat, then
  routes the task accordingly.
- **Training for memory use** — a *Unified Scenario Engine* generates structured
  multi-turn dialogues with retrieval annotations, enabling SFT and RL optimization
  for memory construction and retrieval.

<p align="center">
  <img src="assets/pipeline.png" alt="Pipeline of the Cognitive-structured Multimodal Agent." width="820">
  <br><em>End-to-end pipeline: episodic visual memory, cognitive retrieval engine, and multimodal executive controller.</em>
</p>

## M2CA-Bench

The **Multi-turn Context Agent Benchmark (M2CA-Bench)** is a held-out evaluation set
of **100 sessions × 20 turns (2,000 turns)** designed to stress-test long-horizon
multimodal grounding.

| 2,000 | 100 | 55 | 4 |
| :---: | :---: | :---: | :---: |
| evaluation turns | 20-turn sessions | topics × 8 domains | difficulty strata |

- **Structured scenario representation** — each turn is annotated as
  `(tᵢ, τᵢ, Rᵢ*, dᵢ, fᵢ)`: user input, task type, ground-truth retrieval set,
  difficulty, and challenge tags. Topics span **8 domains** (commercial, industrial,
  educational, public service, hospitality, natural landscape, scientific, space)
  with four task modes per topic — `generate`, `edit`, `cross-reference-edit`,
  `understand`.
- **Four difficulty strata** — stratified by topic shift, temporal span, multi-image
  interaction, and ambiguity:
  - `easy` — same-topic, short-span references
  - `medium` — mid-range recall across turns
  - `hard` — long temporal spans or topic shifts
  - `very_hard` — multi-image comparison, fusion edits, ambiguous references
- **Hard-negative design** — to block shortcut learning we inject
  *high-similarity confounders* (near-duplicate images differing only in color,
  lighting, or material) and *negative retrieval samples* (semantic negatives that
  mention past images conversationally, structural negatives that explicitly request
  a new generation).
- **Three evaluation subsets** — retrieval accuracy is reported on
  **All / Long / Hard** cuts of increasing difficulty.

<p align="center">
  <img src="assets/data-pipeline.png" alt="Data pipeline for structured multi-turn scenario generation." width="820">
  <br><em>Unified Scenario Engine — a closed-loop pipeline that produces every M2CA-Bench session with turn-level retrieval supervision.</em>
</p>

## Demo Gallery

Eight interactive multimodal sessions covering search-driven generation, brand-fusion
editing, cross-reference composition, and long-horizon visual recall. Click through
the stacked cards on the [live project page](https://caseclose.github.io/cma-harness/#demo-gallery)
or open individual `.mp4` files under [`assets/demos/`](assets/demos/):

| # | Title |
| :---: | :--- |
| 1 | Brand Logo Fusion |
| 2 | Multi-image Cross-reference Edit |
| 3 | Long-horizon Visual Recall |
| 4 | Cross-modal Retrieval + Generation |
| 5 | Compositional Scene Editing |
| 6 | Style Transfer with Memory |
| 7 | Multi-turn Understanding |
| 8 | Real-time Web Search + Compose |

## CMA-Harness

The **harness** deploys the same cognitive structure to open-ended real-world
workflows:

- **Persistent multimodal memory** — session memories, user preferences, thumbnails,
  image cards, and gallery assets remain addressable across turns.
- **Tool-augmented action space** — web search, image retrieval, generation,
  editing, composition, deterministic collage, and inspection tools expand MEC's
  decisions.
- **Interactive execution loop** — tool results are appended back into the dialogue
  state, enabling iterative search, retrieval, generation, editing, and final
  response synthesis.

## Citation

```bibtex
@article{wang2026cognitive,
  title   = {Cognitive-structured Multimodal Agent for Multimodal Understanding, Generation, and Editing},
  author  = {Wang, Feng and Fu, Canmiao and Huang, Zhipeng and Li, Chen and LYU, Jing and Li, Ge},
  journal = {arXiv preprint},
  year    = {2026}
}
```
