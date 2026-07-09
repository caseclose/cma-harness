# Cognitive-structured Multimodal Agent (CMA-Harness)

**for Multimodal Understanding, Generation, and Editing**

> Long-horizon multimodal memory, retrieval, generation, and editing — with a
> tool-augmented deployment harness (**CMA-Harness**).

[**Feng Wang**](https://github.com/caseclose)¹\*, Canmiao Fu², Zhipeng Huang², Chen Li², Jing LYU², Ge Li¹

¹Peking University  ²WeChat Vision, Tencent Inc.

<sub>\*Work done during an internship at WeChat Vision, Tencent Inc.</sub>

<p>
  <a href="https://caseclose.github.io/cma-harness/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-4f8cff?style=flat-square"></a>
  <a href="https://caseclose.github.io/cma-harness/#demo-gallery"><img alt="Demo" src="https://img.shields.io/badge/Demo-Gallery-8b5cf6?style=flat-square"></a>
  <a href="https://caseclose.github.io/cma-harness/#benchmark"><img alt="Benchmark" src="https://img.shields.io/badge/M2CA--Bench-2%2C000%20turns-22d3ee?style=flat-square"></a>
  <img alt="Code" src="https://img.shields.io/badge/Code-Coming%20Soon-f59e0b?style=flat-square">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-Coming%20Soon-f59e0b?style=flat-square">
</p>

**🌐 Project page → https://caseclose.github.io/cma-harness/**

---

## 🚧 Code & Dataset — Coming Soon

This repository will host the **official code and dataset** for the
Cognitive-structured Multimodal Agent and its deployment harness (**CMA-Harness**).

We are currently cleaning up the codebase and preparing the dataset for public
release. **The code, model configurations, and the M2CA-Bench dataset will be
released here.** Please **⭐ star / 👀 watch** this repository to be notified when
they become available.

### Release Roadmap

| Component | Description | Status |
| :--- | :--- | :---: |
| 📄 **Paper** | Full technical report | 🔜 Coming soon |
| 🧠 **Agent code** | PAE / EVM / CoRE / MEC inference pipeline | 🔜 Coming soon |
| 🏋️ **Training code** | Staged SFT + RL for memory construction & retrieval | 🔜 Coming soon |
| 🛠️ **CMA-Harness** | Tool-augmented deployment (17-tool registry, persistent memory, OpenAI-compatible serving) | 🔜 Coming soon |
| 📊 **M2CA-Bench** | 100 sessions × 20 turns (2,000 annotated turns) with retrieval labels + hard negatives | 🔜 Coming soon |
| 🧩 **Data engine** | Unified Scenario Engine for generating structured multi-turn dialogues | 🔜 Coming soon |
| 💾 **Checkpoints** | Trained 8B agent weights | 🔜 Under review |

> ⏳ Timelines are subject to internal review and open-source approval. This page
> will be updated as each component is released.

---

## TL;DR

We introduce a **memory-centric multimodal agent** that externalizes visual history
into **Episodic Visual Memory (EVM)**, selectively retrieves relevant visual
episodes, and plans understanding, generation, editing, and composition actions
through a **Multimodal Executive Controller (MEC)**. The same cognitive structure is
instantiated as **CMA-Harness**, a tool-augmented, multi-session deployment.

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
- **Selective cross-turn retrieval** — the *Cognitive Retrieval Engine (CoRE)*
  selects only the visual episodes relevant to the current user turn, improving
  grounding while reducing visual-token overhead.
- **Executive task control** — the *Multimodal Executive Controller (MEC)* infers
  whether a turn requires understanding, generation, editing, composition, or pure
  chat, then routes the task accordingly.
- **Training for memory use** — a *Unified Scenario Engine* generates structured
  multi-turn dialogues with retrieval annotations, enabling SFT and RL optimization
  for memory construction and retrieval.

## M2CA-Bench (to be released)

The **Multi-turn Context Agent Benchmark (M2CA-Bench)** is a held-out evaluation set
of **100 sessions × 20 turns (2,000 turns)** designed to stress-test long-horizon
multimodal grounding.

| 2,000 | 100 | 55 | 4 |
| :---: | :---: | :---: | :---: |
| evaluation turns | 20-turn sessions | topics × 8 domains | difficulty strata |

- **Structured scenario representation** — each turn is annotated as
  `(tᵢ, τᵢ, Rᵢ*, dᵢ, fᵢ)`: user input, task type, ground-truth retrieval set,
  difficulty, and challenge tags. Topics span **8 domains** with four task modes per
  topic — `generate`, `edit`, `cross-reference-edit`, `understand`.
- **Four difficulty strata** — stratified by topic shift, temporal span, multi-image
  interaction, and ambiguity (`easy` / `medium` / `hard` / `very_hard`).
- **Hard-negative design** — *high-similarity confounders* (near-duplicate images
  differing only in color, lighting, or material) and *negative retrieval samples*
  (semantic and structural negatives) block shortcut learning.
- **Three evaluation subsets** — retrieval accuracy is reported on **All / Long /
  Hard** cuts of increasing difficulty.

## Demos

Interactive multimodal sessions — search-driven generation, brand-fusion editing,
cross-reference composition, and long-horizon visual recall — are available on the
[live project page](https://caseclose.github.io/cma-harness/#demo-gallery).

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{wang2026cognitive,
  title   = {Cognitive-structured Multimodal Agent for Multimodal Understanding, Generation, and Editing},
  author  = {Wang, Feng and Fu, Canmiao and Huang, Zhipeng and Li, Chen and LYU, Jing and Li, Ge},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Contact

For questions about the paper, code, or dataset release, please open an
[issue](https://github.com/caseclose/cma-harness/issues).
