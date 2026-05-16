# Project context for Codex

This repository is a research prototype based on FastAvatar.

Research direction:
Text-guided digital human head motion control using MMHead motion clips and FastAvatar rendering.

Current validated pipeline:
MMHead facial motion pkl
→ text_motion/run_mmhead_to_motion.py
→ FastAvatar-readable motion sequence
→ FastAvatar inference/rendering

Current stage:
Build a retrieval-based text-to-motion codebook.

Important constraints:
- Do not modify core FastAvatar model/training code unless explicitly requested.
- Keep text_motion/run_mmhead_to_motion.py backward compatible.
- Do not commit local data, experiment outputs, third-party cloned repos, or build artifacts.
- First version should be simple and robust: keyword retrieval only, no FAISS, no sentence-transformers, no heavy dependencies.
- Prefer modular Python code with clear CLI scripts.

Expected first-version pipeline:
Text prompt
→ keyword retrieval over codebook searchable_text
→ top-k MMHead motion samples
→ top-1 raw MMHead motion
→ MMHead-to-FastAvatar conversion
→ metadata saved for FastAvatar rendering

Data assumptions:
MMHead root may contain:
- t2m_manifest.jsonl
- facial_motion/*.pkl
- text_annotations/action/*.txt
- text_annotations/detail_expression/*.txt
- text_annotations/detail_head_pose/*.txt
- text_annotations/emotion/*.txt
- text_annotations/emotion_scenario/*.txt

Code style:
- Use pathlib.
- Use argparse for CLI.
- Use JSONL for codebook and retrieval outputs.
- Avoid adding heavy dependencies.
- Handle missing files gracefully with warnings.
