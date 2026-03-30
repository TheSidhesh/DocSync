# DocSync: Agentic Documentation Maintenance via Critic-Guided Reflexion

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) <!-- TODO: Replace with actual arXiv ID -->

## Overview

DocSync is an agentic AI workflow designed to combat "documentation debt"—the pervasive issue where software documentation becomes inconsistent with the executable code as it evolves. [1] This drift creates technical debt, hinders maintainability, and can lead to critical API misuse.

While traditional static analysis tools can detect the *absence* of documentation, they cannot verify its *semantic correctness*. Large Language Models (LLMs) on their own often "hallucinate" or miss crucial details when not grounded in the code's structure.

DocSync addresses this by framing documentation maintenance as a structurally-aware, iterative task. It intelligently combines:
1.  **Abstract Syntax Tree (AST) Parsing**: To understand the structural reality of code changes (e.g., modified function signatures).
2.  **Retrieval-Augmented Generation (RAG)**: To pull in semantically relevant context from the existing documentation corpus.
3.  **Critic-Guided Reflexion**: An iterative self-correction loop where a critic model evaluates generated documentation for factual consistency against the source code, providing feedback for refinement. [1]

This repository contains a monolithic prototype of the DocSync framework, implemented as a self-contained Python script ready to run in Google Colab.

## Features

-   **Agentic Workflow:** Proactively monitors and updates documentation in response to simulated code changes.
-   **Structural Grounding:** Leverages Abstract Syntax Trees (AST) to extract concrete details like function signatures and class definitions, reducing hallucinations.
-   **Contextual Awareness:** Employs a FAISS-based Retrieval-Augmented Generation (RAG) system to find semantically similar documentation, providing rich context for updates.
-   **Critic-Guided Self-Correction:** Implements the Reflexion paradigm, where a critic model reviews and provides feedback on generated docstrings, which are then refined in a loop to improve factual consistency. [1]
-   **Resource-Efficient:** Built using a LoRA-adapted Small Language Model (`microsoft/Phi-3-mini-4k-instruct`) with 4-bit quantization, making it runnable on consumer-grade GPUs (e.g., Google Colab's T4).
-   **Comprehensive Evaluation:** Performance is benchmarked against a standard `CodeT5-base` model and a powerful `Gemini-2.5-Pro` Oracle across multiple metrics, including BLEU, BERTScore, and LLM-as-a-Judge.
-   **Reproducible & Self-Contained:** The `docsync.py` script handles dependency installation, data downloading, model training, evaluation, and results generation in a single run.

## Installation

The project is designed to be run in a Python 3.8+ environment like Google Colab. The script includes a function to automatically install all necessary dependencies.

```bash
pip install -q transformers datasets accelerate evaluate bert-score sentence-transformers faiss-cpu bitsandbytes tqdm seaborn matplotlib pandas sentencepiece google-generativeai
```

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TheSidhesh/DocSync.git
    cd DocSync
    ```

2.  **Set up API Keys (Optional but Recommended):**
    If you are using Google Colab, you can add your Gemini API key to the Colab Secrets Manager with the name `GOOGLE_API_KEY`. This is required for the LLM-as-a-Judge evaluation and the Oracle baseline.

3.  **Configure the run (Optional):**
    Parameters for the entire workflow are managed by the `Config` dataclass at the top of `docsync.py`. You can modify settings like:
    -   `base_model`: The generative model to be fine-tuned (default: `microsoft/Phi-3-mini-4k-instruct`).
    -   `train_samples`, `eval_samples`: The number of examples to use for training and evaluation.
    -   `use_peft`, `use_4bit`: Flags to enable/disable LoRA and 4-bit quantization.
    -   `enable_critic`, `use_gemini_judge`: Flags to control the Reflexion loop and the judge model.
    -   `enable_oracle`: Flag to run the powerful Gemini 2.5 Pro baseline.

4.  **Execute the script:**
    ```bash
    python docsync.py
    ```
    The script will perform all steps: data processing, retriever indexing, model training, baseline evaluation, DocSync evaluation, and finally, it will generate and save all output artifacts.

## Key Results

On a proxy documentation maintenance task using the CodeXGLUE dataset, DocSync demonstrates significant improvements over a standard baseline.

| Model | BLEU | BERTScore F1 | Summary Exact Match | Judge Score (1-5) |
| :--- | :---: | :---: | :---: | :---: |
| **DocSync (Final)** | 0.575 | **0.985** | **0.969** | 3.44 |
| DocSync (Initial) | 0.578 | 0.980 | 0.938 | 3.25 |
| CodeT5-base | 0.193 | 0.880 | 0.188 | 1.91 |
| *Oracle (Gemini-2.5-Pro)* | *0.138* | *0.868* | *0.031* | **4.13** |

-   **DocSync outperforms the baseline:** The final DocSync model substantially beats the `CodeT5-base` baseline across all semantic and faithfulness metrics, achieving a judge score of **3.44/5.0** compared to just **1.91**.
-   **The Reflexion loop adds value:** The critic-guided refinement process improves the judge score by **+0.19** from the initial generation to the final one, demonstrating its ability to enhance semantic correctness.
-   **Metrics matter:** The powerful Oracle model (`Gemini-2.5-Pro`) achieves the highest judge score (**4.13**) but the lowest BLEU score. This highlights the limitations of n-gram-based metrics and validates the use of semantic-focused evaluations like LLM-as-a-Judge.

## Output Artifacts

After a successful run, the `/content/outputs` directory (or the `results_dir` you configure) will contain:

-   `results.csv` & `results.tex`: Tables summarizing the quantitative evaluation results.
-   `loss.png`: A plot of the training loss curve.
-   `summary.txt`: A text summary of the run's configuration and key results.
-   `*.json` files: Raw predictions, cleaned outputs, and judge scores for the baseline, DocSync, and Oracle models (`baseline_outputs.json`, `docsync_outputs.json`, `oracle_outputs.json`).
-   `judge_cache.json` & `judge_debug.json`: Cached results and detailed logs from the LLM-as-a-Judge evaluation.
-   `docsync_model/`: The saved LoRA adapter for the trained DocSync model.

## Limitations

This is an early-stage research prototype with several limitations:
-   **Proxy Task:** The evaluation uses an artificial task (repairing truncated docstrings) from the CodeXGLUE dataset, not real-world repository history.
-   **Limited Scope:** Training is performed for a single epoch on Python code only.
-   **Metric Normalization:** Scores are calculated on cleaned docstring content, not the raw, end-to-end model output, to isolate semantic quality from formatting artifacts.
-   **No Execution:** The workflow does not yet verify functional correctness by executing code snippets (doctests) from the generated documentation.

## Future Work

Future research will focus on bridging the gap to a production-ready system:
-   **Historical Replay:** Evaluating the agent on real-world code evolution by replaying `git` commit histories.
-   **Execution-Based Verification:** Integrating `doctest` execution to provide a strong correctness signal for reinforcement learning.
-   **Multi-Modal Documentation:** Extending the agent with Vision-Language Models (VLMs) to understand and update diagrams (e.g., UML, architecture flows).
-   **Human-in-the-Loop (HITL):** Building a system to incorporate feedback from human maintainers to further align the agent's behavior.

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{badrinarayan2024docsync,
  title={DocSync: Agentic Documentation Maintenance via Critic-Guided Reflexion},
  author={Badrinarayan, Sidhesh and Parthasarathy, Adithya},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Authors

-   **Sidhesh Badrinarayan** (California, USA)
-   **Adithya Parthasarathy** (California, USA)

## License

This project is not explicitly licensed. For open-source use, a permissive license like MIT or Apache 2.0 would be appropriate.
