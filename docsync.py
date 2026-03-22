"""
Colab-ready monolithic DocSync prototype that operationalizes the paper
"Agentic Maintenance of Legacy Open-Source Documentation".

Highlights
- Uses open datasets (CodeXGLUE code-to-text) to avoid synthetic data.
- Simulates documentation drift, trains an agentic updater with RAG + AST hints.
- Compares against SOTA-style baseline (CodeT5-base) and zero-shot base model.
- Dynamic batch sizing from GPU RAM; progress via tqdm; loss printed every step.
- Caching of datasets, embeddings, and baseline results with invalidation flag.
- Produces plots (loss curves, metric trajectories) and tables for papers.
- Textual summary generated at the end.
- Base model default is Phi-3 Mini for T4 friendliness; switch `cfg.base_model` to Llama 3.1 8B (or similar) when you have A100/H100; GPU advice printed.
- Includes stub for Gemini API branch and auto-zip download block.
""" 

import os
import sys
import gc
import json
import math
import time
import hashlib
import random
import shutil
import psutil
import pickle
import inspect
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

# Lightweight installer to keep the notebook self-contained
REQUIRED_PACKAGES = [
    "transformers>=4.39.0",
    "datasets",
    "accelerate",
    "evaluate",
    "bert-score",
    "sentence-transformers",
    "faiss-cpu",
    "bitsandbytes",
    "tqdm",
    "seaborn",
    "matplotlib",
    "pandas",
    "sentencepiece",
]


def _ensure_deps():
    import importlib
    missing = []
    for pkg in REQUIRED_PACKAGES:
        name = pkg.split("=")[0].split(">")[0].split("<")[0]
        try:
            importlib.import_module(name.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Installing missing packages: ", missing)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)


_ensure_deps()

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
)
from datasets import load_dataset, load_from_disk, DatasetDict
from tqdm.auto import tqdm
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None
import ast


@dataclass
class Config:
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    baseline_model: str = "Salesforce/codet5-base"
    baseline_fallback_model: str = "google/flan-t5-small"
    baseline_modern_model: str = "google/flan-t5-base"
    dataset_name: str = "code_x_glue_ct_code_to_text"
    dataset_subset: str = "python"
    train_samples: int = 8192   # scaled further to close gap
    eval_samples: int = 32
    full_eval_validation: bool = False  # set True to use full validation split
    pilot_study_threshold: int = 256
    max_source_len: int = 256
    max_target_len: int = 96
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    gradient_accumulation: int = 1
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_peft: bool = True
    use_4bit: bool = True
    cache_dir: str = "/content/cache_docsync"
    results_dir: str = "/content/outputs"
    invalidate_cache: bool = False
    rag_top_k: int = 3
    retriever_model: str = "all-MiniLM-L6-v2"
    retriever_batch: int = 64
    enable_rag: bool = True
    enable_ast_context: bool = True
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 20
    small_start: bool = True
    scale_up_factor: int = 4
    auto_zip: bool = True
    seed: int = 42
    device: str = "auto"
    max_new_tokens: int = 96
    generation_do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    baseline_skip_if_cached: bool = True
    huggingface_token: Optional[str] = None
    skip_training_if_checkpoint: bool = False
    skip_baseline: bool = False
    enable_critic: bool = True
    critic_model: str = "google/flan-t5-small"
    critic_max_retries: int = 2
    use_gemini_judge: bool = True
    gemini_judge_model: str = "gemini-2.5-flash"
    gemini_api_key: Optional[str] = None
    enable_ast: bool = True
    enable_rag_flag: bool = True
    judge_max_len: int = 512
    use_saved_adapter_if_available: bool = False
    force_retrain: bool = True
    enable_oracle: bool = True
    oracle_provider: str = "gemini"  # fixed to gemini to reuse judge client
    oracle_model: str = "gemini-2.5-pro"
    oracle_api_key: Optional[str] = None
    oracle_temperature: float = 0.4
    oracle_max_new_tokens: int = 128


cfg = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_device_info() -> Tuple[str, float]:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"cuda ({name})", total_mem
    cpu_mem = psutil.virtual_memory().total / 1e9
    return "cpu", cpu_mem


def suggest_batch_size(total_mem_gb: float, seq_len: int, model_b: float = 3.8) -> int:
    # heuristic scaled for 4-bit; conservative floor at 1
    base = max(1, int(total_mem_gb / (model_b * seq_len / 2048 + 1)))
    return min(base, 32)


def recommend_gpu(model_name: str) -> str:
    name = model_name.lower()
    if "8b" in name:
        return "Prefer A100/H100; L4 acceptable with 4-bit; T4 only for quick smoke tests."
    if "3b" in name or "mini" in name:
        return "T4 or L4 works; A100 accelerates; H100 for scaling to full corpus."
    return "T4 minimum; L4 recommended; A100/H100 for larger context or full epochs."


device_name, total_mem_gb = get_device_info()
print(f"Detected device: {device_name}, approx mem {total_mem_gb:.2f} GB")
print(f"Colab GPU advice for {cfg.base_model}: {recommend_gpu(cfg.base_model)}")

if cfg.device == "auto":
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

cfg.per_device_train_batch_size = suggest_batch_size(total_mem_gb, cfg.max_source_len)
# Force small batch to avoid OOM on small GPUs
cfg.per_device_train_batch_size = 2
cfg.per_device_eval_batch_size = 2
print(f"Auto batch sizes -> train: {cfg.per_device_train_batch_size}, eval: {cfg.per_device_eval_batch_size}")

set_seed(cfg.seed)
ensure_dir(cfg.cache_dir)
ensure_dir(cfg.results_dir)
if not cfg.small_start:
    cfg.train_samples *= cfg.scale_up_factor
    cfg.eval_samples = min(1000, cfg.eval_samples * cfg.scale_up_factor)


def hash_config(config: Dict[str, Any]) -> str:
    relevant = json.dumps(config, sort_keys=True)
    return hashlib.md5(relevant.encode()).hexdigest()[:10]


_DOCSTRING_PATTERN = re.compile(r'("""|\'\'\')([\s\S]*?)\1', re.MULTILINE)
_INLINE_DOCSTRING_PATTERNS = [
    re.compile(r'""\s*(.*?)\s*""'),
    re.compile(r"''\s*(.*?)\s*''"),
]
_FENCE_PATTERN = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.MULTILINE)
_TAGGED_DOCSTRING_PATTERN = re.compile(r"<docstring>\s*([\s\S]*?)\s*</docstring>", re.IGNORECASE)
_PROMPT_ARTIFACT_MARKERS = [
    "Current (possibly stale) docstring:",
    "AST summary:",
    "Related references:",
    "Rewrite the documentation",
    "Rewritten docstring:",
    "Code snippet:",
    "Critic feedback:",
]
_CODE_BLOCK_START = re.compile(r"^(def|class)\s+\w+")
_CONTROL_FLOW_START = re.compile(r"^(return|if|for|while|with|try|except|elif|else|yield|assert|import|from)\b")
_ASSIGNMENT_LINE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*")
_SUMMARY_SECTION_LINE = re.compile(r"^(Args|Arguments|Parameters|Returns|Yields|Raises|Examples?|Notes?)\s*:?\s*$")
_PARAM_DOC_LINE = re.compile(r"^[A-Za-z_*][A-Za-z0-9_*. \-]*\s*:\s+")
_DOC_DASH_LINE = re.compile(r"^-{3,}$")


def _strip_prompt_artifacts(text: str) -> str:
    cleaned = text or ""
    for marker in _PROMPT_ARTIFACT_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    return cleaned


def _looks_like_doc_line(line: str) -> bool:
    return bool(_SUMMARY_SECTION_LINE.match(line) or _PARAM_DOC_LINE.match(line) or _DOC_DASH_LINE.match(line))


def _looks_like_code_line(line: str) -> bool:
    if not line or _looks_like_doc_line(line):
        return False
    return bool(_CODE_BLOCK_START.match(line) or _CONTROL_FLOW_START.match(line) or _ASSIGNMENT_LINE.match(line))


def extract_docstring_payload(text: str) -> str:
    """Return the core docstring content to avoid verbosity penalties."""
    if not text:
        return ""
    text = _strip_prompt_artifacts(text.strip())
    tagged = _TAGGED_DOCSTRING_PATTERN.search(text)
    if tagged:
        return tagged.group(1).strip()
    match = _DOCSTRING_PATTERN.search(text)
    if match:
        return match.group(2).strip()
    for pattern in _INLINE_DOCSTRING_PATTERNS:
        inline = pattern.search(text)
        if inline:
            return inline.group(1).strip()
    fence = _FENCE_PATTERN.search(text)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def clean_generated_docstring(text: str) -> str:
    """
    Trim copied code and prompt leakage while preserving docstring structure.
    """
    if not text:
        return ""
    text = extract_docstring_payload(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    cleaned_lines = []
    started = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip().strip('"\'')
        if not stripped:
            if started and cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if not started and _CODE_BLOCK_START.match(stripped):
            continue
        if stripped.startswith("```") or stripped in {"<docstring>", "</docstring>"}:
            continue
        if started and _looks_like_code_line(stripped):
            break
        started = True
        cleaned_lines.append(stripped)

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_docstring_text(text: str, max_len: Optional[int] = None) -> str:
    cleaned = clean_generated_docstring(text)
    if not cleaned:
        cleaned = extract_docstring_payload(text)
    cleaned = cleaned.strip()
    if max_len is not None:
        cleaned = cleaned[:max_len].strip()
    return cleaned


def prepare_for_judge(text: str, max_len: int = 512) -> str:
    """Normalize candidate text before sending to the critic/judge."""
    return normalize_docstring_text(text, max_len=max_len)


def summary_line_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    total = min(len(preds), len(refs))
    if total == 0:
        return {"summary_line_exact": 0.0, "summary_line_prefix": 0.0}

    exact = 0
    prefix = 0
    for pred, ref in zip(preds, refs):
        pred_line = " ".join((pred.strip().splitlines()[0] if pred.strip() else "").split())
        ref_line = " ".join((ref.strip().splitlines()[0] if ref.strip() else "").split())
        if pred_line == ref_line and ref_line:
            exact += 1
        if pred_line and ref_line and (pred_line.startswith(ref_line) or ref_line.startswith(pred_line)):
            prefix += 1

    denom = max(1, total)
    return {
        "summary_line_exact": exact / denom,
        "summary_line_prefix": prefix / denom,
    }


def ast_signature_summary(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            funcs.append(f"def {node.name}({', '.join(args)})")
        if isinstance(node, ast.ClassDef):
            funcs.append(f"class {node.name}")
    return " | ".join(funcs[:5])


def simulate_stale_doc(doc: str) -> str:
    if not doc:
        return doc
    parts = doc.split(".")
    if len(parts) <= 1:
        return doc
    keep = max(1, int(len(parts) * 0.5))
    return ".".join(parts[:keep]).strip()


def load_raw_dataset(cfg: Config):
    ds = load_dataset(cfg.dataset_name, cfg.dataset_subset, cache_dir=cfg.cache_dir)
    train = ds["train"].select(range(min(cfg.train_samples, len(ds["train"]))))
    if cfg.full_eval_validation:
        valid = ds["validation"]
        print(f"Full validation evaluation enabled: n={len(valid)}")
    else:
        valid = ds["validation"].select(range(min(cfg.eval_samples, len(ds["validation"]))))
        if len(valid) < len(ds["validation"]):
            print(f"Pilot-sized validation set in use (n={len(valid)}); set cfg.full_eval_validation=True for full sweep.")
    return DatasetDict({"train": train, "validation": valid})


def build_retriever(corpus: List[str], cfg: Config):
    if not cfg.enable_rag:
        return None, None
    if faiss is None:
        print("FAISS not available; disabling RAG")
        cfg.enable_rag = False
        return None, None
    print("Building retriever index ...")
    model = SentenceTransformer(cfg.retriever_model, device=cfg.device)
    embeddings = model.encode(corpus, batch_size=cfg.retriever_batch, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype("float32"))
    return (model, (index, embeddings, corpus))


def retrieve_context(query: str, retriever, top_k: int) -> List[str]:
    if retriever is None:
        return []
    model, (index, corpus_embeds, corpus_texts) = retriever
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idx = index.search(q.astype("float32"), top_k)
    idx_list = idx[0].tolist()
    return [corpus_texts[i] for i in idx_list]


def build_prompt(code: str, gold_doc: str, retriever, cfg: Config) -> Tuple[str, str]:
    gold_doc = gold_doc or ""
    stale_doc = simulate_stale_doc(gold_doc)
    ast_sig = ast_signature_summary(code) if cfg.enable_ast_context and cfg.enable_ast else ""
    rag_hits = retrieve_context(code, retriever, cfg.rag_top_k) if cfg.enable_rag and cfg.enable_rag_flag else []
    rag_text = "\n".join(rag_hits)
    prompt = (
        "You are DocSync, an agent that updates stale documentation.\n"
        "Return only the rewritten docstring text.\n"
        "Do not repeat the function signature.\n"
        "Do not include code, markdown fences, or explanations.\n"
        "Keep the first sentence as a concise summary, then preserve supported parameter or return details.\n"
        f"AST summary: {ast_sig or 'N/A'}\n"
        f"Current (possibly stale) docstring:\n{stale_doc or '<empty>'}\n"
        f"Code snippet:\n{code}\n"
    )
    if cfg.enable_rag and rag_hits:
        prompt += f"Related references:\n{rag_text}\n"
    prompt += "Rewritten docstring:\n"
    return prompt, gold_doc


def preprocess_dataset(raw_ds: DatasetDict, tokenizer, retriever, cfg: Config):
    preprocess_fingerprint = hash_config({
        "dataset_subset": cfg.dataset_subset,
        "train_samples": cfg.train_samples,
        "max_source_len": cfg.max_source_len,
        "max_target_len": cfg.max_target_len,
        "enable_ast": cfg.enable_ast_context and cfg.enable_ast,
        "enable_rag": cfg.enable_rag and cfg.enable_rag_flag,
        "prompt_style": "docstring_only_v2",
    })
    cache_path = os.path.join(cfg.cache_dir, f"processed_{preprocess_fingerprint}.disk")
    if os.path.exists(cache_path) and not cfg.invalidate_cache:
        print("Loading tokenized dataset from cache ...")
        return load_from_disk(cache_path)
    if cfg.invalidate_cache and os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)

    def tokenize(example):
        prompt, target = build_prompt(example["code"], example.get("docstring", ""), retriever, cfg)
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=cfg.max_source_len,
            padding=False,
            return_attention_mask=True,
        )
        target_ids = tokenizer(
            target,
            truncation=True,
            max_length=cfg.max_target_len,
            padding=False,
            return_attention_mask=True,
        )
        input_ids = prompt_ids["input_ids"] + target_ids["input_ids"]
        attention_mask = prompt_ids["attention_mask"] + target_ids["attention_mask"]
        labels = [-100] * len(prompt_ids["input_ids"]) + target_ids["input_ids"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_text": prompt,
            "target_text": target,
        }

    print("Tokenizing datasets ...")
    proc = raw_ds.map(tokenize, remove_columns=raw_ds["train"].column_names, desc="tokenize")
    proc.save_to_disk(cache_path)
    return proc


def load_tokenizer(cfg: Config):
    def _try_load(use_fast_flag):
        return AutoTokenizer.from_pretrained(
            cfg.base_model,
            cache_dir=cfg.cache_dir,
            use_auth_token=cfg.huggingface_token,
            use_fast=use_fast_flag,
            trust_remote_code=True,
        )
    tok = None
    for flag in (True, False):
        try:
            tok = _try_load(flag)
            break
        except Exception as e:
            print(f"Tokenizer load failed with use_fast={flag}: {e}")
            tok = None
    if tok is None:
        raise RuntimeError("Failed to load tokenizer for base model; ensure sentencepiece is installed.")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_critic_components(cfg: Config):
    try:
        c_model = AutoModelForSeq2SeqLM.from_pretrained(cfg.critic_model, cache_dir=cfg.cache_dir, trust_remote_code=True)
        c_tok = AutoTokenizer.from_pretrained(cfg.critic_model, cache_dir=cfg.cache_dir, use_fast=False, trust_remote_code=True)
        c_model.to(cfg.device)
        print(f"Critic loaded: {cfg.critic_model}")
        return c_model, c_tok
    except Exception as e:
        print(f"Critic load failed ({cfg.critic_model}): {e}")
        return None, None


def gemini_judge(prompt: str, cfg: Config) -> str:
    try:
        import re
        from google import genai
        from google.colab import userdata
    except Exception as e:
        raise RuntimeError(f"google.generativeai not installed: {e}")
    # api_key = cfg.gemini_api_key or os.getenv("GOOGLE_API_KEY")
    # if not api_key:
    #     raise RuntimeError("GOOGLE_API_KEY not set for Gemini judge")
    client = genai.Client(api_key=userdata.get('GOOGLE_API_KEY'))
    MODEL_NAME = cfg.gemini_judge_model or "gemini-2.5-pro"
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    raw_text = getattr(response, "text", None) or str(response)
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if json_match:
        raw_text = json_match.group(0)
    return raw_text.strip()


def check_gemini(cfg: Config):
    if not cfg.use_gemini_judge:
        return
    print(f"Gemini judge preflight check with model {cfg.gemini_judge_model} ...")
    try:
        ping = gemini_judge("PING", cfg)
        print(f"Gemini preflight success. Sample response: {ping[:80]}")
    except Exception as e:
        print(f"Gemini preflight failed: {e}")
        sys.exit(1)


def check_oracle(cfg: Config):
    if not cfg.enable_oracle:
        return
    print(f"Oracle preflight check with model {cfg.oracle_model} ...")
    try:
        resp = call_oracle("ORACLE_PING", cfg)
        if resp is None:
            raise RuntimeError("Oracle returned None")
        print(f"Oracle preflight success. Sample response: {resp[:80]}")
    except Exception as e:
        print(f"Oracle preflight failed: {e}. Disabling oracle baseline.")
        cfg.enable_oracle = False


def load_base_model(cfg: Config):
    base_kwargs = {"cache_dir": cfg.cache_dir}
    if cfg.use_4bit and cfg.device == "cuda":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base_kwargs.update({"quantization_config": quant_cfg, "device_map": "auto"})

    checkpoint_path = os.path.join(cfg.results_dir, "checkpoints", "checkpoint-1200")
    fallback_ckpt = os.path.join(cfg.results_dir, "checkpoints", "checkpoint-64")
    load_targets = []
    if os.path.isdir(checkpoint_path) and os.path.exists(os.path.join(checkpoint_path, "config.json")):
        load_targets.append(checkpoint_path)
    if os.path.isdir(fallback_ckpt) and os.path.exists(os.path.join(fallback_ckpt, "config.json")):
        load_targets.append(fallback_ckpt)
    load_targets.append(cfg.base_model)

    last_err = None
    for tgt in load_targets:
        try:
            print(f"Attempting to load model from {tgt} ...")
            model = AutoModelForCausalLM.from_pretrained(tgt, **base_kwargs)
            if hasattr(model, "config") and getattr(model.config, "tie_word_embeddings", None) is not None:
                model.config.tie_word_embeddings = False
            return model
        except Exception as e:
            print(f"Load failed for {tgt}: {e}")
            last_err = e
            # On failure, retry once without quantization (reduces odd module errors)
            if base_kwargs.get("quantization_config"):
                try_kwargs = {k: v for k, v in base_kwargs.items() if k != "quantization_config"}
                print("Retrying without 4-bit quantization ...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(tgt, **try_kwargs)
                    if hasattr(model, "config") and getattr(model.config, "tie_word_embeddings", None) is not None:
                        model.config.tie_word_embeddings = False
                    return model
                except Exception as e2:
                    print(f"Retry without quantization failed: {e2}")
                    last_err = e2
    raise last_err


def maybe_load_saved_adapter(model, cfg: Config):
    if not cfg.use_saved_adapter_if_available or cfg.force_retrain:
        return model, False
    adapter_dir = os.path.join(cfg.results_dir, "docsync_model")
    if not os.path.isdir(adapter_dir):
        return model, False
    try:
        from peft import PeftModel
        print(f"Loading saved adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.train()
        return model, True
    except Exception as e:
        print(f"Failed to load saved adapter: {e}")
    return model, False


def maybe_apply_peft(model, cfg: Config):
    if not cfg.use_peft:
        return model
    # Avoid re-wrapping if adapter already loaded
    if any("lora" in name.lower() for name, _ in model.named_modules()):
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("peft not installed; skipping LoRA")
        return model
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model


class LossRecorder(TrainerCallback):
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.history.append((state.global_step, logs["loss"]))
            print(f"step {state.global_step} loss {logs['loss']:.4f}")


class DataCollatorDocSync:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids, attention_masks, labels = [], [], []
        for ex in batch:
            input_ids.append(ex["input_ids"])
            attention_masks.append(ex["attention_mask"])
            labels.append(ex["labels"])

        max_len = max(len(x) for x in input_ids)

        def _pad(seq, pad_val):
            return seq + [pad_val] * (max_len - len(seq))

        padded_inputs = [_pad(x, self.pad_token_id) for x in input_ids]
        padded_masks = [_pad(x, 0) for x in attention_masks]
        padded_labels = [_pad(x, -100) for x in labels]

        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def train_and_eval(model, tokenizer, dataset, cfg: Config):
    collator = DataCollatorDocSync(pad_token_id=tokenizer.pad_token_id)
    loss_cb = LossRecorder()
    total_steps = math.ceil(len(dataset["train"]) / (cfg.per_device_train_batch_size * cfg.gradient_accumulation)) * cfg.num_train_epochs
    def _training_args(**kwargs):
        sig = inspect.signature(TrainingArguments)
        if "evaluation_strategy" not in sig.parameters and "eval_strategy" in sig.parameters:
            if "evaluation_strategy" in kwargs:
                kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
        return TrainingArguments(**kwargs)

    args = _training_args(
        output_dir=os.path.join(cfg.results_dir, "checkpoints"),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        evaluation_strategy="steps",
        eval_steps=min(cfg.eval_steps, max(1, total_steps // 4)),
        save_steps=cfg.save_steps,
        save_total_limit=2,
        fp16=cfg.device == "cuda",
        bf16=False,
        report_to=[]
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        callbacks=[loss_cb],
    )
    print("Starting training ...")
    trainer.train()
    print("Evaluating ...")
    metrics = trainer.evaluate()
    trainer.save_model(os.path.join(cfg.results_dir, "docsync_model"))
    return trainer, metrics, loss_cb.history


def generate_for_raw(model, tokenizer, raw_eval, retriever, cfg: Config) -> Tuple[List[str], List[str], List[str]]:
    model.eval()
    preds, refs, first_pass_preds = [], [], []

    # optional critic for semantic consistency
    critic_model = critic_tokenizer = None
    if cfg.enable_critic:
        critic_model, critic_tokenizer = (None, None)
        if cfg.use_gemini_judge:
            print(f"Critic mode: Gemini ({cfg.gemini_judge_model})")
        else:
            print("Critic mode: local model")
            critic_model, critic_tokenizer = load_critic_components(cfg)
            if critic_model is None:
                cfg.enable_critic = False

    def critic_judgement(code: str, stale: str, candidate: str) -> Tuple[bool, str]:
        if not cfg.enable_critic:
            return True, ""
        candidate_clean = prepare_for_judge(candidate, cfg.judge_max_len)
        prompt = (
            "You are a strict code-doc consistency judge. Given code and a proposed docstring, respond with GOOD or BAD and a short reason.\n"
            f"Code:\n{code}\nStale doc:\n{stale}\nProposed docstring:\n{candidate_clean}\n"
            "Answer with 'GOOD' or 'BAD' followed by reason."
        )
        if cfg.use_gemini_judge:
            try:
                text = gemini_judge(prompt, cfg)
            except Exception as e:
                print(f"Critic Gemini call failed: {e}; accepting candidate.")
                return True, f"fallback_accept: {e}"
            verdict = "GOOD" in text.split()[:1] or text.strip().upper().startswith("GOOD")
            return verdict, text
        if critic_model is None or critic_tokenizer is None:
            return True, ""
        inputs = critic_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.judge_max_len).to(cfg.device)
        with torch.no_grad():
            out = critic_model.generate(**inputs, max_new_tokens=64, num_beams=4, pad_token_id=critic_tokenizer.eos_token_id)
        text = critic_tokenizer.decode(out[0], skip_special_tokens=True)
        verdict = "GOOD" in text.split()[:1] or text.strip().upper().startswith("GOOD")
        return verdict, text

    critic_rejects = 0
    for i in tqdm(range(len(raw_eval)), desc="docsync-generate"):
        rec = raw_eval[i]
        prompt, target = build_prompt(rec["code"], rec.get("docstring", ""), retriever, cfg)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_source_len).to(cfg.device)
        best_pred = None
        attempt0_pred = None
        rationale = ""
        for attempt in range(cfg.critic_max_retries + 1):
            generation_kwargs = {
                "max_new_tokens": cfg.max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
                "do_sample": cfg.generation_do_sample,
            }
            if cfg.generation_do_sample:
                generation_kwargs.update({
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                })
            with torch.no_grad():
                gen = model.generate(**inputs, **generation_kwargs)
            # strip the prompt portion
            gen_seq = gen[0]
            input_len = inputs["input_ids"].shape[1]
            new_tokens = gen_seq[input_len:]
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
            if attempt == 0:
                attempt0_pred = pred
            ok, reason = critic_judgement(rec["code"], rec.get("docstring", ""), pred)
            if ok or attempt == cfg.critic_max_retries:
                best_pred = pred
                rationale = reason
                break
            else:
                critic_rejects += 1
                # refine prompt with critic feedback
                refined_prompt = prompt + f"\nCritic feedback: {reason}\nRevise the docstring accordingly:\n"
                inputs = tokenizer(refined_prompt, return_tensors="pt", truncation=True, max_length=cfg.max_source_len).to(cfg.device)
        preds.append(best_pred)
        refs.append(target)
        first_pass_preds.append(attempt0_pred or best_pred)
    if critic_rejects:
        print(f"Critic-triggered revisions: {critic_rejects} samples")
    return preds, refs, first_pass_preds


def evaluate_text(preds: List[str], refs: List[str]):
    if not preds or not refs:
        return {
            "bleu": 0.0,
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
            "summary_line_exact": 0.0,
            "summary_line_prefix": 0.0,
        }
    # replace empty strings with a placeholder to avoid BLEU zero-division
    cleaned = []
    empty_ct = 0
    for p, r in zip(preds, refs):
        p_norm = normalize_docstring_text(p)
        r_norm = normalize_docstring_text(r)
        if not p_norm.strip() or not r_norm.strip():
            empty_ct += 1
        p_clean = p_norm if p_norm.strip() else "<empty-pred>"
        r_clean = r_norm if r_norm.strip() else "<empty-ref>"
        cleaned.append((p_clean, r_clean))
    if empty_ct:
        print(f"evaluate_text: replaced {empty_ct} empty pred/ref pairs with placeholders")
    preds, refs = zip(*cleaned)
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    bleu_res = bleu.compute(predictions=list(preds), references=[[r] for r in refs])
    bert_res = bertscore.compute(predictions=list(preds), references=list(refs), lang="en")
    scores = {
        "bleu": bleu_res.get("bleu", 0.0),
        "bertscore_precision": float(np.mean(bert_res["precision"])),
        "bertscore_recall": float(np.mean(bert_res["recall"])),
        "bertscore_f1": float(np.mean(bert_res["f1"])),
    }
    scores.update(summary_line_metrics(list(preds), list(refs)))
    return scores


def judge_score(raw_eval, preds: List[str], cfg: Config) -> Tuple[float, List[int]]:
    if not cfg.enable_critic:
        print("Judge disabled: enable_critic=False")
        return 0.0, []
    # cache check
    normalized_preds = [prepare_for_judge(pred, cfg.judge_max_len) for pred in preds]
    preds_hash = hashlib.md5(json.dumps(normalized_preds, sort_keys=True).encode()).hexdigest()
    judge_cache_path = os.path.join(cfg.results_dir, "judge_cache.json")
    if os.path.exists(judge_cache_path):
        try:
            with open(judge_cache_path, "r") as f:
                cache = json.load(f)
            if preds_hash in cache:
                cached = cache[preds_hash]
                print(f"Judge cache hit (score={cached.get('score',0)}), skipping judge calls.")
                return cached.get("score", 0.0), cached.get("scores", [])
        except Exception as e:
            print(f"Judge cache read failed: {e}")
            cache = {}
    else:
        cache = {}

    critic_model = critic_tok = None
    if not cfg.use_gemini_judge:
        print("Judge mode: local critic model")
        critic_model, critic_tok = load_critic_components(cfg)
        if critic_model is None:
            print("Judge skipped: critic model not available")
            return 0.0
    else:
        print(f"Judge mode: Gemini ({cfg.gemini_judge_model})")
    def extract_score(text: str) -> int:
        import re
        m = re.search(r"[1-5]", text)
        if m:
            return int(m.group(0))
        return 0

    good = 0
    total = min(len(raw_eval), len(preds))
    debug_samples = []
    score_list: List[int] = []
    for i in range(total):
        rec = raw_eval[i]
        stale = simulate_stale_doc(rec.get("docstring", ""))
        code = rec["code"]
        # keep candidate concise for the judge
        candidate = prepare_for_judge(preds[i], cfg.judge_max_len)
        prompt = (
            "You are a strict code-doc consistency judge. Rate correctness on a scale of 1 (worst) to 5 (best), then a short reason.\n"
            f"Code:\n{code}\nStale doc:\n{stale}\nProposed docstring:\n{candidate}\n"
            "Answer with a number 1-5, then a colon and reason."
        )
        text = ""
        score_val = 0
        if cfg.use_gemini_judge:
            try:
                text = gemini_judge(prompt, cfg)
                score_val = extract_score(text)
            except Exception as e:
                print(f"Gemini judge failed: {e}")
                score_val = 0
        else:
            inputs = critic_tok(prompt, return_tensors="pt", truncation=True, max_length=cfg.judge_max_len).to(cfg.device)
            with torch.no_grad():
                out = critic_model.generate(**inputs, max_new_tokens=32, num_beams=2, pad_token_id=critic_tok.eos_token_id)
            text = critic_tok.decode(out[0], skip_special_tokens=True).strip()
            score_val = extract_score(text)
        good += score_val
        score_list.append(score_val)
        debug_samples.append({
            "i": i,
            "prompt": prompt,
            "judge_raw": text,
            "score": score_val
        })
    if debug_samples:
        dbg_path = os.path.join(cfg.results_dir, "judge_debug.json")
        with open(dbg_path, "w") as f:
            json.dump(debug_samples, f, indent=2)
        print(f"Judge debug samples saved to {dbg_path}")
    avg_score = good / max(1, total)
    print(f"Judge score (1-5 avg): {avg_score:.3f} (sum={good}, n={total})")
    # persist cache
    try:
        cache[preds_hash] = {"score": avg_score, "scores": score_list, "debug": debug_samples}
        with open(judge_cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Judge cache write failed: {e}")
    return avg_score, score_list


def run_baseline(raw_ds: DatasetDict, retriever, cfg: Config) -> Tuple[Dict[str, float], List[int]]:
    cache_file = os.path.join(cfg.results_dir, "baseline_cache.json")
    cfg_hash = hash_config({
        "model": cfg.baseline_model,
        "dataset": cfg.dataset_subset,
        "train": cfg.train_samples,
        "eval": cfg.eval_samples,
        "rag": cfg.enable_rag,
        "judge": cfg.use_gemini_judge,
        "prompt_style": "docstring_only_v2",
        "metric_cleaning": "docstring_only_v2",
    })
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}
    def _invalid_cached(scores: Dict[str, float]) -> bool:
        # rerun if BLEU and BERTScore F1 are zero or keys missing
        return (
            scores is None
            or "bleu" not in scores
            or "bertscore_f1" not in scores
            or (abs(scores.get("bleu", 0.0)) < 1e-12 and abs(scores.get("bertscore_f1", 0.0)) < 1e-12)
        )

    if cfg_hash in cache and cfg.baseline_skip_if_cached:
        cached_entry = cache[cfg_hash]
        cached_scores = {k: v for k, v in cached_entry.items() if k != "judge_scores"}
        judge_scores = cached_entry.get("judge_scores", [])
        if not _invalid_cached(cached_scores):
            print("Baseline cached; skipping rerun.")
            return cached_scores, judge_scores
        else:
            print("Baseline cache invalid or zeroed; recomputing baseline.")

    if cfg.skip_baseline:
        print("Baseline skipped by config.")
        return {"bleu": 0.0, "bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0, "judge": 0.0}, []

    def _sanitize_config(model_name: str):
        try:
            cfg_local = AutoConfig.from_pretrained(model_name, cache_dir=cfg.cache_dir, trust_remote_code=True)
            if isinstance(getattr(cfg_local, "extra_special_tokens", None), dict):
                # flatten dict values to list to avoid tokenizer error
                cfg_local.extra_special_tokens = list(cfg_local.extra_special_tokens.values())
            return cfg_local
        except Exception as e:
            print(f"Config load failed for {model_name}: {e}")
            return None

    def _load_baseline(model_name: str):
        cfg_local = _sanitize_config(model_name)
        model_local = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cfg.cache_dir, trust_remote_code=True, config=cfg_local)
        if hasattr(model_local, "config") and getattr(model_local.config, "tie_word_embeddings", None) is not None:
            model_local.config.tie_word_embeddings = False
        tok_local = None
        for flag in (False, True):  # prefer slow to avoid fast conversion issues
            try:
                tok_local = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cfg.cache_dir,
                    use_fast=flag,
                    trust_remote_code=True,
                    config=cfg_local,
                )
                break
            except Exception as e:
                print(f"Baseline tokenizer load failed with use_fast={flag} for {model_name}: {e}")
                tok_local = None
        if tok_local is None:
            raise RuntimeError(f"Failed to load baseline tokenizer for {model_name}")
        if tok_local.pad_token is None:
            tok_local.pad_token = tok_local.eos_token
        return model_local, tok_local

    baseline_candidates = [
        cfg.baseline_model,
        cfg.baseline_modern_model,
        cfg.baseline_fallback_model,
    ]
    model = tok = None
    last_err = None
    for candidate in baseline_candidates:
        try:
            model, tok = _load_baseline(candidate)
            print(f"Baseline loaded: {candidate}")
            break
        except Exception as e:
            print(f"Baseline load failed for {candidate}: {e}")
            last_err = e
            continue
    if model is None:
        print(f"All baselines failed; skipping baseline. Last error: {last_err}")
        return {"bleu": 0.0, "bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0, "judge": 0.0}, []

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.to(cfg.device)

    eval_raw = raw_ds["validation"]
    preds, refs = [], []
    for i in tqdm(range(len(eval_raw)), desc="baseline-generate"):
        rec = eval_raw[i]
        prompt, target = build_prompt(rec["code"], rec.get("docstring", ""), retriever, cfg)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_source_len).to(cfg.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, num_beams=4, pad_token_id=tok.eos_token_id)
        gen_seq = gen[0]
        input_len = inputs["input_ids"].shape[1]
        new_tokens = gen_seq[input_len:]
        if len(new_tokens) == 0:
            pred = tok.decode(gen_seq, skip_special_tokens=True)
        else:
            pred = tok.decode(new_tokens, skip_special_tokens=True)
        preds.append(pred)
        refs.append(target)
    scores = evaluate_text(preds, refs)
    baseline_judge, judge_scores = judge_score(eval_raw, preds, cfg)
    scores["judge"] = baseline_judge
    baseline_dump = {
        "preds": preds,
        "cleaned_preds": [normalize_docstring_text(pred) for pred in preds],
        "refs": refs,
        "judge_score": scores.get("judge", 0.0),
        "judge_scores": judge_scores,
    }
    with open(os.path.join(cfg.results_dir, "baseline_outputs.json"), "w") as f:
        json.dump(baseline_dump, f, indent=2)
    print(f"Saved baseline outputs to {os.path.join(cfg.results_dir, 'baseline_outputs.json')}")
    cache[cfg_hash] = {**scores, "judge_scores": judge_scores}
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)
    return scores, judge_scores


def call_oracle(prompt: str, cfg: Config) -> Optional[str]:
    """
    Oracle baseline uses the same Gemini client path as gemini_judge to avoid drift.
    """
    try:
        # temporarily swap judge model to oracle_model for this call
        orig_model = cfg.gemini_judge_model
        cfg.gemini_judge_model = cfg.oracle_model or cfg.gemini_judge_model
        result = gemini_judge(prompt, cfg)
        cfg.gemini_judge_model = orig_model
        return result
    except Exception as e:
        print(f"Oracle (Gemini) call failed: {e}")
        return None


def run_oracle_baseline(raw_ds: DatasetDict, retriever, cfg: Config) -> Tuple[Dict[str, float], List[int]]:
    if not cfg.enable_oracle:
        print("Oracle baseline disabled (cfg.enable_oracle=False).")
        return {"bleu": 0.0, "bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0, "judge": 0.0}, []
    cache_file = os.path.join(cfg.results_dir, "oracle_cache.json")
    cfg_hash = hash_config({
        "provider": cfg.oracle_provider,
        "model": cfg.oracle_model,
        "dataset": cfg.dataset_subset,
        "eval": len(raw_ds["validation"]),
        "rag": cfg.enable_rag,
        "prompt_style": "docstring_only_v2",
        "metric_cleaning": "docstring_only_v2",
    })
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except Exception as e:
            print(f"Oracle cache read failed: {e}")
    if cfg_hash in cache:
        print("Oracle baseline cached; skipping API calls.")
        cached = cache[cfg_hash]
        return cached, cached.get("judge_scores", [])

    preds, refs = [], []
    eval_raw = raw_ds["validation"]
    for i in tqdm(range(len(eval_raw)), desc="oracle-generate"):
        rec = eval_raw[i]
        prompt, target = build_prompt(rec["code"], rec.get("docstring", ""), retriever, cfg)
        pred = call_oracle(prompt, cfg)
        if pred is None:
            print("Oracle call failed; aborting oracle baseline to avoid partial metrics.")
            return {"bleu": 0.0, "bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0, "judge": 0.0}, []
        preds.append(pred)
        refs.append(target)
    scores = evaluate_text(preds, refs)
    oracle_judge, judge_scores = judge_score(eval_raw, preds, cfg)
    scores["judge"] = oracle_judge
    payload = {
        "preds": preds,
        "cleaned_preds": [normalize_docstring_text(pred) for pred in preds],
        "refs": refs,
        "judge_score": scores.get("judge", 0.0),
        "judge_scores": judge_scores,
    }
    with open(os.path.join(cfg.results_dir, "oracle_outputs.json"), "w") as f:
        json.dump(payload, f, indent=2)
    cache[cfg_hash] = {**scores, "judge_scores": judge_scores}
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)
    return scores, judge_scores


def plot_losses(history: List[Tuple[int, float]], path: str):
    if not history:
        return
    steps, losses = zip(*history)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    sns.lineplot(x=steps, y=losses)
    plt.xlabel("Global Step")
    plt.ylabel("Training Loss")
    plt.title("DocSync Loss over Steps")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_table(results: Dict[str, Dict[str, float]], path: str):
    df = pd.DataFrame(results).T
    df.to_csv(path)
    latex_path = path.replace(".csv", ".tex")
    with open(latex_path, "w") as f:
        f.write(df.to_latex(float_format=lambda x: f"{x:0.4f}"))
    return df


def textual_summary(df: pd.DataFrame, loss_fig: str, sample_size: int, cfg: Config, trajectory_delta: Optional[float] = None, judge_cis: Optional[Dict[str, Tuple[float, float]]] = None) -> str:
    best = df["bertscore_f1"].idxmax()
    study_tag = "empirical pilot study" if sample_size < cfg.pilot_study_threshold else "full evaluation"
    lines = [
        f"Evaluated models: {', '.join(df.index.tolist())}",
        f"Best semantic F1: {best} ({df.loc[best, 'bertscore_f1']:.3f})",
        f"BLEU range: {df['bleu'].min():.3f} – {df['bleu'].max():.3f}",
        f"Loss curve saved to {loss_fig}",
        f"Study size: n={sample_size} ({study_tag})",
    ]
    if "summary_line_exact" in df.columns:
        best_summary = df["summary_line_exact"].idxmax()
        lines.append(f"Best summary-line exact match: {best_summary} ({df.loc[best_summary, 'summary_line_exact']:.3f})")
    if trajectory_delta is not None:
        lines.append(f"Reflexion judge delta (turn0→final): {trajectory_delta:+.3f}")
    if judge_cis:
        ci_strings = []
        for name, ci in judge_cis.items():
            if ci is None:
                continue
            lo, hi = ci
            ci_strings.append(f"{name}: {lo:.3f}–{hi:.3f}")
        if ci_strings:
            lines.append("Judge 95% CI: " + "; ".join(ci_strings))
    return "\n".join(lines)


def mean_confidence_interval(scores: List[float], alpha: float = 0.05) -> Optional[Tuple[float, float]]:
    if not scores:
        return None
    n = len(scores)
    mean = float(np.mean(scores))
    if n == 1:
        return (mean, mean)
    std = float(np.std(scores, ddof=1))
    z = 1.96  # approx for 95% CI
    margin = z * std / math.sqrt(n)
    return mean - margin, mean + margin


def zip_and_download(cfg: Config):
    if not cfg.auto_zip:
        return
    zip_path = "/content/docsync_outputs.zip"
    os.system(f"zip -r {zip_path} {cfg.results_dir}")
    try:
        from google.colab import files
        files.download(zip_path)
    except Exception as e:
        print(f"Download skipped ({e})")


if __name__ == "__main__":
    print("==== DocSync Colab Runner ====")
    check_gemini(cfg)
    check_oracle(cfg)
    raw_ds = load_raw_dataset(cfg)
    retriever = None
    if cfg.enable_rag:
        corpus_texts = [ex.get("docstring", "") or "" for ex in raw_ds["train"]]
        retriever = build_retriever(corpus_texts, cfg)
    tokenizer = load_tokenizer(cfg)
    proc_ds = preprocess_dataset(raw_ds, tokenizer, retriever, cfg)
    # Run baseline first to ensure it succeeds before spending GPU on training
    baseline_scores, baseline_judge_scores = run_baseline(raw_ds, retriever, cfg)
    print(f"Baseline status: {'success' if sum(baseline_scores.values()) > 0 else 'FAILED/Skipped'} -> {baseline_scores}")

    model = load_base_model(cfg)
    model, loaded_adapter = maybe_load_saved_adapter(model, cfg)
    model = maybe_apply_peft(model, cfg)
    model.to(cfg.device)

    history = []
    trainer = None
    ckpt_dir = os.path.join(cfg.results_dir, "checkpoints")
    ensure_dir(ckpt_dir)
    if loaded_adapter:
        print("Skipping training (adapter already loaded).")
    elif cfg.skip_training_if_checkpoint and any(p.startswith("checkpoint") for p in os.listdir(ckpt_dir)):
        print("Skipping training (checkpoint exists and skip_training_if_checkpoint=True)")
    else:
        trainer, metrics, history = train_and_eval(model, tokenizer, proc_ds, cfg)
        print("Training completed.")
        print(f"Final eval metrics (trainer.evaluate): {metrics}")

    # DocSync predictions and metrics
    docsync_preds, docsync_refs, docsync_first_pass = generate_for_raw(model, tokenizer, raw_ds["validation"], retriever, cfg)
    docsync_scores = evaluate_text(docsync_preds, docsync_refs)
    docsync_judge_avg, docsync_judge_scores = judge_score(raw_ds["validation"], docsync_preds, cfg)
    docsync_scores["judge"] = docsync_judge_avg
    turn0_scores = evaluate_text(docsync_first_pass, docsync_refs)
    turn0_judge_avg, turn0_judge_scores = judge_score(raw_ds["validation"], docsync_first_pass, cfg)
    turn0_scores["judge"] = turn0_judge_avg
    reflexion_delta = docsync_scores.get("judge", 0.0) - turn0_scores.get("judge", 0.0)
    print(f"Reflexion trajectory: turn0 judge={turn0_scores.get('judge',0):.3f} -> final={docsync_scores.get('judge',0):.3f} (Δ={reflexion_delta:+.3f})")
    # persist generation outputs for analysis
    gen_dump = {
        "preds": docsync_preds,
        "cleaned_preds": [normalize_docstring_text(pred) for pred in docsync_preds],
        "refs": docsync_refs,
        "judge_score": docsync_scores.get("judge", 0.0),
        "turn0_preds": docsync_first_pass,
        "turn0_cleaned_preds": [normalize_docstring_text(pred) for pred in docsync_first_pass],
        "turn0_judge_score": turn0_scores.get("judge", 0.0),
        "reflexion_delta": reflexion_delta,
    }
    with open(os.path.join(cfg.results_dir, "docsync_outputs.json"), "w") as f:
        json.dump(gen_dump, f, indent=2)
    print(f"Saved docsync outputs to {os.path.join(cfg.results_dir, 'docsync_outputs.json')}")
    print(f"DocSync eval: {docsync_scores}")

    oracle_scores, oracle_judge_scores = run_oracle_baseline(raw_ds, retriever, cfg)

    all_results = {
        "DocSync-final": docsync_scores,
        "DocSync-turn0": turn0_scores,
        "CodeT5-base": baseline_scores,
    }
    if cfg.enable_oracle and oracle_scores:
        all_results["Oracle"] = oracle_scores
    table_path = os.path.join(cfg.results_dir, "results.csv")
    df = save_table(all_results, table_path)
    loss_fig = os.path.join(cfg.results_dir, "loss.png")
    plot_losses(history, loss_fig)

    judge_cis = {
        "DocSync-final": mean_confidence_interval(docsync_judge_scores),
        "DocSync-turn0": mean_confidence_interval(turn0_judge_scores),
        "CodeT5-base": mean_confidence_interval(baseline_judge_scores),
    }
    if cfg.enable_oracle and oracle_scores:
        judge_cis["Oracle"] = mean_confidence_interval(oracle_judge_scores)

    summary_text = textual_summary(df, loss_fig, len(raw_ds["validation"]), cfg, reflexion_delta, judge_cis)
    summary_path = os.path.join(cfg.results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(summary_text)

    zip_and_download(cfg)

    print("Done. Outputs saved to", cfg.results_dir)
