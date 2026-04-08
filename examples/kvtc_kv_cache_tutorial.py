#!/usr/bin/env python3
# Copyright 2026 SGLang contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
KVTC + LMCache offline Engine tutorial

This script assumes it stays at ``examples/kvtc_kv_cache_tutorial.py`` under the
SGLang repo. The repository root is ``Path(__file__).resolve().parent.parent``.

You pass two config paths:

  * **LMCache (SGLang)**: YAML read via ``LMCACHE_CONFIG_FILE`` — CPU KV offload
    settings for LMCache.
  * **KVTC**: JSON read via ``KVTC_JSON_CONFIG_PATH`` — KV compression recipe
    (omit with ``--no-kvtc`` for LMCache-only).

**KVTC ``mini_examples`` JSON** (``kvtc/src/kvtc/integration/LMCache/configs/mini_examples/``) — quick reference:

- ``mini_kvtc_keys_without_rope_x20_noquant.json`` — default ``--kvtc-json``. Full KVTC (LRPCA + NVComp); ``NoQuant`` (no FP8 quant module); ``rope_overrides`` Neox; ``kvtc_worst_compression_rate`` 20.

- ``mini_kvtc_keys_without_rope_x20.json`` — same family; adds ``Quantization_kvtc-0`` (FP8): PCA + quant + NVComp; heavier, often smaller on disk.

- ``mini_kvtc_keys_with_rope_x20.json`` — x20 + quant; no top-level ``rope_overrides``; pair with the ``with_rope`` tensor layout / model family (not blindly interchangeable with ``without_rope``).

- ``mini_kvtc_keys_without_rope.json`` / ``mini_kvtc_keys_with_rope.json`` — **4×** tier (``kvtc_worst_compression_rate`` 4); ``without_rope`` has ``rope_overrides``, ``with_rope`` does not.

Example::

  python examples/kvtc_kv_cache_tutorial.py \\
    --lmcache-config python/sglang/srt/mem_cache/storage/lmcache/example_config.yaml \\
    --kvtc-json kvtc/src/kvtc/integration/LMCache/configs/mini_examples/mini_kvtc_keys_without_rope_x20_noquant.json

NVTX ranges are always emitted (``torch.cuda.nvtx``). To capture them with Nsight Systems::

  nsys profile -o kvtc_tutorial --trace=cuda,nvtx python examples/kvtc_kv_cache_tutorial.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import sglang as sgl
import torch
import torch.cuda.nvtx as nvtx
from transformers import AutoTokenizer

# Fixed layout: this file lives under ``examples/`` in the SGLang repo.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent


def setup_lmcache_env(lmcache_yaml: Path) -> Path:
    """
    Point LMCache at a YAML config and enable experimental LMCache features.

    Returns the resolved path to the YAML file used.
    """
    cfg = lmcache_yaml.expanduser().resolve()
    if not cfg.is_file():
        raise FileNotFoundError(f"LMCache config not found: {cfg}")
    os.environ["LMCACHE_CONFIG_FILE"] = str(cfg)
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    os.environ.setdefault("LMCACHE_LOG_LEVEL", "INFO")
    return cfg


def setup_kvtc_env(kvtc_json: Optional[Path]) -> None:
    """
    Set or clear ``KVTC_JSON_CONFIG_PATH``. Pass ``None`` to disable KVTC
    (LMCache CPU path without KVTC compression).
    """
    if kvtc_json is None:
        os.environ.pop("KVTC_JSON_CONFIG_PATH", None)
        return
    path = kvtc_json.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"KVTC JSON config not found: {path}")
    os.environ["KVTC_JSON_CONFIG_PATH"] = str(path)


def load_long_prompt(
    file_path: Optional[str],
    num_tokens: int,
    tokenizer,
    fallback_text: str,
) -> str:
    """
    Load text from a UTF-8 file or use ``fallback_text``, then truncate so the
    encoded length is at most ``num_tokens`` tokens.
    """
    if file_path:
        p = Path(file_path).expanduser().resolve()
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = fallback_text

    full_text = (
        "Read the passage below, then answer the follow-up question.\n\n"
        f"{content}\n\n"
    )
    token_ids = tokenizer.encode(full_text, add_special_tokens=False)
    total = len(token_ids)
    if total < num_tokens:
        print(
            f"[WARN] Text only tokenizes to {total} tokens "
            f"(requested {num_tokens}); using all tokens."
        )
    else:
        token_ids = token_ids[:num_tokens]

    prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"[INFO] Prompt tokens: {len(token_ids)} (raw text length before truncate: {total})")
    return prompt


def print_banner(title: str) -> None:
    """Print a section header for the tutorial narrative."""
    line = "=" * 72
    print(f"\n{line}\n  {title}\n{line}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the tutorial script."""
    default_lmcache = (
        _REPO_ROOT / "python/sglang/srt/mem_cache/storage/lmcache/example_config.yaml"
    )
    default_kvtc = (
        _REPO_ROOT
        / "kvtc/src/kvtc/integration/LMCache/configs/mini_examples"
        / "mini_kvtc_keys_without_rope_x20.json" #high acc
        # / "mini_kvtc_keys_without_rope_x20_noquant.json" #fast
    )

    parser = argparse.ArgumentParser(
        description="KVTC + LMCache tutorial: pass LMCache YAML and KVTC JSON paths."
    )
    parser.add_argument(
        "--lmcache-config",
        type=str,
        default=str(default_lmcache),
        help="Path to LMCache YAML (sets LMCACHE_CONFIG_FILE). "
        f"Default: repo {default_lmcache.name} under python/sglang/...",
    )
    parser.add_argument(
        "--kvtc-json",
        type=str,
        default=str(default_kvtc),
        help="Path to KVTC JSON (sets KVTC_JSON_CONFIG_PATH). "
        f"Default: mini example under kvtc/.... Ignored if --no-kvtc.",
    )
    parser.add_argument(
        "--no-kvtc",
        action="store_true",
        help="Do not set KVTC_JSON_CONFIG_PATH (LMCache only, no KVTC compression).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Optional UTF-8 text file for a long prefix. If omitted, a built-in sample is used.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=32000,
        help="Target number of prompt tokens.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate per request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_banner("Step 0 — Environment")
    lmcache_cfg = setup_lmcache_env(Path(args.lmcache_config))
    print(f"LMCache (SGLang) YAML -> LMCACHE_CONFIG_FILE={lmcache_cfg}")

    kvtc_path: Optional[Path] = None if args.no_kvtc else Path(args.kvtc_json)
    setup_kvtc_env(kvtc_path)
    if args.no_kvtc:
        print("KVTC: OFF (KVTC_JSON_CONFIG_PATH unset).")
    else:
        print(f"KVTC JSON -> KVTC_JSON_CONFIG_PATH={os.environ.get('KVTC_JSON_CONFIG_PATH')}")

    print_banner("Step 1 — Load tokenizer and build a long prefix")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    fallback = (
        "In a village of La Mancha, the name of which I have no desire to call to mind, "
        "there lived not long since one of those gentlemen that keep a lance in the lance-rack, "
        "an old buckler, a lean hack, and a greyhound for coursing. "
        * 400
    )
    novel_prompt = load_long_prompt(args.text_file, args.num_tokens, tokenizer, fallback)
    print(f"[INFO] Prefix character length: {len(novel_prompt)}")

    print_banner("Step 2 — Create SGLang Engine (LMCache on)")
    engine = sgl.Engine(
        model_path=args.model,
        disable_cuda_graph=False,
        enable_lmcache=True,
    )

    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }

    print_banner("Step 3 — First generation: long prefix + question (KV stored toward CPU cache)")
    prompts_r1 = [novel_prompt + " Question: Summarize the main theme of the passage."]
    with nvtx.range("Round1_store"):
        results1 = engine.generate(prompts_r1, sampling_params)
        torch.cuda.synchronize()
    for out in results1:
        print(f"Answer:\n{out['text']}\n")

    print_banner("Step 4 — flush_cache(): drop GPU KV so the next step must reload from CPU")
    time.sleep(1)
    with nvtx.range("Flush_cache"):
        ok = engine.flush_cache()
        print(f"flush_cache() -> {ok}")
        torch.cuda.synchronize()

    print_banner(
        "Step 5 — Second generation: same prefix, new question (reload / decompress from CPU)"
    )
    prompts_r2 = [novel_prompt + " Question: What literary device stands out most?"]
    with nvtx.range("Round2_load"):
        results2 = engine.generate(prompts_r2, sampling_params)
        torch.cuda.synchronize()
    for out in results2:
        print(f"Answer:\n{out['text']}\n")

    print_banner("Done — shutdown")
    engine.shutdown()
    print("Engine shut down cleanly. NVTX ranges: Round1_store, Flush_cache, Round2_load.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
