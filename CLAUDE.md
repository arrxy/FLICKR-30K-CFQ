# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research pipeline for enhancing text queries and improving image-text retrieval on the Flickr30K dataset. The system uses vision-language models (CLIP, GroupViT, Align, CLIPSeg) for encoding and a local LLM (Vicuna-13b) for query expansion.

## Running Scripts

No build system exists — scripts are executed directly in pipeline order:

```bash
# 1. Prepare combined data
python combine/combineData.py

# 2. Generate image tags (requires LLaVA model)
python tag/tag.py
python tag/processTagResponses.py

# 3. Enhance queries (requires local Vicuna server on localhost:8081)
python enhance/enhance.py
python enhance/deal_raw_enhanced.py
python enhance/check.py

# 4. Encode images and text (requires GPU + HuggingFace models at ../../../models/)
python encoder/img_encoder.py
python encoder/raw_text_encoder.py
python encoder/enhanced_text_encoder.py

# 5. Build test splits and evaluate
python retrieval/make_test_data.py
python retrieval/retrieval.py
```

The `flickr30k/` directory mirrors the above pipeline but is scoped specifically to Flickr30K captions.

## Architecture

### Pipeline Stages

**combine/** → merges image-text pairs from multiple sources (sentences, similar sentences, fragments, phrases, tags), deduplicates using `phrase_map.json` / `sentence_map.json`, and produces `test_sen_to_ids.json`.

**tag/** → uses LLaVA to generate multi-word descriptive tags per image → `img_to_tags.json`.

**enhance/** → sends raw text queries to a local OpenAI-compatible API (Vicuna at `http://localhost:8081/v1`), parses the numbered-list LLM output, then validates each enhancement. Key files:
- `config.py` — all LLM prompts (enhancement + validation templates with `{query}`/`{text}` placeholders)
- `deal_raw_enhanced.py` — parses raw LLM list output into clean JSON arrays
- `check.py` — validates that enhanced text preserves original query intent

**encoder/** → produces `.pt` feature tensors for images and text using four models: `clip-vit-base-patch32`, `groupvit-gcc-yfcc`, `align-base`, `clipseg-rd64-refined`. Models are loaded from `../../../models/` (path relative to repo root).

**retrieval/** → loads feature tensors + 10-fold splits (`test_data_11.json`), runs four retrieval strategies, and prints recall@k broken down by text type (rawSentence, simSentence, fragment, phrase, tag). Strategies:
- `retrieval_no_enhance` — baseline cosine similarity on raw text
- `retrieval_enhance_v1` — counter-based voting across enhanced expansions
- `retrieval_enhance_v2` — average similarity across expansions
- `retrieval_enhance_v3` — two-stage retrieval (broad candidates, then refine)

### Key Data Flow

```
Raw Flickr30K data
  → combine/       : text-to-image-id mappings
  → tag/           : image-to-tag mappings
  → enhance/       : text-to-enhanced-list mappings
  → encoder/       : .pt feature files per model
  → retrieval/     : recall metrics
```

### Infrastructure Requirements

- **GPU**: CUDA required; some scripts hardcode `cuda:3`
- **LLM server**: Vicuna-13b must be running locally on port 8081 (OpenAI-compatible)
- **Models directory**: HuggingFace model checkpoints at `../../../models/` relative to repo root
- **tools.py**: Shared utility at repo root for JSON I/O, imported by multiple modules

## Important Caveats

- Paths are hardcoded with the original author's username (`/Users/aidan/...`); update these before running on a new machine.
- No requirements.txt exists; key dependencies are `torch`, `transformers`, `openai`, `llava`, `PIL`, `tqdm`.
- The `GPT/` subdirectory under `enhance/` contains alternative GPT-based enhancement variants that are not part of the main pipeline.
