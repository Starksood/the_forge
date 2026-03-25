# ⚗️ The Forge

**Turn any document into instruction-tuning data — entirely offline.**

The Forge is a local Streamlit application that converts source documents (PDF, DOCX, TXT) into `{system, user, assistant}` conversation triples suitable for fine-tuning language models. It runs entirely on your machine using [Ollama](https://ollama.ai) — no API keys, no cloud, no data leaving your environment.

---

## How it works

The pipeline has five sequential phases:

```
Document → Concept Extraction → Semantic Chunking → Triple Generation → Review & Export
```

**Phase 0 — Concept Extraction**
The full document is segmented into 24k-char windows and analyzed by the LLM to extract a taxonomy of key concepts, entities, and relationships. These are synthesized into a 3-sentence `compressed_context` that is injected into every downstream generation prompt, grounding all triples in the document's actual subject matter.

**Phase 1 — Semantic Chunking**
The document is split into conceptual units using sentence embeddings (`nomic-embed-text`) and cosine similarity. Chunk boundaries are placed where semantic similarity drops below a configurable threshold — not at arbitrary character counts. This ensures each chunk contains a coherent idea rather than a mid-sentence fragment.

**Phase 2 — Triple Generation**
For each chunk, 10 triples are generated: 5 conversational perspectives × 2 response styles. Each triple is a realistic `{system, user, assistant}` exchange grounded in the chunk's content. The generation prompt instructs the model to construct a new scene — not summarize the source — applying the chunk's knowledge through the assistant's response.

**Phase 3 — Review**
A two-panel interface shows the source passage alongside its generated triples. Each triple displays inline quality violations, supports manual editing with auto-save, and can be regenerated individually. A batch repair pass rewrites flagged triples automatically.

**Export**
Approved triples are exported as JSONL in the standard chat format used by OpenAI fine-tuning, Axolotl, LLaMA-Factory, and most other fine-tuning frameworks.

---

## Perspectives and styles

The Forge generates triples across a matrix of conversational dimensions:

| Perspective | Description |
|---|---|
| **Introduction** | User encounters the concept for the first time |
| **Clarification** | User asks what something means or how it differs from related concepts |
| **Challenge** | User is skeptical or pushing back |
| **Acceptance** | User is beginning to engage — moving from resistance toward understanding |
| **Synthesis** | User connects this concept to their broader knowledge |

| Style | Description |
|---|---|
| **Terse** | Short, direct exchange — one-sentence question, concise answer |
| **Detailed** | Fuller exchange — user provides context, assistant gives a thorough response |

---

## Prerequisites

- **Python 3.9+**
- **Ollama** — [https://ollama.ai](https://ollama.ai)
- Two models pulled in Ollama:

```bash
ollama pull gemma3:4b        # generation and analysis
ollama pull nomic-embed-text # semantic chunking embeddings
```

The app is tested with `gemma3:4b`. Any instruction-following model available in Ollama will work. Set the model name in the sidebar settings at runtime.

### Context window

The Forge is designed for an **8K context window** (Ollama default). All prompts are budgeted to stay within this limit:

- Phase 0 analysis: ~1,500 chars fixed overhead + 24,000 chars document text = fits in 8K with `num_ctx=8192`
- Generation: ~4,000 chars fixed overhead + 3,000 chars passage = fits in 8K with `num_ctx=8192`

If you have a model with a larger context window, you can increase `MAX_SEG_CHARS` in `render_analyze_phase()` and the passage limit in `build_generation_prompt()` proportionally.

---

## Installation

```bash
git clone https://github.com/your-username/the-forge.git
cd the-forge
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

---

## Running

```bash
# Option 1 — shell script
bash run.sh

# Option 2 — direct
source venv/bin/activate
streamlit run datasetforge.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Project structure

```
the-forge/
├── datasetforge.py              # Streamlit app — all UI and orchestration
├── requirements.txt
├── run.sh
├── .streamlit/
│   └── config.toml              # Dark theme, port config
├── backend/
│   ├── personas.py              # Role definitions, angle/intensity constants
│   ├── ollama_client.py         # generate_sync() — raw Ollama /api/generate wrapper
│   └── phases/
│       ├── phase0_analysis.py   # CHUNKED_ANALYSIS_PROMPT, SYNTHESIS_PROMPT, call_ollama_blocking()
│       ├── phase2_generation.py # ANGLE_INTENSITY_COMBINATIONS (legacy reference)
│       └── semantic_chunker.py  # Embedding-based document segmentation
├── tests/
│   ├── test_personas.py
│   ├── test_validate_triple.py
│   ├── test_build_generation_prompt.py
│   ├── test_get_triples_for_chunk.py
│   ├── test_phase0_prompts.py
│   └── test_properties.py       # Hypothesis property-based tests
└── sessions/                    # SQLite session databases (gitignored)
```

---

## Database schema

Each session is an isolated SQLite database in `sessions/`. WAL mode is enabled for safe concurrent reads during the UI render loop.

```sql
documents          -- uploaded file text and metadata
framework_analyses -- taxonomy JSON, compressed_context, relationships
chunks             -- semantic segments with char ranges and page estimates
triples            -- generated {system, user, assistant} triples with status and tags
relationships      -- concept-to-concept edges from Phase 0
```

Triple status lifecycle: `pending` → `approved` / `rejected` / `needs_manual`

The `tags_json` column stores a JSON array of strings. The review UI supports filtering by tag. The repair pass adds a `repaired` tag automatically.

---

## Export format

Approved triples are exported as JSONL, one object per line, in the standard chat messages format:

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

This format is directly compatible with:
- OpenAI fine-tuning API
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) (`sharegpt` format with minor adapter)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Unsloth](https://github.com/unslothai/unsloth)

A `metadata.json` is also exported alongside the JSONL with per-angle and per-style breakdowns, duplicate pair detection results, and session provenance.

---

## Triple validation

Every generated triple is checked against these rules before being written to the database:

| Rule | Check |
|---|---|
| Required fields | `system`, `user`, `assistant` must all be non-empty |
| No meta-references | `user` must not contain phrases like "this passage", "the text", "the author" |
| First-person response | `assistant` must begin with "I " or contain a first-person present-tense construction |
| Length bounds | `assistant` must be between 40 and 600 characters |

Triples that fail validation are written with `status='needs_manual'` and flagged in the review UI. The batch repair pass rewrites them via a second LLM call with the violations listed explicitly.

---

## Configuration

All settings are adjusted in the sidebar at runtime and persisted in `settings/local_settings.json`. No restart required.

| Setting | Default | Description |
|---|---|---|
| Ollama host | `http://localhost:11434` | Ollama API endpoint |
| Model | `gemma3:4b` | Any model available in your Ollama instance |
| Chunk sensitivity | `0.75` | Cosine similarity threshold for chunk boundaries. Lower = more, smaller chunks |
| Cross-reference pass | on | Reserved for future cross-chunk triple generation |

---

## Running tests

```bash
source venv/bin/activate
pytest tests/test_personas.py tests/test_validate_triple.py \
       tests/test_build_generation_prompt.py tests/test_get_triples_for_chunk.py \
       tests/test_phase0_prompts.py tests/test_properties.py -v
```

The `test_properties.py` suite uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing. It verifies invariants like prompt structure, field presence, and truncation bounds across randomly generated inputs.

---

## Extending The Forge

**Adding a new perspective**
Add an entry to `ANGLE_DEFINITIONS` in `backend/personas.py`. The generation loop in `render_generate_phase()` reads directly from this dict — no other changes needed. The total triple count per chunk will increase automatically.

**Changing the generation model**
Set the model name in the sidebar. The model is passed through to both `call_ollama_blocking()` (Phase 0) and `_generate_triple_json()` (Phase 2) at call time.

**Adjusting context budget**
- Phase 0: change `MAX_SEG_CHARS` in `render_analyze_phase()` and `num_ctx` in `call_ollama_blocking()`
- Phase 2: change the passage slice `chunk_content[:3000]` in `build_generation_prompt()` and `num_ctx` in `_generate_triple_json()`

**Custom validation rules**
Add checks to `validate_triple()` in `datasetforge.py`. The function returns a list of violation strings — empty list means pass. It is called at generation time, repair time, and in the review card UI.

---

## License

MIT
