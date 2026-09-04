# RISE for Qwen3-VL: finding visual reflection points with a sparse autoencoder

This repository implements the **RISE** framework from *"Fantastic
Reasoning Behaviors and Where to Find Them: Unsupervised Discovery of
the Reasoning Process"* (Zhang et al., Google DeepMind, 2025), applied
to **Qwen3-VL-4B-Thinking**, with the pipeline extended so it can
answer a question the paper doesn't ask: *is a given reasoning step a
visual reflection point* — a step where the model revisits, re-examines,
or corrects its reading of the image — as opposed to a purely textual
/ symbolic reflection or backtracking step?

The paper trains a sparse autoencoder (SAE) on step-level residual-stream
activations of a text-only reasoning model (DeepSeek-R1-1.5B) and shows
that individual SAE decoder columns correspond to interpretable
reasoning behaviors (reflection, backtracking), which can then be used
to steer generation. Nothing in the method is text-specific — it only
needs "a representation per reasoning step" — so it transfers directly
to a vision-language thinking model, provided step-boundary activations
are extracted correctly through the model's image-token expansion. That
plumbing is the main engineering addition here (see `rise/activations.py`).

## How the code maps to the paper

| Paper (Sec.) | This code |
|---|---|
| 3.1 SAE, Eq. 1-2 | `rise/sae.py::SparseAutoencoder` |
| 3.2 Thought Representation Construction | `rise/activations.py`, `rise/utils.py::split_into_steps` |
| 4.2 Setup (D=2048, batch 1024, lr 1e-4, λ=2e-3, Adam+cosine) | `rise/train_sae.py::TrainConfig` defaults, `configs/default.yaml` |
| 4.3 Decoder geometry, UMAP, Fig. 2 | `rise/geometry.py::umap_projection`, `plot_decoder_geometry` |
| 4.3 Silhouette scores, Fig. 3 | `rise/geometry.py::normalized_silhouette_scores`, `scripts/05_layer_sweep.py` |
| 4.4 Eq. 6, causal intervention | `rise/intervene.py::project_intervene`, `SteeringHook`, `generate_with_intervention` |
| Appendix D, LLM-judge + keyword annotation | `rise/annotate.py` (extended with a 4th class, `visual_reflection`) |
| 5. Eq. 7, unsupervised entropy-vector discovery | `rise/intervene.py::search_entropy_vector` |
| 4.2 training data (paper: 500 MATH examples) | `rise/mathvista.py` (this project: MathVista, a visual-reasoning benchmark) |

## Pipeline

```
scripts/00_download_mathvista.py     # download MathVista -> prompts.jsonl + images
scripts/01_generate_responses.py     # (image, question) -> CoT response, cached
scripts/02_extract_activations.py    # response -> step-boundary activations per layer
scripts/03_train_sae.py              # train the SAE on one layer's activations
scripts/04_annotate_and_visualize.py # label steps, reproduce Fig. 2/3
scripts/05_layer_sweep.py            # train+evaluate across layers, reproduce Fig. 3's curve
scripts/06_intervene_demo.py         # build a behavior vector, steer generation, measure the effect
```

Every script reads `configs/default.yaml` and accepts dotted-key
overrides, e.g.:

```bash
python scripts/01_generate_responses.py \
  --data.prompts_jsonl data/my_prompts.jsonl \
  --generation.max_new_tokens 8192
```

**Environment note:** this pipeline was implemented and unit-tested
(synthetic SAE recovery, mocked image-token-expansion arithmetic,
mocked MathVista schema — see [Testing](#testing)) in a sandbox whose
network policy blocks `huggingface.co` outright, so downloading
MathVista, downloading Qwen3-VL-4B-Thinking's weights, and generating
real model responses have **not** been run end-to-end against the real
dataset/model. Everything below is ready to run as-is on a machine with
normal Hugging Face access (and a GPU — see [Requirements](#requirements));
if something in step 0/1 throws on a live pull (most likely a MathVista
column-name mismatch — the schema was implemented from documentation,
not a live pull), `rise/mathvista.py`'s module docstring says exactly
where to look.

### 0. Download MathVista

```bash
pip install datasets
python scripts/00_download_mathvista.py
```

Pulls `AI4Math/MathVista`'s `testmini` split (1,000 examples with public
ground-truth answers; `test` (~5,100) has answers withheld for
leaderboard submission) from Hugging Face, subsamples `mathvista.num_samples`
(200 by default — raise it, or set to `null` for the whole split, once
you've confirmed the pipeline works end-to-end), and writes
`data/mathvista/prompts.jsonl` + `data/mathvista/images/<pid>.png` in
the format `rise.dataset.load_prompts` expects. Every example gets a
MathVista-tailored system prompt (`rise.mathvista.MATHVISTA_SYSTEM_PROMPT`)
that explicitly asks the model to look back at the image before using
any detail read off it — this is what makes `visual_reflection` steps
actually show up often enough to find.

MathVista is a good fit for this project's goal specifically: a large
fraction of its problems (chart/figure/table reading, geometry) require
pulling a precise value off the image, which gives a careful solver a
real reason to re-check it mid-reasoning — unlike generic VQA, where
one glance usually suffices. To concentrate on those problem types,
filter to `rise.mathvista.VISUAL_HEAVY_TASKS`:

```bash
python scripts/00_download_mathvista.py \
  --mathvista.task_filter "['chart question answering','figure question answering','table question answering','geometry problem solving']"
```

Want a different source dataset instead? Anything that can be written
to the same `prompts.jsonl` shape works — see `rise/dataset.py::load_prompts`
for the exact fields (`id`, `image`/`images`, `question`, optional
`system_prompt`/`answer`); `rise/mathvista.py::export_rows` is a
template for adapting another Hugging Face dataset the same way.

### 1. Generate responses and extract activations

```bash
python scripts/01_generate_responses.py
python scripts/02_extract_activations.py
```

Stage 2 re-feeds `(question, full_response)` through the model in a
single forward pass (no sampling) and reads off `hidden_states[l]` at
the token spanning each `"\n\n"` step delimiter, for every layer in
`activations.layers` — exactly Sec. 3.2's construction, done for
several layers at once so you don't have to re-run the (slow) VLM
forward pass per layer.

Qwen3-VL-specific detail: the processor expands each image placeholder
into many vision tokens, shifting every downstream text token's
position. `locate_step_tokens` in `rise/activations.py` computes that
shift exactly (rather than assuming a fixed image-token count) so the
returned positions are correct in the *actual* sequence the model
consumes — see that module's docstring for the full argument.

### 2. Train the SAE

```bash
python scripts/03_train_sae.py --sae.train_layer 16
```

Matches Sec. 4.2's hyperparameters by default (`D=2048`, batch 1024,
Adam lr=1e-4 with 10% warmup + cosine decay, λ=2e-3). `||z||_0` in the
paper's loss (Eq. 2) is non-differentiable; as is standard for this SAE
family (the paper itself cites Cunningham et al., 2023 for the "standard
SAE" recipe it uses), we optimize the L1 relaxation and log the true L0
as a diagnostic (`sae.sparsity_penalty: l1` in the config; `l0` selects a
literal straight-through estimator instead). Decoder columns are kept
exactly unit-norm throughout training (gradient-projection + renormalize
each step), matching the invariant Sec. 4.4.1 relies on for interventions.

### 3. Annotate steps and inspect the decoder geometry

```bash
python scripts/04_annotate_and_visualize.py
```

Labels every cached step `reflection` / `backtracking` /
`visual_reflection` / `others` (offline keyword annotator by default —
see `rise/annotate.py::KeywordAnnotator`; swap in an LLM judge with
`LLMJudgeAnnotator(call_fn)` for higher-quality labels, mirroring the
paper's GPT-5/GPT-4o/Claude judges and their reported >85% agreement
with keyword matching), then reproduces Fig. 2 (UMAP of decoder columns,
highlighted per behavior) and Fig. 3 (normalized silhouette scores) for
the layer the SAE was trained on.

Run `scripts/05_layer_sweep.py` to get the full per-layer curve (trains
one SAE per cached layer) and pick the layer where `visual_reflection`
is most separable from plain `reflection`/`backtracking` — the paper
finds mid-to-late layers consistently win (Fig. 3).

### 4. Steer generation with the discovered vector

```bash
python scripts/06_intervene_demo.py --intervene.target_label visual_reflection
```

Builds a `visual_reflection` vector by averaging the decoder columns
most specifically associated with that label (filtering out columns
that also fire strongly for `reflection`/`backtracking`, per Sec.
4.4.1), then generates the same prompt under negative / vanilla /
positive intervention (`h' = h - α·w(wᵀh)`, Eq. 6) and reports how the
count of visual-reflection steps shifts — the causal-effect evidence
style of Fig. 5/6, specialized to visual grounding.

## Using this for your actual analysis goal

Once you have a trained SAE and annotations for a layer with good
`visual_reflection` separability (step 3/4 above):

1. **Classify any step post-hoc**: encode its activation with
   `sae.encode(h)` and check whether its top-firing latent(s) overlap
   with the `visual_reflection` column set from
   `associate_columns_with_behaviors` — this gives you an unsupervised,
   per-step visual-reflection score without needing a fresh LLM judge
   call for every trace you analyze later.
2. **Validate against the LLM-judge labels**: swap `KeywordAnnotator`
   for `LLMJudgeAnnotator` in step 3 and compare agreement
   (`rise.annotate.agreement_ratio`) the way the paper validates its
   own annotator (Appendix D, Fig. 10) — high agreement is your
   evidence the discovered direction really is "visual reflection" and
   not a confound (e.g. response length, or generic uncertainty).
3. **Discover unsupervised sub-behaviors**: run
   `rise.intervene.search_entropy_vector` (Eq. 7) restricted to steps
   already labeled `visual_reflection` as an alternative discovery
   objective, the way Sec. 5 discovers "confidence" — this can surface
   *why* a visual reflection happens (e.g. a "low-confidence-in-a-visual-
   detail" sub-direction) beyond the coarse label.

## Testing

None of these tests need the actual Qwen3-VL weights, MathVista itself,
or a GPU — they were all run in the sandbox this was authored in, which
had none of those available (see the environment note above).

- `tests/test_sae.py` validates the SAE implementation on synthetic data
  generated exactly per the paper's Theorem 1 setup (an incoherent
  random dictionary `W`, k-sparse codes, bounded noise): after training,
  the learned decoder columns should recover `W`'s columns up to
  permutation and scaling (cosine similarity ~1 to the best-matching
  true column).
- `tests/test_activations_locate.py` validates the trickiest piece of
  `rise/activations.py` -- correctly locating each `"\n\n"` step
  delimiter's token position *after* image-token expansion -- against a
  fake HF-processor-like tokenizer, so the position-shift arithmetic is
  checked without needing the real multimodal processor.
- `tests/test_mathvista.py` validates `rise/mathvista.py`'s conversion
  logic (image export, `prompts.jsonl` writing, `query`/`choices`
  fallback, task filtering + subsampling) against an in-memory fake
  dataset shaped like MathVista's documented schema, and checks the
  output loads back correctly through `rise.dataset.load_prompts` --
  everything except the actual network call to Hugging Face.

```bash
pip install torch numpy pillow
python tests/test_sae.py
python tests/test_activations_locate.py
python tests/test_mathvista.py
```

## Requirements

See `requirements.txt`. Qwen3-VL is a very recent model architecture;
you likely need a `transformers` build with Qwen3-VL support
(`pip install git+https://github.com/huggingface/transformers` if the
released version on PyPI doesn't have it yet).
