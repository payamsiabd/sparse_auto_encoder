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
| 4.3 Silhouette scores, Fig. 3 | `rise/geometry.py::normalized_silhouette_scores`, `scripts/06_layer_sweep.py` |
| 4.4 Eq. 6, causal intervention | `rise/intervene.py::project_intervene`, `SteeringHook`, `generate_with_intervention` |
| Appendix D, LLM-judge + keyword annotation | `rise/annotate.py` (extended with a 4th class, `visual_reflection`) |
| 5. Eq. 7, unsupervised entropy-vector discovery | `rise/intervene.py::search_entropy_vector` |
| 4.2 training data (paper: 500 MATH examples) | `rise/mathvista.py` (this project: MathVista, a visual-reasoning benchmark) |

## Pipeline

```
scripts/00_download_models.py        # download Qwen3-VL-4B-Thinking -> models/
scripts/01_download_mathvista.py     # download MathVista -> train/ + test/ prompts.jsonl + images
scripts/02_generate_responses.py     # (image, question) -> CoT response, cached          [train split]
scripts/03_extract_activations.py    # response -> step-boundary activations per layer     [train split]
scripts/04_train_sae.py              # train the SAE on one layer's activations
scripts/05_annotate_and_visualize.py # label steps, reproduce Fig. 2/3, save behavior vectors
scripts/06_layer_sweep.py            # (optional) train+evaluate across layers, Fig. 3's curve
scripts/07_intervene_demo.py         # quick single-example steering sanity check
scripts/08_evaluate_on_test.py       # the actual result: classification + steering on the held-out test split
```

**Nothing here takes a required argument.** Every script reads
`configs/default.yaml`, whose defaults are all project-local paths
(`models/`, `data/mathvista/`, `runs/mathvista/`) that the download
scripts (00, 01) and each pipeline stage populate for the next one —
run them in order with no flags and it works. Every value is still
overridable with a dotted key if you want to, e.g.:

```bash
python scripts/02_generate_responses.py --generation.max_new_tokens 8192
```

**Environment note:** this pipeline was implemented and unit-tested
(synthetic SAE recovery, mocked image-token-expansion arithmetic, mocked
MathVista schema, mocked model-registry/association logic — see
[Testing](#testing)) in a sandbox whose network policy blocks
`huggingface.co` outright, so downloading the model, downloading
MathVista, and generating real model responses have **not** been run
end-to-end against the real dataset/model. Everything below is ready to
run as-is on a machine with normal Hugging Face access (and a GPU — see
[Requirements](#requirements)); if something in step 0/1 throws on a
live pull (most likely a MathVista column-name mismatch — the schema
was implemented from documentation, not a live pull), `rise/mathvista.py`'s
module docstring says exactly where to look.

### 0. Install and download everything

```bash
pip install -r requirements.txt
python scripts/00_download_models.py     # ~8-9GB, Qwen3-VL-4B-Thinking -> models/Qwen3-VL-4B-Thinking/
python scripts/01_download_mathvista.py  # -> data/mathvista/{train,test}/{prompts.jsonl,images/}
```

`00_download_models.py` snapshots every entry in `configs/default.yaml`'s
`models:` list (just Qwen3-VL-4B-Thinking by default) into a
project-local `models/` directory via `huggingface_hub.snapshot_download`
— safe to re-run if interrupted, only missing/changed files are
re-fetched. `model.model_id` already points at the resulting path, so
every later script loads it fully offline; `rise/models.py` is a small
registry (`rise.models.MODEL_REGISTRY`), so adding a second local model
later is a one-line addition, not a new code path.

`01_download_mathvista.py` pulls `AI4Math/MathVista`'s `testmini` split
(1,000 examples with public ground-truth answers; `test` (~5,100) has
answers withheld for leaderboard submission) from Hugging Face,
subsamples `mathvista.num_samples` (200 by default — raise it, or set
to `null` for the whole split, once you've confirmed the pipeline works
end-to-end), and splits the selection `mathvista.train_frac` (0.8) /
`1 - train_frac` into **train** (fit the SAE + behavior vectors on this)
and **test** (`scripts/08` measures results on this, and only this —
data neither the SAE nor the behavior vectors have seen). Every example
gets a MathVista-tailored system prompt
(`rise.mathvista.MATHVISTA_SYSTEM_PROMPT`) that explicitly asks the
model to look back at the image before using any detail read off it —
this is what makes `visual_reflection` steps actually show up often
enough to find.

MathVista is a good fit for this project's goal specifically: a large
fraction of its problems (chart/figure/table reading, geometry) require
pulling a precise value off the image, which gives a careful solver a
real reason to re-check it mid-reasoning — unlike generic VQA, where
one glance usually suffices. To concentrate on those problem types,
filter to `rise.mathvista.VISUAL_HEAVY_TASKS`:

```bash
python scripts/01_download_mathvista.py \
  --mathvista.task_filter "['chart question answering','figure question answering','table question answering','geometry problem solving']"
```

Want a different source dataset instead? Anything that can be written
to the same `prompts.jsonl` shape works — see `rise/dataset.py::load_prompts`
for the exact fields (`id`, `image`/`images`, `question`, optional
`system_prompt`/`answer`); `rise/mathvista.py::export_rows` is a
template for adapting another Hugging Face dataset the same way.

### 1. Generate responses and extract activations (train split)

```bash
python scripts/02_generate_responses.py
python scripts/03_extract_activations.py
```

Script 02 defaults to `generation.backend: "transformers"`
(`model.generate()`, one prompt at a time — always available, works on
CPU or GPU). On a GPU, switch to vLLM for much higher generation
throughput:

```bash
pip install vllm   # check Qwen3-VL support for your installed version first
python scripts/02_generate_responses.py --generation.backend vllm
```

vLLM only accelerates *this* step. It cannot replace script 03
(extracting hidden states at an arbitrary layer/position), the steering
in `scripts/07`/`08` (editing the residual stream mid-generation), or
`rise.intervene.search_entropy_vector` (needs gradients) — none of
those are things vLLM's inference-only engine exposes through its
public API, so they always run on `transformers` regardless of this
setting. Both backends write the same `responses.jsonl`, so script 03
onward can't tell (or care) which one produced it — see
`rise/vllm_backend.py`'s module docstring for the full reasoning.

**Troubleshooting `--generation.backend vllm`:** if you see `RuntimeError:
Cannot re-initialize CUDA in forked subprocess`, something touched CUDA
in the main process before vLLM's engine (which forks/spawns a worker
subprocess) got a chance to start — vLLM's worker needs to inherit a
completely uninitialized CUDA context. `rise.vllm_backend.load_vllm`
already guards against the known cause of this in this codebase (it
sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`, and `scripts/02` skips its
own CUDA-touching `set_seed()` call on the vLLM path) — if you still
hit it, something else in your environment (a custom launcher, an
earlier line in a notebook, etc.) is initializing CUDA first; move that
after generation, or run generation in its own process.

Script 03 re-feeds `(question, full_response)` through the model (via
`transformers`, always) in a single forward pass (no sampling) and
reads off `hidden_states[l]` at the token spanning each `"\n\n"` step
delimiter, for every layer in `activations.layers` — exactly Sec. 3.2's
construction, done for several layers at once so you don't have to
re-run the (slow) VLM forward pass per layer.

Qwen3-VL-specific detail: the processor expands each image placeholder
into many vision tokens, shifting every downstream text token's
position. `locate_step_tokens` in `rise/activations.py` computes that
shift exactly (rather than assuming a fixed image-token count) so the
returned positions are correct in the *actual* sequence the model
consumes — see that module's docstring for the full argument.

### 2. Train the SAE

```bash
python scripts/04_train_sae.py
```

Matches Sec. 4.2's hyperparameters by default (`D=2048`, batch 1024,
Adam lr=1e-4 with 10% warmup + cosine decay, λ=2e-3, on `sae.train_layer`
= layer 16). `||z||_0` in the paper's loss (Eq. 2) is non-differentiable;
as is standard for this SAE family (the paper itself cites Cunningham
et al., 2023 for the "standard SAE" recipe it uses), we optimize the L1
relaxation and log the true L0 as a diagnostic (`sae.sparsity_penalty:
l1` in the config; `l0` selects a literal straight-through estimator
instead). Decoder columns are kept exactly unit-norm throughout training
(gradient-projection + renormalize each step), matching the invariant
Sec. 4.4.1 relies on for interventions.

### 3. Annotate steps and inspect the decoder geometry

```bash
python scripts/05_annotate_and_visualize.py
```

Labels every cached (train-split) step `reflection` / `backtracking` /
`visual_reflection` / `others` (offline keyword annotator by default —
see `rise/annotate.py::KeywordAnnotator`; swap in an LLM judge with
`LLMJudgeAnnotator(call_fn)` for higher-quality labels, mirroring the
paper's GPT-5/GPT-4o/Claude judges and their reported >85% agreement
with keyword matching), then reproduces Fig. 2 (UMAP of decoder columns,
highlighted per behavior) and Fig. 3 (normalized silhouette scores) for
the layer the SAE was trained on. Also writes
`runs/mathvista/geometry/association_layer16.json` — which decoder
columns belong to which behavior, per `rise.geometry.ColumnAssociation`
— the artifact steps 4 and 5 below build behavior vectors from, so
nothing downstream needs to recompute it from the train activations.

Run `scripts/06_layer_sweep.py` to get the full per-layer curve (trains
one SAE per cached layer) and pick the layer where `visual_reflection`
is most separable from plain `reflection`/`backtracking` — the paper
finds mid-to-late layers consistently win (Fig. 3). If you change
`sae.train_layer` as a result, re-run steps 2-3 for that layer.

### 4. Sanity-check steering on one example

```bash
python scripts/07_intervene_demo.py
```

Builds a `visual_reflection` vector by averaging the decoder columns
most specifically associated with that label (filtering out columns
that also fire strongly for `reflection`/`backtracking`, per Sec.
4.4.1), then generates one training-split prompt under negative /
vanilla / positive intervention (`h' = h - α·w(wᵀh)`, Eq. 6) and reports
how the count of visual-reflection steps shifts — a quick, single-example
version of the causal-effect evidence in Fig. 5/6. This is a fast sanity
check, not the result — that's step 5.

### 5. Get results: evaluate on the held-out MathVista test split

```bash
python scripts/08_evaluate_on_test.py
```

This is the actual "does it work" measurement, run on data the SAE and
the behavior vectors never saw:

1. **Classification agreement.** For every held-out reasoning step,
   predicts its behavior label from its SAE code alone
   (`rise.geometry.predict_label`, using only the train-derived
   column→behavior association from step 3) and compares against the
   keyword annotator applied directly to that step's text. Writes a
   precision/recall/F1 report per label (`classification_report.json`)
   and every individual prediction (`step_predictions.jsonl`) to
   `runs/mathvista/evaluate/`. High agreement, especially on
   `visual_reflection`, is evidence the SAE learned a real, reusable
   direction rather than overfitting train-specific activations.
2. **Causal intervention effect.** Builds the same target behavior
   vector and steers generation on `evaluate.num_intervene_samples`
   (10 by default) held-out prompts, reporting the mean step count and
   mean target-behavior step count under negative/vanilla/positive
   intervention (`intervention_effect.json`) — the Fig. 5 style of
   evidence (effect averaged across several examples), computed on
   unseen data instead of the one example step 4 used.

## Using this for further analysis

1. **Classify any new step post-hoc**: encode its activation with
   `sae.encode(h)` and pass the code to `rise.geometry.predict_label`
   with the saved `association_layer*.json` — this gives you an
   unsupervised, per-step visual-reflection score for any trace you
   analyze later, without a fresh LLM judge call. This is exactly what
   `scripts/08_evaluate_on_test.py` does; reuse it directly.
2. **Validate against LLM-judge labels**: swap `KeywordAnnotator` for
   `LLMJudgeAnnotator` in step 3 (and in `scripts/08`) and compare
   agreement (`rise.annotate.agreement_ratio`) the way the paper
   validates its own annotator (Appendix D, Fig. 10) — high agreement is
   further evidence the discovered direction really is "visual
   reflection" and not a confound (e.g. response length, or generic
   uncertainty).
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
  fallback, task filtering + subsampling, and the train/test split --
  disjoint ids, each half self-contained, degenerate splits rejected)
  against an in-memory fake dataset shaped like MathVista's documented
  schema, and checks the output loads back correctly through
  `rise.dataset.load_prompts` -- everything except the actual network
  call to Hugging Face.
- `tests/test_models.py` validates `rise/models.py`'s registry
  resolution and the local-snapshot readiness check
  `rise.utils.load_qwen3_vl` uses to fail clearly instead of silently
  falling back to a hub download -- no network call.
- `tests/test_geometry.py` validates `rise.geometry.predict_label` (the
  post-hoc, activation-only behavior classifier `scripts/08` scores
  against annotations) and `save_association`/`load_association`'s
  round-trip, against a synthetic `ColumnAssociation`.
- `tests/test_evaluate_report.py` validates the precision/recall/F1/
  confusion-matrix computation in `scripts/08_evaluate_on_test.py`
  against known label sequences with a hand-computable answer.
- `tests/test_vllm_backend.py` validates `rise/vllm_backend.py`'s
  message-format conversion (this project's HF-style content items ->
  vLLM/OpenAI-style `image_url` data URIs, round-tripped back to a real
  image to confirm the encoding is correct), output parsing, and the
  batched-call shape, against a fake `vllm` module injected into
  `sys.modules` -- so it runs with no `vllm` install and no GPU, while
  still exercising the real conversion/parsing code `scripts/02` calls.

```bash
pip install torch numpy pillow pyyaml
for f in tests/test_*.py; do python "$f"; done
```

## Requirements

See `requirements.txt`. Qwen3-VL is a very recent model architecture;
you likely need a `transformers` build with Qwen3-VL support
(`pip install git+https://github.com/huggingface/transformers` if the
released version on PyPI doesn't have it yet). `huggingface_hub`
(used by `scripts/00_download_models.py`) is a transitive dependency of
`transformers`, so it's normally already present once that's installed.
`vllm` is optional, only needed for `--generation.backend vllm` — check
its release notes for Qwen3-VL support before relying on that path,
since it's a very recent architecture.
