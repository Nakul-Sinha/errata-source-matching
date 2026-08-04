# Technical Standard Errata Provenance Cards

## Challenge
- Task: 16-way candidate ranking — map errata text to source standard title+year
- Metric: `0.85 * exact_title + 0.05 * exact_year + 0.10 * confidence_calibration` (higher better)
- AI baseline: 0.60 (target ≥ 0.51, ideally ≥ 0.60)
- Train: 5230 · Test: 1524 · no source-document overlap
- Test docs are newer than train (years: train ≤2013, test ≥2014) — temporal OOD
- Title overlap train answers ↔ test candidates ≈ 2% — cannot memorize titles
- Correct answer always exactly one of `reference_title_01..16` as `Title (YYYY)`
- Labels uniform across 16 slots
- Banned: live errata lookup, external LLM APIs at inference, hardcoded id→answer maps
- Allowed: local fine-tuning, retrieval over train, open-weight HF models, training inside script

## Environment
- Official: Kaggle Docker, A10G, ~1h (+grace), internet for HF/timm weights
- Must train/fine-tune inside submission script (platform guidebook)
- Paths: `./dataset/public/` → `./working/submission.csv`
- Local: RTX 4050 6GB; Kaggle API as `nakuls1nha` (ACCESS_TOKEN)

## Submission schema
`id, provenance_card, confidence`
- provenance_card JSON: `{"source_title":"...","source_year":"..."}` exact match to a candidate
- confidence in [0,100], discriminating (constant conf → cal=0)

## Hard constraints (user + challenge)
- No TF-IDF as predictive solution (not allowed unless challenge explicitly permits)
- No runtime internet / no downloading HF or other online weights during the run
- No external LLM APIs; no live errata lookup
- Must train/fine-tune inside the submission script on provided data only
- From-scratch neural ranking is the compliant path (same family as Meridian Ashes)

## Approach plan
1. Fit train-only tokenizer (word/BPE) on errata + reference titles
2. Train a from-scratch Transformer bi-encoder or cross-encoder with listwise CE over 16 candidates
3. Optional: feed stateless lexical overlap features into the neural head (not TF-IDF)
4. Time-based local CV (train ≤2010, val ≥2011) as OOD proxy — critical because test years ≥2014
5. Confidence = temperature-scaled top softmax; fit on holdout
6. Paths: `./dataset/public` → `./working/submission.csv`; offline Kaggle only if needed

## Local probes
- Token overlap time-holdout ~0.13; chance 0.0625
- Duplicate titles (diff years) in ~66% of time-val rows — year disambiguation is hard
- Official metric weights title at 0.85 — optimize title-group ranking first
- Coverage/phrase oracles ≈ title 0.22–0.26; unique-title overlap train↔time-val golds only ~12%
- Rich pairwise feature MLP (title-group loss, early stop) ≈ **title 0.366 / score 0.337** on year≥2011 holdout (best compliant so far)
- MiniLM CE alone collapses on time-OOD (~0); feature-gated blend still trails pure features unless CE is isolated to low-coverage rows
- Word/char encoder unfreeze often destroys OOD lexical signal — keep neural residual small or cascaded
- Kaggle research: `enable_internet=false`, pin `NvidiaTeslaT4` (P100 breaks current torch wheels)
- Repo: https://github.com/Nakul-Sinha/errata-provenance-cards

## Best local time-split score
- **0.3373** (title_rate 0.3658, both 0.3643, cal ~0.08) — rich feature MLP, epoch 1, seed0
- Validated ensemble (3 seeds): **0.3324**
- Split sensitivity: time≥2012 ≈0.340; time≥2011 ≈0.335; frozen DeBERTa+lex ≈0.33 (no gain)
- Neural MLM/CE residuals repeatedly destroy OOD lexical signal when mixed
- Target ≥0.55 still open; zero-overlap semantic cases (~58% of val) need compositional matching that does not erase the feature prior
- PRs: https://github.com/Nakul-Sinha/errata-provenance-cards/pull/1 (merged)
