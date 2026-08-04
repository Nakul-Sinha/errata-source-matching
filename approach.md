# Approach: Technical Standard Errata Provenance Cards

## Problem
16-way ranking from errata text (`submitter_note`, `proposed_correction`, `original_excerpt`) to a provenance card `{"source_title","source_year"}` with calibrated confidence. Metric weights exact title at 0.85.

## Constraints
- Train/fine-tune inside the submission script on provided data only
- No runtime internet / no downloading pretrained weights
- No TF-IDF as the predictive solution
- No live errata lookup or external LLM APIs

## Method
1. **Rich pairwise features (OOD-stable)**  
   For each (query, candidate) pair: distinctive token coverage/weighted coverage, acronym hits, 2 to 3gram phrase hits, char-4gram overlap, year presence/proximity, containment, length-bucket hit counts. No corpus IDF.

2. **Title-group listwise MLP**  
   Small MLP scores the 16 candidates; loss is cross-entropy over title groups (logsumexp within duplicate titles) plus a light exact-candidate term for year. Early stopping on a year≥2011 holdout, longer training overfits train-era titles and hurts temporal OOD.

3. **Multi-seed ensemble + temperature**  
   Average logits across seeds; fit softmax temperature on the time holdout for confidence in `[0,100]`.

## Why not large neural encoders alone
Time-split probes show word/char Transformers and offline MiniLM cross-encoders collapse or underperform pure features when unfrozen (title memorization of train-era names). Lexical features transfer better under temporal OOD. From-scratch MLM/BERT residuals are being explored as gated add-ons that must not erase the feature prior.

## Local validation (time holdout year≥2011)
Best observed: **score ≈ 0.337** (title_rate ≈ 0.366). Ensemble validate run ≈ **0.332**. Target ≥0.55 remains open; further work focuses on compositional matching for zero-overlap semantic cases without destroying the lexical prior.

## Files
- `solution.py`: offline trainer/inferencer
- `working/submission.csv`: full-mode predictions
- `notes.md`: probe history and score tracking
