# Errata Source Matching

## The problem

Each row is an erratum against a technical standard: a submitter note, a proposed
correction, and the original excerpt it refers to. I have to identify which
source document it came from, picking the title and year out of 16 candidates,
and attach a calibrated confidence to the choice. The metric weights getting the
exact title right at 0.85, so confidence is a rounding error next to simply being
correct.

## What I did

The constraints shaped this more than anything else. I have to train or fine tune
inside the submission script using only the provided data, with no internet at
runtime and no downloading pretrained weights, no TF-IDF as the predictive model,
and no looking the erratum up against a live source. So the whole thing has to be
learned end to end from what ships in the box.

That pushes the work into learning a representation that lines up erratum text
with source identity directly, rather than leaning on the surface term overlap
that a retrieval baseline would use.

## Layout

`solution.py` is the entry point. `approach.md` is the write up, `notes.md` is
the running log, and `research/` holds the failure analysis and probes that fed
the final design. Datasets are not committed.
