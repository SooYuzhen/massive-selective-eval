# Installation and execution

All examples run from the release directory. `massive-eval` and `python -m massive_eval` are the same entry point. Commands refuse existing output files; use a new local output directory. No command automatically downloads data or weights.

## Minimal installation

```bash
python -m pip install -e .
massive-eval --help
massive-eval demo --out local_outputs/demo
python -m unittest discover -s tests -v
```

The minimal package needs NumPy/SciPy only. For a compatible preinstalled offline environment, use `PYTHONPATH=src python -m massive_eval ...`. The demo has no connection to the measured MASSIVE accuracy tables.

## Local data

Supply your own official MASSIVE 1.1 archive. The archive audit checks 52 locale members, the 1.1 changelog and Catalan marker; a local checksum fixes byte identity but is not a verified publisher checksum. Archive contents are read directly without extracting members to filesystem paths.

```bash
massive-eval audit --source data/amazon-massive-dataset-1.1.tar.gz --out local_outputs/raw_audit
massive-eval load --source data/amazon-massive-dataset-1.1.tar.gz --locale en-US --partition test --membership private/round10/split_membership.csv --output local_outputs/en_test.jsonl
```

The loader accepts existing JSONL, a directory of locale JSONL files, or Parquet with the original `id`, `locale`, `partition`, `utt`, `intent` schema. Parquet requires the optional extra `pip install -e '.[parquet]'`. Standalone files do not prove version identity; retain their source revision and hashes. Loading without the membership file does **not** reproduce the cleaned historical split. Records and annotation strings are preserved, and locale/ID duplicates fail rather than silently overwriting a join.

## Frozen local inference

Install `pip install -e '.[predict]'` for character models. The historical joblib bundles were generated with scikit-learn 1.7.2, so use that version. They are not included in this public candidate: vectorizers may encode training vocabulary, and this release avoids any model redistribution. SHA256 matching fixes identity but does not make an arbitrary pickle safe; load only models you trust.

```bash
massive-eval predict --records local_outputs/en_test.jsonl --model private/round10/models/char_lr_en-US.joblib --sha256 MODEL_SHA256 --model-name char_lr_en-US --trusted-local-model --output local_outputs/en_predictions.csv
```

For the shared head, install the `encoder` extra as well, pass `--model-name minilm_shared_lr` and `--encoder models/local-minilm-snapshot`. Use encoder revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`; embeddings are normalized, maximum length 128, batch size 64, CPU. The existing saved head is loaded without fitting. Tokenizer/model loading is local-only. This command produces a new inference output; it never replaces the historical frozen test files.

The bundle contains `classifier`, `vectorizer`, `classes` and selected `C`. Historical vectorizers' `common.normalize` is resolved to the identical reused normalization function. Supplying records without `intent` permits inference but leaves reference labels empty; evaluation correctly rejects absent reference labels.

## Frozen-threshold evaluation and curves

```bash
massive-eval evaluate --predictions private/round10/test_predictions.csv --thresholds private/round10/dev_thresholds.json --output local_outputs/selective.csv
massive-eval curve --predictions private/round10/test_predictions.csv --output local_outputs/risk_coverage.csv
massive-eval paired --predictions private/round10/test_predictions.csv --thresholds private/round10/dev_thresholds.json --output local_outputs/paired.csv
```

Prediction CSV columns: `model, locale, id, gold_intent, predicted_intent, confidence, margin, negative_entropy`. Threshold JSON is a list with `model, locale, strategy, target_dev_coverage, threshold, selected_on="dev"`. Max probability, top-two margin and normalized negative entropy increase with acceptance confidence. The package applies thresholds, never refits them on test. All ties are accepted.

Risk is errors among accepted predictions; zero accepted predictions have undefined risk. Curves use complete tie groups and right-step integration, and are descriptive, not a source of deployable test-fitted thresholds. Pairing uses source IDs, reports non-common exclusions and rejects conflicting reference intents. Model names `char_lr_<locale>` are combined into the character family; `minilm_shared_lr` is the shared family. The compact paired command reports event counts; uncertainty and paired model tests use the full replay command below.

## Full historical paired reanalysis

```bash
massive-eval replay-round11 --round10 private/round10 --round9 private/round9 --out local_outputs/round11_replay --bootstrap 5000
```

This reuses the Round 11 implementation verbatim. It requires the original Round 9/10 manifests, frozen probabilities, membership and model-hash records; those private inputs are not distributed. It validates hashes, all six ID sets, saved probabilities and 108 thresholds before computation. It never trains or regenerates model predictions. Outputs include source-ID bootstrap intervals, duplicate-component sensitivity, exact McNemar, Holm correction, risk differences, per-intent diagnostics and input identity. No new paper claim follows from passing this replay.

## Observed environment

Release checks: Python 3.12.14, NumPy 2.4.6, SciPy 1.18.0. Historical model inference: scikit-learn 1.7.2, sentence-transformers 5.6.1, torch 2.13.0, transformers 5.14.0. Core function tests do not require these optional inference packages or raw data. Broad dependency bounds are installability bounds, not a promise that every combination was tested; `audit/validation_results.json` names the executed environment and checks.

## Publishing safely

Keep data, model files, private replay outputs, local environment files and credentials outside the candidate tree. Run `python tools/audit_release.py` before packaging. The current release contains aggregate result tables only, with upstream attribution in the main README and license report. Follow the [publication checklist](PREPUBLICATION_CHECKLIST.md); nothing here creates a repository or pushes code.

After generating `selective.csv`, `paired.csv`, `curves.csv` and `round11_replay/` inside the same local replay directory, run `python tools/verify_frozen_outputs.py --round10 private/round10 --round11 private/round11 --replay local_outputs` to independently compare 172 historical numerical checks.
