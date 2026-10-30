# Multilingual Selective Prediction and Reliability Evaluation

A reproducible Python portfolio project for **MASSIVE 1.1 intent classification** in English, Simplified Chinese and Japanese. It provides local data loading, frozen-model inference, dev-threshold selective evaluation, risk–coverage curves, source-ID pairing and clustered uncertainty analysis.

**Engineering: GO. The independent MASSIVE paper route: DROP.** This is not a validated LLM-as-a-Judge project and does not claim a novel selective-prediction method. A separate local Judge feasibility experiment is deliberately outside this release.

[中文简介](README.zh-CN.md) · [Installation and commands](docs/RUNNING.md) · [Licenses and exclusions](docs/LICENSE_AND_SECURITY.md) · [Publication checklist](docs/PREPUBLICATION_CHECKLIST.md)

## Measured historical results

The historical experiment retained the same 10,499 train, 1,886 dev and 2,847 test source IDs in each language after source-ID/text-component leakage controls. Both systems used the same test IDs. There are 17,082 frozen prediction rows and 108 dev-frozen thresholds: two systems × three languages × three scores × six target coverages.

| System | English accuracy | Chinese accuracy | Japanese accuracy |
| --- | ---: | ---: | ---: |
| Per-language character TF-IDF + logistic regression | 83.81% | 82.09% | 82.40% |
| Frozen multilingual MiniLM + shared logistic head | 84.37% | 80.65% | 80.72% |

At the previously reported margin score / 80% target dev coverage:

| System | English realized coverage / risk | Chinese realized coverage / risk | Japanese realized coverage / risk |
| --- | ---: | ---: | ---: |
| Character TF-IDF + LR | 80.30% / 7.61% | 81.74% / 9.67% | 79.73% / 7.71% |
| Frozen MiniLM + shared LR | 80.19% / 6.92% | 80.96% / 10.20% | 80.15% / 10.12% |

All scores measure agreement with existing MASSIVE intent labels, not semantic localization quality. Three-language prediction agreement is 2,100/2,847 (73.76%) for character models and 2,187/2,847 (76.82%) for the shared head. At the operating point above, acceptance decisions differ across languages on 838/2,847 and 831/2,847 IDs. These are descriptive observations, not proof of innovation.

The paired reanalysis used 5,000 source-ID bootstrap replicates and a sensitivity analysis with 2,707 text/source connected components. All three model-accuracy comparisons have Holm-adjusted McNemar p-values above 0.05; this is not evidence of equivalence. [Aggregate result tables](results/) retain exact values, denominators and intervals. Historical training and test timings are in `results/cost_by_language.csv`; shared-encoder language timings are allocated from a mixed batch, not independently measured per-language latency.

## Quick start without data or model weights

Python 3.10+ is required. From this directory:

```bash
python -m venv .venv
# Activate this environment using your shell's usual command.
python -m pip install -e .
massive-eval demo --out local_outputs/demo
python -m unittest discover -s tests -v
```

For an existing offline environment with NumPy and SciPy:

```bash
PYTHONPATH=src python -m massive_eval demo --out local_outputs/offline_demo
PYTHONPATH=src python -m unittest discover -s tests -v
```

The demo uses **synthetic software fixtures**, no downloaded corpus, no model training and no human evaluation labels. It tests metric plumbing, not model performance. The recorded release checks use the existing offline environment listed in `audit/validation_results.json`; optional inference dependencies and historical versions are documented separately.

## Unified command line

- `load`: stream local tar/JSONL/Parquet data, optionally applying the frozen membership file.
- `audit`: run the original read-only MASSIVE 1.1 archive audit.
- `predict`: use a supplied, hash-verified trusted local character head or frozen encoder/head.
- `evaluate`: apply existing dev thresholds, including ties, with explicit counts.
- `curve`: generate descriptive score-ranked risk–coverage tables; no test threshold fitting.
- `paired`: compare languages using source IDs and report exclusions and decision events.
- `replay-round11`: reuse the complete historical paired/clustered-bootstrap implementation when the private Round 9/10 assets are supplied.

See [RUNNING.md](docs/RUNNING.md) for complete commands and input contracts. There is no automatic data/model download, training command, credential requirement or network inference in the package. Historical predictions and weights are deliberately not bundled. Aggregate tables can be inspected immediately; exact full reanalysis requires separately supplied original frozen artifacts.

## Evidence and limitations

The core metric functions, raw audit and full paired reanalysis are reused from the completed project, with source hashes in `audit/code_provenance.json`. Thin loading/inference/CLI adapters are new. Release validation replays all 108 operating points and all historical paired summary tables against the frozen artifacts; it does not overwrite them or retrain models.

- The 168 original cross-language slot-structure differences are not error gold labels; 20 remain in the retained test set.
- Original test has 59 supported reference intents (no `cooking_query`), although the classifier output space has 60. Original dev lacks `audio_volume_other`.
- Normalized duplicate-component leakage was controlled; paraphrase overlap and public pretraining contamination are unresolved. Duplicate-related test IDs motivated component-bootstrap sensitivity.
- The shared head uses pooled three-language training, whereas character heads are language-specific. This is a system comparison, not an isolated encoder causal effect.
- Thresholds are fitted separately on each language's dev set. This is not a cross-language threshold-transfer experiment or a risk guarantee.
- Bootstrap intervals are conditional on fixed models and thresholds. Round 11 diagnostics are exploratory because test results were already seen.
- Abstention reduces automated coverage; there is no human fallback and no assumed correction of abstained answers.
- Engineering tests establish computation integrity, not research novelty. The MASSIVE paper route remains DROP.

## Attribution

MASSIVE is provided by Amazon and contributors under [CC BY 4.0](https://github.com/alexa/massive). The frozen encoder is [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, with an Apache-2.0 model card. No dataset utterances, trained heads or pretrained weights are distributed here. Original package code is MIT; data/model licenses remain separate. This is a reviewable release candidate, not an already published repository.
