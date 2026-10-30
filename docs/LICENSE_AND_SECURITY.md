# License and publication-safety review

Review scope: the files actually included in this release candidate. The original working project, raw datasets and model directories are not automatically approved for publication.

| Component | Evidence / license | Included? | Decision |
| --- | --- | --- | --- |
| New package adapters and reused project analytical code | Project-owned implementation; source-file hashes in `audit/code_provenance.json`; MIT text in `LICENSE` | Yes | Candidate code license is MIT; final publication remains the owner's action |
| MASSIVE dataset | Original downloaded archive's license is CC BY 4.0, copyright Amazon.com Inc. or affiliates; [upstream](https://github.com/alexa/massive) | No utterances, judgments, worker identifiers or raw records | No corpus redistribution; preserve attribution for aggregate evidence |
| MASSIVE aggregate results | Derived from the attributed MASSIVE experiment; full machine precision and original denominators | Yes | Research evidence with limitations; no claim of new labels or paper novelty |
| Frozen MiniLM pretrained weights | [Model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) declares Apache-2.0; fixed historical revision in README | No | Local-only loading; no weight licensing or redistribution shortcut |
| Trained TF-IDF/LR and shared LR heads | Locally trained project artifacts; serialized vectorizers may retain vocabulary | No | Supply privately for reproduction; avoid pickle and vocabulary redistribution |
| M-RewardBench and Qwen Judge assets | Separate Phase 2 experiment; independent data/model terms | No | Outside this public candidate |
| Runtime dependencies | NumPy, SciPy, scikit-learn, joblib, PyTorch, Transformers, Sentence Transformers and optional PyArrow retain their own licenses | Not vendored | Package metadata declares dependencies; this MIT license does not relicense them |
| Synthetic test fixtures | Authored for software behavior, clearly marked synthetic | Yes | No real-person or dataset examples; no scientific accuracy claim |

The copied MASSIVE license text is in `licenses/MASSIVE-DATA-CC-BY-4.0.txt`. No upstream Amazon modeling implementation or M-RewardBench evaluation implementation is copied into this package. The code reused from Rounds 9–11 is this project's own analytical implementation, not a claim of ownership over datasets, model architecture or upstream dependencies.

The automated scanner checks included text for personal home paths, common credential formats, private key headers, email-like identifiers, disallowed model/data binaries, symlinks and oversized files. File allowlisting and a manifest complement this pattern scan. A scan cannot prove ownership of all intellectual property or detect every possible secret. Scope and actual findings are saved in `audit/security_scan.json`; no placeholder token has been replaced with a real credential.

Joblib loading can execute Python objects. The inference interface requires a hash and an explicit trusted-local-model flag. Hash identity alone is not a trust guarantee. Archive loading does not extract tar paths onto disk. Offline model flags and local-only encoder loading prevent unintended model downloads.

No GitHub account, remote repository, push, token or publication action was used. This directory is a reviewable candidate; final publication authorization is not inferred from packaging.
