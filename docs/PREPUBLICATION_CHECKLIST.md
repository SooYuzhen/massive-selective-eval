# GitHub prepublication checklist

The candidate is prepared for review; this task does not publish it.

- [x] Correct project name: Multilingual Selective Prediction and Reliability Evaluation.
- [x] MASSIVE engineering GO and independent paper DROP are both preserved.
- [x] No claim that MASSIVE intent classification is an LLM Judge experiment.
- [x] Reused source functions identified by hash; new adapters distinguished.
- [x] Raw corpus, worker metadata, model weights, private inference inputs and credentials excluded.
- [x] Dataset and pretrained model licenses distinguished from the package MIT license.
- [x] English README, Chinese introduction and complete command guide present.
- [x] Synthetic tests run without the complete raw corpus.
- [x] Historical frozen-threshold metrics and paired analysis checked without training or changing old artifacts.
- [x] New inference adapter smoke-tested on synthetic unlabeled strings with local existing models; no historical test predictions regenerated.
- [x] Candidate paths and credential patterns scanned; exact manifest produced.
- [x] No repository created and no GitHub push performed.
- [ ] Owner performs final review of the exact ZIP/manifest and desired public identity, then separately authorizes publication.

The final unchecked item is the publication action requested to remain manual, not a reason to alter the scientific decision or launch more research. Add no local data or replay folders when publishing. If files change, rerun the release scan and tests before updating the manifest.
