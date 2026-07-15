# Kokoro TTS Voice Cloning (Style Space Search)

This repository contains the source code and deliverables for cloning a target speaker's voice using the frozen **Kokoro-82M** text-to-speech (TTS) model without any fine-tuning or gradients (zero-gradient optimization in style space).

## Approach Overview

Kokoro-82M is a frozen model where vocal characteristics are determined solely by a `510 x 256` style tensor. To clone the target speaker:
1. **Phase 1: Convex Simplex Weight Search**: We rank all 54 stock voices and optimize the blending weights (convex combination) of the top-5 closest matches. This establishes a clean, natural base voice.
2. **Phase 2: Shared 256-D Perturbation Walk**: Instead of independent row perturbations (which degrade on unseen sentences), we apply a uniform 256-dimensional shift across all 510 rows.
3. **L2 Manifold Regularization**: We apply a soft penalty on L2 drift from the stock voice manifold to prevent vocal degradation and guarantee compliance with the ASR Intelligibility Gate (WER < 25%).

## Repository Structure

* [NOTES.md](NOTES.md): Documenting technical insights, fitness function design, and voice similarity plateaus.
* [RUNLOG.md](RUNLOG.md): Chronological log of experimental runs, parameters, and results.
* [SUMMARY.html](SUMMARY.html): Web presentation dashboard of the research.
* `starter/`:
  * `voice.pt`: The final optimized style tensor (`510 x 1 x 256`) that clones the target speaker.
  * `search.py`: The finalized optimization search script.
  * `synth.py`: Speech synthesis helper.
  * `similarity.py`: Resemblyzer speaker-similarity encoder wrapper.
  * `blend.py`: Baseline stock voice ranker.

## How to Run

1. Place your target speaker clips in `reference/` and model assets in `kokoro_assets/`.
2. Run the search script on CPU:
   ```bash
   python starter/search.py --reference_dir reference --iters 100 --out starter/voice.pt
   ```
