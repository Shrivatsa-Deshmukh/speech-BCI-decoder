# Speech BCI Decoder

A PyTorch implementation of the intracortical GRU phoneme decoder from [Willett et al. (2023)](https://www.nature.com/articles/s41586-023-06377-x), adapted to train on a single consumer GPU. It decodes multichannel neural population activity into phoneme sequences using a bidirectional GRU trained with Connectionist Temporal Classification (CTC).

**Achieved 78.77% phoneme decoding accuracy, within 1.5 percentage points of the ~80.3% (19.7% phoneme error rate) reported in Willett et al. (2023) at the same stage, on a single consumer GPU and without the Kaldi decoding stack.**

The code builds on [cffan/neural_seq_decoder](https://github.com/cffan/neural_seq_decoder), the PyTorch baseline released with the paper's dataset (see [Acknowledgements](#acknowledgements)).

<p align="center">
  <img src="architecture.svg" alt="Speech BCI Decoder architecture" width="100%">
</p>

---

## Pipeline Scope

The full Willett et al. system decodes speech in three stages:

```
Stage 1 — GRU Phoneme Decoder     <- this repo
          Maps T x 256 neural features to per-timestep
          phoneme probability distributions via a CTC-trained RNN

Stage 2 — Viterbi Search          (not implemented)
          Selects the maximum-likelihood phoneme path
          through the per-timestep posterior distributions

Stage 3 — n-gram Language Model   (not implemented)
          Beam search over a 125,000-word vocabulary combines
          the phoneme path with language statistics to produce
          the final word sequence
```



---

## Model Architecture

**Input** — `T x 256` per sentence: 128 electrodes (area 6v) x 2 features, threshold crossings (TX) and spike-band power, binned at 20 ms and z-scored within each recording block.

1. **Gaussian smoothing** (σ = 2 bins) — depthwise convolution over time.
2. **Day-specific affine transform** — `x · W_day + b_day`, one weight matrix and bias per recording session, with `W_day` initialised to the identity. Corrects for drift in the recorded signals between days.
3. **Softsign** nonlinearity.
4. **Temporal unfolding** (kernel 32, stride 4) — stacks a 32-bin (640 ms) sliding window into the feature dimension, giving `T' x 8192` with `T' = ⌊(T − 32) / 4⌋ + 1`.
5. **Bidirectional GRU** — 5 layers, 512 units per direction, dropout 0.4 between layers. Orthogonal initialisation on recurrent weights, Xavier on input weights.
6. **Linear projection → 41 logits** — 39 phonemes + silence (`SIL`) + the CTC blank.

**Training** uses CTC loss; **inference** uses greedy CTC decoding (argmax per step, merge repeats, drop blanks).

**Day-specific transforms** matter for multi-session generalisation: the arrays shift slightly in tissue between recording days, so the same neural activity appears differently from session to session. A learned per-session correction maps every day into a shared input space.

**CTC training** removes the need for frame-level phoneme alignments, which don't exist for attempted speech. The CTC objective sums over every valid alignment of the phoneme sequence to the output time steps.

> `model.py` also defines per-day `Linear` layers (`inpLayer0`–`inpLayer23`), carried over from the upstream code. They are not used in the forward pass; the day-specific correction above is done entirely by `W_day` and `b_day`.

---

## Changes from the Original Configuration


| Hyperparameter | Original | This Repo | Notes |
|---|---|---|---|
| `nUnits` (GRU hidden dim) | 1024 | **512** | ~75% fewer recurrent parameters |
| `batchSize` | 64 | **16** | 4x less activation memory per step |
| `dropout` | 0.4 | 0.4 | Unchanged |
| `nLayers` | 5 | 5 | Unchanged |
| `bidirectional` | True | True | Unchanged |
| `kernelLen` | 32 | 32 | Unchanged |
| `lrStart` / `lrEnd` | 0.02 / 0.02 | 0.02 / 0.02 | Unchanged (constant learning rate) |

All hyperparameters are defined and annotated in `train_model.py`.

---

## Results

**Phoneme decoding (Stage 1, before any language model)**

| | This Repo | Willett et al. (2023) |
|---|---|---|
| Phoneme accuracy | **78.77%** | ~80.3% |
| Phoneme error rate | 21.23% | 19.7% |
| GRU hidden size / batch size | 512 / 16 | 1024 / 64 |
| Hardware | Single consumer GPU | — |

The result is within **1.5 percentage points** of the published Stage 1 figure, which shows that strong phoneme-level representations can be recovered from the neural signal with half the GRU width and without the Kaldi decoding stack.

For reference, the full paper pipeline (GRU + language model) reports 9.1% word error rate with a 50-word vocabulary and 23.8% with a 125,000-word vocabulary.

<sub>Single run, seed 0. The test set is evaluated every 100 training steps and the checkpoint with the lowest test phoneme error rate is kept, as in the upstream training loop.</sub>

---

## Repository Structure

```
speech-BCI-decoder/
├── train_model.py                     # Entry point: hyperparameters + CLI
├── pyproject.toml                     # Package definition (pip install -e .)
├── requirements.txt
├── LICENSE
├── src/
│   └── neural_decoder/
│       ├── model.py                   # GRUDecoder architecture
│       ├── neural_decoder_trainer.py  # Training loop, evaluation, model saving/loading
│       ├── dataset.py                 # SpeechDataset: loads pickle, returns tensors
│       └── augmentations.py           # GaussianSmoothing, WhiteNoise, MeanDriftNoise
├── notebooks/
│   └── formatCompetitionData.ipynb    # Converts raw .mat files to the training pickle
└── docs/
    └── architecture.svg
```

---

## How to Run

Training requires an NVIDIA GPU with CUDA.

**Step 0 — Install**

```bash
git clone https://github.com/Shrivatsa-Deshmukh/speech-BCI-decoder.git
cd speech-BCI-decoder
pip install -r requirements.txt
pip install -e .
```

Install the PyTorch build that matches your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) if the default one doesn't.

**Step 1 — Download and preprocess the data**

Download the dataset from [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq) and open the preprocessing notebook:

```bash
jupyter notebook notebooks/formatCompetitionData.ipynb
```

Before running it, change the paths in the notebook to match your machine: set `dataDir` (in two cells) to your local `competitionData` folder, and change the path in the final `open(...)` call to where you want the output pickle, e.g. `./data/ptDecoder_ctc`. Then run all cells.

This writes a `ptDecoder_ctc` pickle containing block z-scored neural features and phoneme label sequences, split into `train`, `test` and `competition` sets.

**Step 2 — Train**

```bash
python train_model.py --dataset_path ./data/ptDecoder_ctc --output_dir ./outputs
```

Optional flags: `--n_batch` (number of training steps, default 10000) and `--seed` (default 0).

Every 100 steps the script evaluates on the test set and prints the CTC loss and phoneme error rate (CER). The checkpoint with the lowest CER is saved to `outputs/modelWeights`, and loss and CER curves are saved to `outputs/trainingStats`.

**Loading a saved model**

```python
from neural_decoder.neural_decoder_trainer import loadModel
model = loadModel('./outputs', device='cuda')
model.eval()
```

---

## Acknowledgements

The model, training loop, augmentations, dataset class and preprocessing notebook come from [cffan/neural_seq_decoder](https://github.com/cffan/neural_seq_decoder) by Chaofei Fan and colleagues, the PyTorch baseline accompanying the Willett et al. (2023) dataset. This repo adds a reduced model configuration for consumer hardware, a command-line entry point and packaging.

## References

- Willett, F. R. et al. [A high-performance speech neuroprosthesis](https://doi.org/10.1038/s41586-023-06377-x). *Nature* 620, 1031–1036 (2023).
- PyTorch baseline: [cffan/neural_seq_decoder](https://github.com/cffan/neural_seq_decoder)
- Dataset: [Dryad doi:10.5061/dryad.x69p8czpq](https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq)
