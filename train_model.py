"""Train the GRU phoneme decoder.

Usage:
    python train_model.py --dataset_path ./data/ptDecoder_ctc --output_dir ./outputs
"""
import argparse
import os
import sys

# Allow running from a fresh clone without `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from neural_decoder.neural_decoder_trainer import trainModel  # noqa: E402

# --- Hyperparameters ---
args = {}
args['batchSize'] = 16          # original: 64 (reduced for a single consumer GPU)
args['lrStart'] = 0.02
args['lrEnd'] = 0.02            # equal to lrStart, so the learning rate stays constant
args['nUnits'] = 512            # GRU hidden size per direction; original: 1024
args['nBatch'] = 10000          # number of training steps
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40           # 39 phonemes + silence (CTC blank is added by the model)
args['nInputFeatures'] = 256    # 128 electrodes x (threshold crossings + spike-band power)
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8      # training-time augmentation
args['constantOffsetSD'] = 0.2  # training-time augmentation
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GRU phoneme decoder for Speech BCI')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the preprocessed ptDecoder_ctc pickle')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save model weights and training stats')
    parser.add_argument('--n_batch', type=int, default=None,
                        help=f"Override number of training steps (default: {args['nBatch']})")
    parser.add_argument('--seed', type=int, default=None,
                        help=f"Override random seed (default: {args['seed']})")
    cli = parser.parse_args()

    args['datasetPath'] = cli.dataset_path
    args['outputDir'] = cli.output_dir
    if cli.n_batch is not None:
        args['nBatch'] = cli.n_batch
    if cli.seed is not None:
        args['seed'] = cli.seed

    trainModel(args)
