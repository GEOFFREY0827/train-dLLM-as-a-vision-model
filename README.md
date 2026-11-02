# Train dLLM as a Vision Model

Can a **text** diffusion language model (dLLM) learn to *see* and *draw*?

This repo trains a small masked-diffusion LM on ASCII art: text tokens laid out
on a 2D canvas. The same model recognizes what a canvas depicts and draws a
canvas for a given label, using only a character vocabulary.

## Idea

Text dLLMs are bidirectional, so every canvas cell can see its neighbours
above, below, left and right. Still, three things keep them from treating
ASCII art as an image:

1. **Tokenization breaks the grid.** BPE merges runs like `____` or `/\`, so token
   index no longer equals column index. Fix: one character = one token.
2. **Position is 1D.** The cell directly above is a whole row away in the sequence.
   Fix: row/column embeddings on canvas cells.
3. **No local 2D inductive bias.** Image diffusion relies on convolutional denoisers;
   text dLLMs have none. Fix: **2D token convolution** inside the transformer.

## Model

```
<canvas> c_1 ... c_{W*H} </canvas> <label> <cls:y>
```

- Vocabulary: 95 printable ASCII chars + class tokens + `<mask>` and markers.
- Bidirectional transformer denoiser, MDLM-style masked diffusion loss.
- **Recognition**: mask the label and predict it in one step.
- **Generation**: mask the canvas and unmask it iteratively by confidence.
- Optional `GridConv` after every layer: tokens are reshaped to `[B, d, H, W]`, passed
  through a depthwise conv and a zero-initialised pointwise projection, and added
  back as a residual.

## Files

| File | What it does |
|---|---|
| `ascii_data.py` | Labelled ASCII art on the fly: procedural shapes or MNIST, edge (`- / \| \`) or tone (`@%#*+=-:.`) style |
| `train_dllm.py` | Model, masked-diffusion training, recognition eval, label-conditioned sampling |
| `char_tokenizer.py` | Adds one-token-per-char canvas tokens to an existing HF tokenizer, for later work on pretrained dLLMs |

## Quick start

```bash
pip install -r requirements.txt

python ascii_data.py                                      # preview the data
python train_dllm.py --source shapes --style edge         # smoke test
python train_dllm.py --source mnist  --style mixed        # main run
python train_dllm.py --source mnist  --style mixed --conv # + 2D token conv
python train_dllm.py --source mnist  --style mixed --no_2d
python train_dllm.py --source mnist  --space_weight 0.3   # down-weight blank cells
python train_dllm.py --source mnist  --conv --kernel 3x5  # kernel = ROWSxCOLS

bash sweep_kernel.sh                         # off / 1x1 / 3x3 / 3x5 / 5x5 / 1x5 / 3x1
bash sweep_kernel.sh mnist mixed 10000 0 1 2 # same, 3 seeds
```

Every run appends a row to `results.csv` and writes its samples to `ckpt/*_samples.txt`.

The default config (~5M params, 32x16 canvas) fits on an 8 GB GPU.

## Planned experiments

- [ ] 2D token conv vs. none (recognition accuracy and sample quality)
- [ ] 3x3 vs. 3x5 kernel (cells are ~2x taller than wide)
- [ ] Conv placement: after the block vs. inside the FFN (ConvFFN)
- [ ] Mask-aware (partial) convolution for heavily masked inputs
- [ ] Checkerboard / Halton unmasking order for generation
- [ ] Translation robustness test
- [ ] Blank-cell down-weighting (the canvas analogue of trailing EOS)
- [ ] Plug-in 2D conv adapter for a pretrained dLLM (LLaDA-8B + LoRA)

## Related work

- Chargrid: Towards Understanding 2D Documents (EMNLP 2018): 2D character grid + CNN
- SVE-ASCII / Unlocking the Latent Canvas (2026): AR LLM fine-tuned for ASCII art
- ASCIIEval (ICLR 2026), ASCIIBench (2025): ASCII art recognition benchmarks
- MaskGIT (CVPR 2022), MDLM (NeurIPS 2024), LLaDA (2025): masked generative / diffusion models
- CvT, CPVT, PVTv2: convolution inside vision transformers

## Status

Early prototype. The data pipeline has been tested. The training script passes
a syntax check but has not been run end to end yet.
