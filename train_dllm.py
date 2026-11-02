"""
Train a small masked-diffusion LM (dLLM) on labelled ASCII art.

One model, one sequence format, two tasks:
    <canvas> c_1 ... c_{W*H} </canvas> <label> <cls:y>
  recognition : canvas given, label masked   -> predict the class token (1 step)
  generation  : label given, canvas masked   -> iterative unmasking

Every canvas cell is exactly one token (95 printable ASCII chars). Canvas cells
get 2D (row, col) embeddings on top of the 1D position embedding, and --conv
adds a 2D token convolution after every transformer layer.

Examples (RTX 4060, ~5M params, a few minutes per 10k steps):
    python train_dllm.py --source shapes --style edge                 # smoke test
    python train_dllm.py --source mnist  --style mixed                # main run
    python train_dllm.py --source mnist  --style mixed --no_2d        # ablate 2D pos
    python train_dllm.py --source mnist  --style mixed --conv         # + 2D token conv
    python train_dllm.py --source mnist  --space_weight 0.3           # down-weight blank cells
"""
import argparse
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ascii_data import MnistSource, ShapeSource, make_source, to_ascii

CHARSET = [chr(i) for i in range(0x20, 0x7F)]
SPECIALS = ["<mask>", "<canvas>", "</canvas>", "<label>"]


# ------------------------------------------------------------------ vocab / data
class Vocab:
    def __init__(self, classes):
        self.classes = classes
        self.itos = SPECIALS + CHARSET + [f"<cls:{c}>" for c in classes]
        self.MASK, self.OPEN, self.CLOSE, self.LABEL = range(4)
        self.char0 = len(SPECIALS)
        self.cls0 = self.char0 + len(CHARSET)
        self.SPACE = self.char0                                   # ' ' is CHARSET[0]

    def __len__(self):
        return len(self.itos)

    def encode(self, grid, y):
        cells = [self.char0 + ord(c) - 0x20 for row in grid for c in row]
        return [self.OPEN] + cells + [self.CLOSE, self.LABEL, self.cls0 + y]

    def decode_grid(self, ids, W):
        chars = "".join(CHARSET[i - self.char0] if self.char0 <= i < self.cls0 else "?" for i in ids)
        return [chars[i:i + W] for i in range(0, len(chars), W)]


def build_dataset(source, style, n, W, H, vocab, train, seed, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    path = f"{cache_dir}/{source}_{style}_{W}x{H}_{'train' if train else 'test'}_{n}_{seed}.pt"
    if os.path.exists(path):
        return torch.load(path)
    random.seed(seed + (0 if train else 1))
    src = make_source(source, train=train, seed=seed)
    rows = []
    t0 = time.time()
    for i in range(n):
        img, y = src.sample()
        rows.append(vocab.encode(to_ascii(img, W, H, style), y))
        if (i + 1) % 10000 == 0:
            print(f"  built {i + 1}/{n} ({time.time() - t0:.0f}s)")
    x = torch.tensor(rows, dtype=torch.long)
    torch.save(x, path)
    return x


# ------------------------------------------------------------------ model
class GridConv(nn.Module):
    """
    2D token convolution on the canvas cells: [B, W*H, d] -> [B, d, H, W] -> depthwise kh x kw conv
    -> pointwise -> residual. Non-canvas positions pass through untouched.
    The pointwise layer is zero-initialised, so the block starts as an identity.
    kernel=(kh, kw) in (rows, cols); cells are ~2x taller than wide, so e.g. 3x5 covers a
    roughly square physical area. (1, 1) adds the same parameters with no spatial mixing.
    """
    def __init__(self, d, W, H, kernel=(3, 3)):
        super().__init__()
        self.W, self.H = W, H
        kh, kw = kernel
        assert kh % 2 == 1 and kw % 2 == 1, "kernel sizes must be odd"
        self.norm = nn.LayerNorm(d)
        self.dw = nn.Conv2d(d, d, (kh, kw), padding=(kh // 2, kw // 2), groups=d)
        self.pw = nn.Linear(d, d)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, h):
        B, _, d = h.shape
        n = self.W * self.H
        c = h[:, 1:1 + n]
        x = self.dw(self.norm(c).transpose(1, 2).reshape(B, d, self.H, self.W))
        x = self.pw(F.gelu(x.reshape(B, d, n).transpose(1, 2)))
        return torch.cat([h[:, :1], c + x, h[:, 1 + n:]], dim=1)


class GridTransformer(nn.Module):
    """Bidirectional transformer (no causal mask) = the dLLM denoiser."""
    def __init__(self, vocab_size, W, H, d=256, layers=6, heads=8, use_2d=True, conv=None):
        super().__init__()
        self.L = W * H + 4
        self.use_2d = use_2d
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(self.L, d)
        self.row = nn.Embedding(H, d)
        self.col = nn.Embedding(W, d)
        # full-length row/col indices; gate = 1 only on canvas cells (positions 1..W*H)
        r = [0] + [i // W for i in range(W * H)] + [0, 0, 0]
        c = [0] + [i % W for i in range(W * H)] + [0, 0, 0]
        g = [0.0] + [1.0] * (W * H) + [0.0, 0.0, 0.0]
        self.register_buffer("pos_ids", torch.arange(self.L), persistent=False)
        self.register_buffer("row_ids", torch.tensor(r), persistent=False)
        self.register_buffer("col_ids", torch.tensor(c), persistent=False)
        self.register_buffer("gate", torch.tensor(g).unsqueeze(-1), persistent=False)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(layers)])
        # conv = None (off) or a (kh, kw) kernel shape
        self.convs = nn.ModuleList([GridConv(d, W, H, conv) for _ in range(layers)]) if conv else None
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok.weight                         # tie embeddings

    def forward(self, x):
        h = self.tok(x) + self.pos(self.pos_ids)
        if self.use_2d:
            h = h + self.gate * (self.row(self.row_ids) + self.col(self.col_ids))
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.convs is not None:
                h = self.convs[i](h)
        return self.head(self.norm(h))


# ------------------------------------------------------------------ loss
def dllm_loss(model, x, vocab, p_recog, space_weight):
    """
    MDLM-style loss on a mixed batch.
      generation rows : mask each token w.p. t ~ U(0,1), loss = CE / t over masked tokens
      recognition rows: mask only the class token, loss = CE at that position
    Structural tokens (<canvas>, </canvas>, <label>) are never masked.
    space_weight < 1 down-weights blank cells (the canvas analogue of trailing EOS).
    """
    B, L = x.shape
    lab = L - 1
    dev = x.device
    t = torch.rand(B, 1, device=dev).clamp_min(1e-3)
    m = torch.rand(B, L, device=dev) < t
    m[:, [0, L - 3, L - 2]] = False
    rec = torch.rand(B, device=dev) < p_recog
    m[rec] = False
    m[rec, lab] = True
    m[~rec & (m.sum(1) == 0), lab] = True                          # never an empty mask

    logits = model(torch.where(m, torch.full_like(x, vocab.MASK), x))
    ce = F.cross_entropy(logits.float().transpose(1, 2), x, reduction="none")
    w = torch.where(x == vocab.SPACE, space_weight, 1.0)

    gen = ~rec
    loss_gen = ((ce * w * m.float() / t)[gen]).sum() / max(1, gen.sum().item() * L)
    loss_rec = ce[rec, lab].mean() if rec.any() else ce.new_zeros(())
    return loss_gen + loss_rec, loss_gen.item(), loss_rec.item()


# ------------------------------------------------------------------ eval / sampling
@torch.no_grad()
def recognition_acc(model, x, vocab, bs=256):
    model.eval()
    lab = x.shape[1] - 1
    correct = 0
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        inp = xb.clone()
        inp[:, lab] = vocab.MASK
        logits = model(inp)[:, lab]
        pred = logits[:, vocab.cls0:vocab.cls0 + len(vocab.classes)].argmax(-1) + vocab.cls0
        correct += (pred == xb[:, lab]).sum().item()
    model.train()
    return correct / len(x)


SAMPLE_ORDERS = ("confidence", "random", "ink_first")


@torch.no_grad()
def generate(model, vocab, y, W, H, steps=64, temp=1.0, order="random", device="cpu"):
    """
    Label-conditioned sampling: start with an all-masked canvas, unmask a batch of cells per step.
    Which cells get committed each step:
      confidence : highest-confidence cells first (LLaDA/MaskGIT default). Blank cells are the
                   easiest to predict, so they get locked first and the canvas tends to collapse
                   to empty -- the canvas analogue of premature trailing EOS.
      random     : uniformly random cells (plain MDLM ancestral sampling).
      ink_first  : confidence order, but cells whose sampled char is a space are ranked last,
                   so strokes are committed before background.
    """
    model.eval()
    n = W * H
    x = torch.tensor([[vocab.OPEN] + [vocab.MASK] * n + [vocab.CLOSE, vocab.LABEL, vocab.cls0 + y]], device=device)
    for s in range(steps):
        masked = (x[0, 1:n + 1] == vocab.MASK).nonzero().squeeze(-1) + 1
        if len(masked) == 0:
            break
        logits = model(x)[0, masked, vocab.char0:vocab.cls0].float() / temp
        probs = logits.softmax(-1)
        cand = torch.multinomial(probs, 1).squeeze(-1)
        conf = probs.gather(-1, cand[:, None]).squeeze(-1)
        remain = math.cos(math.pi / 2 * (s + 1) / steps)          # cosine schedule
        k = min(len(masked), max(1, len(masked) - int(round(remain * n))))
        if order == "confidence":
            score = conf
        elif order == "random":
            score = torch.rand_like(conf)
        elif order == "ink_first":
            score = conf - 2.0 * (cand == 0).float()                # cand index 0 = ' '
        else:
            raise ValueError(order)
        keep = score.topk(k).indices
        x[0, masked[keep]] = cand[keep] + vocab.char0
    model.train()
    return vocab.decode_grid(x[0, 1:n + 1].tolist(), W)


def ink_fraction(grid):
    cells = "".join(grid)
    return sum(c != " " for c in cells) / max(1, len(cells))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="shapes", choices=["shapes", "mnist"])
    ap.add_argument("--style", default="edge", choices=["edge", "tone", "mixed"])
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--p_recog", type=float, default=0.5, help="fraction of batch trained on recognition")
    ap.add_argument("--space_weight", type=float, default=1.0)
    ap.add_argument("--no_2d", action="store_true", help="disable row/col embeddings")
    ap.add_argument("--conv", action="store_true", help="2D token conv after each layer")
    ap.add_argument("--kernel", default="3x3", help="conv kernel as ROWSxCOLS, e.g. 3x3, 3x5, 1x1")
    ap.add_argument("--results", default="results.csv", help="append final metrics here")
    ap.add_argument("--sample_orders", default="confidence,random,ink_first",
                    help=f"comma-separated subset of {SAMPLE_ORDERS}")
    ap.add_argument("--gen_steps", type=int, default=64)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ckpt")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes = (ShapeSource if a.source == "shapes" else MnistSource).classes
    vocab = Vocab(classes)

    print("building data ...")
    xtr = build_dataset(a.source, a.style, a.n_train, a.W, a.H, vocab, True, a.seed).to(dev)
    xte = build_dataset(a.source, a.style, a.n_test, a.W, a.H, vocab, False, a.seed).to(dev)

    kernel = tuple(int(v) for v in a.kernel.lower().split("x")) if a.conv else None
    model = GridTransformer(len(vocab), a.W, a.H, a.d, a.layers, a.heads,
                            use_2d=not a.no_2d, conv=kernel).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    conv_desc = f"{kernel[0]}x{kernel[1]}" if kernel else "off"
    print(f"params: {n_params / 1e6:.2f}M | seq len {model.L} | "
          f"2D pos {'off' if a.no_2d else 'on'} | conv {conv_desc} | device {dev}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" else dict(device_type="cpu", enabled=False)

    t0 = time.time()
    for step in range(1, a.steps + 1):
        xb = xtr[torch.randint(0, len(xtr), (a.bs,), device=dev)]
        with torch.autocast(**amp):
            loss, lg, lr_ = dllm_loss(model, xb, vocab, a.p_recog, a.space_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            print(f"step {step:6d} | loss {loss.item():.4f} (gen {lg:.4f} rec {lr_:.4f}) | {time.time() - t0:.0f}s")
        if step % a.eval_every == 0 or step == a.steps:
            with torch.autocast(**amp):
                acc = recognition_acc(model, xte, vocab)
            print(f"== step {step} | test recognition acc {acc:.3f}")

    os.makedirs(a.out, exist_ok=True)
    tag = (f"{a.source}_{a.style}{'_no2d' if a.no_2d else ''}"
           f"_conv{conv_desc}_sw{a.space_weight}_seed{a.seed}")
    torch.save({"model": model.state_dict(), "args": vars(a)}, f"{a.out}/{tag}.pt")

    # reference: how much of a real canvas is ink
    lab = xte.shape[1] - 1
    real_ink = (xte[:, 1:lab - 2] != vocab.SPACE).float().mean().item()
    print(f"\nreal test canvases: ink fraction {real_ink:.3f}")

    samples, ink = [], {}
    for order in a.sample_orders.split(","):
        print(f"\n=== label-conditioned samples | order = {order} ===")
        fr = []
        for y, name in enumerate(classes):
            with torch.autocast(**amp):
                g = generate(model, vocab, y, a.W, a.H, steps=a.gen_steps, order=order, device=dev)
            fr.append(ink_fraction(g))
            samples.append(f"--- [{order}] {name} ---\n" + "\n".join(g))
            print(samples[-1])
        ink[order] = sum(fr) / len(fr)
        print(f"[{order}] mean ink fraction {ink[order]:.3f} (real {real_ink:.3f})")
    with open(f"{a.out}/{tag}_samples.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(samples) + "\n")

    new = not os.path.exists(a.results)
    ink_cols = ",".join(f"ink_{o}" for o in SAMPLE_ORDERS)
    ink_vals = ",".join(f"{ink[o]:.3f}" if o in ink else "" for o in SAMPLE_ORDERS)
    with open(a.results, "a", encoding="utf-8") as f:
        if new:
            f.write(f"source,style,pos2d,conv,space_weight,seed,steps,params_M,test_acc,"
                    f"ink_real,{ink_cols},minutes\n")
        f.write(f"{a.source},{a.style},{not a.no_2d},{conv_desc},{a.space_weight},{a.seed},{a.steps},"
                f"{n_params / 1e6:.3f},{acc:.4f},{real_ink:.3f},{ink_vals},{(time.time() - t0) / 60:.1f}\n")
    print(f"\nappended to {a.results}: conv {conv_desc} | acc {acc:.4f}")


if __name__ == "__main__":
    main()
