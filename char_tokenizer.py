"""
One character = one token for ASCII-art canvases.

Adds 95 dedicated tokens (one per printable ASCII char, 0x20-0x7E) plus
<canvas> / </canvas> to an existing HF tokenizer. Inside a canvas every grid
cell becomes exactly one token, so token k of row r is always column k.
No newline tokens: the grid is a fixed W x H, rows are implied by position.

Usage (on your machine / Colab):
    python char_tokenizer.py --model Qwen/Qwen2.5-0.5B     # compare BPE vs char tokens
    python char_tokenizer.py                               # self-test, no transformers needed
"""
import argparse

CHARSET = [chr(i) for i in range(0x20, 0x7F)]          # 95 printable ASCII chars
CANVAS_OPEN, CANVAS_CLOSE = "<canvas>", "</canvas>"
UNKNOWN_CHAR = "?"                                      # non-ASCII chars map to this


def char_token(c: str) -> str:
    """'#' -> '<c23>'. Hex code keeps token names free of spaces/brackets."""
    return f"<c{ord(c):02x}>"


ALL_NEW_TOKENS = [char_token(c) for c in CHARSET] + [CANVAS_OPEN, CANVAS_CLOSE]
TOKEN_TO_CHAR = {char_token(c): c for c in CHARSET}


# ---------------------------------------------------------------- grid <-> text
def normalize_grid(text: str, W: int, H: int, fill: str = " ") -> list[str]:
    """Crop/pad an ASCII art string to exactly H rows of W chars."""
    rows = text.split("\n")[:H]
    rows = ["".join(c if c in TOKEN_TO_CHAR.values() else UNKNOWN_CHAR for c in r) for r in rows]
    rows = [r[:W].ljust(W, fill) for r in rows]
    rows += [fill * W] * (H - len(rows))
    return rows


def grid_to_token_string(grid: list[str]) -> str:
    """Grid -> '<canvas><c20><c2f>...</canvas>' (W*H cell tokens, no newlines)."""
    return CANVAS_OPEN + "".join(char_token(c) for row in grid for c in row) + CANVAS_CLOSE


def tokens_to_grid(tokens: list[str], W: int) -> list[str]:
    """Inverse of grid_to_token_string, given token *strings*. Ignores anything outside the canvas."""
    cells = [TOKEN_TO_CHAR[t] for t in tokens if t in TOKEN_TO_CHAR]
    return ["".join(cells[i:i + W]) for i in range(0, len(cells), W)]


def cell_positions(W: int, H: int) -> tuple[list[int], list[int]]:
    """Row / column index for each cell token, in canvas order (for 2D embeddings later)."""
    rows = [r for r in range(H) for _ in range(W)]
    cols = [c for _ in range(H) for c in range(W)]
    return rows, cols


# ---------------------------------------------------------------- HF integration
def add_char_tokens(tokenizer, model=None):
    """
    Register the char tokens as *special* tokens, so BPE never merges or splits them.
    If a model is given, resize its embeddings and initialise each new token from the
    embedding of the original character, so training starts from something sensible.
    """
    tokenizer.add_tokens(ALL_NEW_TOKENS, special_tokens=True)
    if model is None:
        return tokenizer
    import torch

    model.resize_token_embeddings(len(tokenizer))
    tables = [model.get_input_embeddings().weight]
    out = model.get_output_embeddings()
    if out is not None and out.weight.data_ptr() != tables[0].data_ptr():   # untied lm_head
        tables.append(out.weight)

    with torch.no_grad():
        for c in CHARSET:
            src = tokenizer.encode(c, add_special_tokens=False)            # original char's id(s)
            dst = tokenizer.convert_tokens_to_ids(char_token(c))
            for W_ in tables:
                W_[dst] = W_[src].mean(0)
        for t in (CANVAS_OPEN, CANVAS_CLOSE):                               # start canvas markers at mean
            dst = tokenizer.convert_tokens_to_ids(t)
            for W_ in tables:
                W_[dst] = W_[: dst].mean(0)
    return tokenizer


def encode_canvas(tokenizer, grid: list[str]) -> list[int]:
    return tokenizer.encode(grid_to_token_string(grid), add_special_tokens=False)


def decode_canvas(tokenizer, ids: list[int], W: int) -> list[str]:
    return tokens_to_grid(tokenizer.convert_ids_to_tokens(ids), W)


# ---------------------------------------------------------------- demo / tests
EXAMPLE = r"""    /\     /\
   /  \___/  \
  |   o   o   |
  |     ^     |
   \  \___/  /
    \_______/
     |     |"""


def self_test(W=16, H=8):
    grid = normalize_grid(EXAMPLE, W, H)
    s = grid_to_token_string(grid)
    toks = [CANVAS_OPEN] + [s[i:i + 5] for i in range(len(CANVAS_OPEN), len(s) - len(CANVAS_CLOSE), 5)] + [CANVAS_CLOSE]
    assert all(len(r) == W for r in grid) and len(grid) == H
    assert len(toks) == W * H + 2, len(toks)
    assert tokens_to_grid(toks, W) == grid
    rows, cols = cell_positions(W, H)
    assert len(rows) == len(cols) == W * H and rows[W] == 1 and cols[W] == 0
    print(f"self-test OK: {W}x{H} grid -> {W*H} cell tokens + 2 markers, round-trip exact")
    print("grid:"); print("\n".join("|" + r + "|" for r in grid))
    print("first row as tokens:", " ".join(toks[1:W + 1]))


def compare(model_name, W=16, H=8):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    grid = normalize_grid(EXAMPLE, W, H)

    print("BPE tokens per row (original tokenizer):")
    for r in grid:
        pieces = [tok.decode([i]) for i in tok.encode(r, add_special_tokens=False)]
        print(f"  {len(pieces):2d}  {pieces}")

    add_char_tokens(tok)
    ids = encode_canvas(tok, grid)
    assert len(ids) == W * H + 2, f"expected {W*H+2} ids, got {len(ids)}"
    assert decode_canvas(tok, ids, W) == grid
    print(f"\nchar tokens: every row = {W} tokens, total {len(ids)} ids, round-trip exact")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF model name, e.g. Qwen/Qwen2.5-0.5B")
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--H", type=int, default=8)
    a = ap.parse_args()
    self_test(a.W, a.H)
    if a.model:
        print()
        compare(a.model, a.W, a.H)
