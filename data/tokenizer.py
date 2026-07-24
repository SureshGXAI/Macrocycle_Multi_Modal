"""
Regex-based SMILES tokenizer (multi-character tokens like Cl, Br, ring-closure
numbers %nn, and bracket atoms [nH], [C@@H] etc. are kept as single tokens).
"""
import re
import json
from typing import List, Dict

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]

# Standard SMILES regex pattern (Schwaller et al. style)
SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|se|@@|@|=|#|\$|:|/|\\|\(|\)|\.|%\d{2}|\d|[A-Za-z])"
)


def smiles_tokenize(smiles: str) -> List[str]:
    tokens = SMILES_REGEX.findall(smiles)
    if "".join(tokens) != smiles:
        raise ValueError(f"Tokenization mismatch for SMILES: {smiles}")
    return tokens


class SmilesTokenizer:
    def __init__(self, vocab: Dict[str, int] = None):
        if vocab is None:
            vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.vocab = vocab
        self.inv_vocab = {i: tok for tok, i in vocab.items()}

    @property
    def pad_id(self):
        return self.vocab[PAD]

    @property
    def bos_id(self):
        return self.vocab[BOS]

    @property
    def eos_id(self):
        return self.vocab[EOS]

    @property
    def unk_id(self):
        return self.vocab[UNK]

    def __len__(self):
        return len(self.vocab)

    @classmethod
    def build_from_smiles_list(cls, smiles_list: List[str]) -> "SmilesTokenizer":
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        charset = set()
        for smi in smiles_list:
            try:
                charset.update(smiles_tokenize(smi))
            except ValueError:
                continue
        for tok in sorted(charset):
            if tok not in vocab:
                vocab[tok] = len(vocab)
        return cls(vocab)

    def encode(self, smiles: str, max_len: int = None, add_special: bool = True) -> List[int]:
        try:
            tokens = smiles_tokenize(smiles)
        except ValueError:
            tokens = list(smiles)  # fallback, should not normally happen
        ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        toks = []
        for i in ids:
            tok = self.inv_vocab.get(int(i), UNK)
            if strip_special and tok in SPECIAL_TOKENS:
                if tok == EOS:
                    break
                continue
            toks.append(tok)
        return "".join(toks)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SmilesTokenizer":
        with open(path) as f:
            vocab = json.load(f)
        return cls(vocab)
