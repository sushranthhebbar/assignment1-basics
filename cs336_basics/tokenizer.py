import regex as re
import json
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        
        # Map (id, id) -> (new_id, rank) for O(1) lookup during encoding
        self.merge_map = {}
        for rank, (p0, p1) in enumerate(self.merges):
            p0_id, p1_id = self.byte_to_id[p0], self.byte_to_id[p1]
            new_id = self.byte_to_id[p0 + p1]
            self.merge_map[(p0_id, p1_id)] = (new_id, rank)

    def _apply_merges(self, ids: list[int]) -> list[int]:
        while len(ids) >= 2:
            # Find all possible merges and their ranks
            candidates = []
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                if pair in self.merge_map:
                    candidates.append((i, self.merge_map[pair])) # (index, (new_id, rank))
            
            if not candidates:
                break
            
            # Apply the merge with the lowest rank (learned first)
            idx, (target_id, _) = min(candidates, key=lambda x: x[1][1])
            ids = ids[:idx] + [target_id] + ids[idx+2:]
        return ids

    def encode(self, text: str) -> list[int]:
        special_pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        chunks = re.split(special_pattern, text)
        pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        token_ids = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                token_ids.append(self.byte_to_id[chunk.encode("utf-8")])
            elif chunk:
                for match in pat.finditer(chunk):
                    piece_ids = list(match.group().encode("utf-8"))
                    token_ids.extend(self._apply_merges(piece_ids))
        return token_ids

    def decode(self, ids: list[int]) -> str:
        byte_data = b"".join(self.vocab[idx] for idx in ids)
        return byte_data.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            v_raw = json.load(f)
            vocab = {int(k): v.encode('latin-1') for k, v in v_raw.items()}
        
        merges = []
        with open(merges_path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.rstrip('\n').split(' ')
                if len(parts) == 2:
                    merges.append((parts[0].encode('latin-1'), parts[1].encode('latin-1')))
        
        return cls(vocab, merges, special_tokens)