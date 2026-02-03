import regex as re
from collections import defaultdict
import multiprocessing
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _count_pretokens(chunk):
    counts = defaultdict(int)
    for match in re.finditer(PAT, chunk):
        counts[match.group().encode("utf-8")] += 1
    return counts

def get_stats(word_counts):
    counts = defaultdict(int)
    for word, freq in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            counts[pair] += freq
    return counts

def merge(word, pair, new_id):
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)

def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    for i, st in enumerate(special_tokens):
        vocab[256 + i] = st.encode("utf-8")
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    special_pattern = "|".join(re.escape(s) for s in special_tokens)
    chunks = re.split(f"({special_pattern})", text)
    
    text_chunks = [c for c in chunks if c and c not in special_tokens]
    with multiprocessing.Pool() as pool:
        counts_list = pool.map(_count_pretokens, text_chunks)
    
    word_counts = defaultdict(int)
    for counts in counts_list:
        for k, v in counts.items():
            word_counts[tuple(k)] += v # Store as tuple of ints for consistency
    
    merges = []
    num_merges = vocab_size - len(vocab)
    
    for i in range(num_merges):
        stats = get_stats(word_counts)
        if not stats: break
            
        best_pair = max(stats.items(), key=lambda item: (item[1], item[0]))[0]
        new_id = 256 + len(special_tokens) + i
        
        # Store merge as (bytes, bytes) for the tokenizer
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        new_word_counts = defaultdict(int)
        for word, freq in word_counts.items():
            new_word = merge(word, best_pair, new_id)
            new_word_counts[new_word] += freq
        word_counts = new_word_counts
    
    return vocab, merges

if __name__ == "__main__":
    input_file = "C:\\Users\\sushr\\OneDrive\\Documents\\assignment1-basics\\data\\TinyStoriesV2-GPT4-valid.txt"
    vocab, merges = train_bpe(input_file, 500, ["<|endoftext|>"])
    
    # Save Vocab (latin-1 for 1:1 byte mapping)
    serial_vocab = {k: v.decode('latin-1') for k, v in vocab.items()}
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(serial_vocab, f, indent=4)
    
    # Save Merges (latin-1 to ensure raw bytes are preserved)
    with open("merges.txt", "w", encoding="latin-1") as f:
        for p0, p1 in merges:
            f.write(f"{p0.decode('latin-1')} {p1.decode('latin-1')}\n")