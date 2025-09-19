import os
import regex as re
import heapq

from cs336_basics.pretokenization_example import find_chunk_boundaries


def pre_tokenize(chunk: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    """Pre-tokenize a chunk of text.
    """
    # Split with special tokens first
    special_pattern = "|".join(re.escape(special_token) for special_token in special_tokens)
    #print(f'special_pattern: {special_pattern}')
    split_documents = re.split(special_pattern, chunk)
    #print(f"Split documents: {split_documents}")
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # Now process each split document separately
    frequency_dict = {}
    for doc in split_documents:
        #print(f"Processing document: {doc}")
        tokens = re.findall(pattern, doc)
        for token in tokens:
            # convert to tuple of bytes
            encoded_token = token.encode("utf-8")
            token = tuple(bytes([b]) for b in encoded_token)
            frequency_dict[token] = frequency_dict.get(token, 0) + 1
    return frequency_dict

def find_bytes_pair_in_tuple(data_tuple, target_pair):
    for i in range(len(data_tuple) - 1):
        if data_tuple[i] == target_pair[0] and data_tuple[i+1] == target_pair[1]:
            return i  # Return the index of the first byte in the pair
    return -1  # Pair not found

def lex_desc_key(pair):
    # negate to force max-lex first
    return tuple(-x for x in pair[0] + pair[1])  # join both bytes and invert

def calculate_pair_frequencies(frequency_dict: dict[tuple[bytes], int]) -> dict[tuple[bytes, bytes], int]:
    """Calculate the frequency of each adjacent byte pair in the tokens.
    """
    pair_freq = {}
    pair_to_tuples = {}
    for token_tuple, freq in frequency_dict.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_freq[pair] = pair_freq.get(pair, 0) + freq
            # Record all token pairs
            if pair not in pair_to_tuples:
                pair_to_tuples[pair] = set()
            pair_to_tuples[pair].add(token_tuple)
    return pair_freq, pair_to_tuples

def merge_tokens(
        vocab: dict[int, bytes],
        frequency_dict: dict[tuple[bytes], int],
        vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Merge tokens based on frequency counts from multiple frequency dictionaries.
    """
    # calculate pair frequencies
    pair_freq, pair_to_tuples = calculate_pair_frequencies(frequency_dict)
    #print(f"Pair frequencies: {pair_freq}")

    merges = []
    while len(vocab) < vocab_size:
        heap = [(-freq, lex_desc_key(pair), pair) for pair, freq in pair_freq.items()]
        heapq.heapify(heap)
        freq, _, pair = heapq.heappop(heap)
        freq = -freq
        vocab[len(vocab)] = pair[0] + pair[1]
        merges.append(pair)
        new_frequency_dict = {}
        tuple_to_update = pair_to_tuples.get(pair, set())
        for token_tuple, count in frequency_dict.items():
            if token_tuple not in tuple_to_update:
                new_frequency_dict[token_tuple] = new_frequency_dict.get(token_tuple, 0) + count
                continue
            new_token_tuple = list(token_tuple)
            while True:
                index = find_bytes_pair_in_tuple(new_token_tuple, pair)
                if index == -1:
                    break
                # Merge the pair
                new_token_tuple = (
                    new_token_tuple[:index] +
                    [pair[0] + pair[1]] +
                    new_token_tuple[index + 2:]
                )
            new_frequency_dict[tuple(new_token_tuple)] = new_frequency_dict.get(tuple(new_token_tuple), 0) + count
        frequency_dict = new_frequency_dict
        pair_freq, pair_to_tuples = calculate_pair_frequencies(frequency_dict)
    return vocab, merges
 

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # init the vocabulary
    vocab = {}
    # init the special tokens to the vocabulary
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    # add the vocabulary with 256  utf-8 bytes
    vocab.update({i: bytes([i]) for i in range(256)})

    # initialize the merges
    merges = []

    # Pretokenize the input corpus
    total_frequency_dicts = {}
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            frequency_dict = pre_tokenize(chunk, special_tokens)
            for token, count in frequency_dict.items():
                total_frequency_dicts[token] = total_frequency_dicts.get(token, 0) + count 
    #print(frequency_dict)
    vocab, merges = merge_tokens(vocab, total_frequency_dicts, vocab_size)
    #print(f"Vocab after processing chunk: {vocab}")
    return vocab, merges