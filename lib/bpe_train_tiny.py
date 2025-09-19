from bpe_train import train_bpe

input_path = "data/TinyStoriesV2-GPT4-valid.txt"

vocab, merges = train_bpe(input_path, 1000, ["<|endoftext|>"])
