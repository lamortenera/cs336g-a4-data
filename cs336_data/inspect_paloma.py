import numpy as np
data = np.fromfile(
    "data/leaderboard_testrun/tokenized.bin",
    dtype=np.uint16)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer.decode(data[0:2000]))