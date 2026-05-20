import numpy as np
data = np.fromfile(
    "data/paloma_c4_100_domains_val.bin",
    dtype=np.uint16)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer.decode(data[0:2000]))