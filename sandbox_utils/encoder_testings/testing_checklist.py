import torch
from part_1_encoder.transformer import Transformer

m = Transformer(vocab_size=27, num_positions=20, d_model=64, d_internal=64, num_classes=3, num_layers=1)
idx = torch.randint(0, 27, (20,))
lp, atts = m(idx)
print(lp.shape, len(atts), atts[0].shape)
