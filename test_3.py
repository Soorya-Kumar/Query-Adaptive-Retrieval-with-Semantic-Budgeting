from embedding.encoding import embed
from storage.store_vector import search
import numpy as np

vec = np.array(embed("documents about technology"), dtype=float)
print(search(vec, top_k=10))