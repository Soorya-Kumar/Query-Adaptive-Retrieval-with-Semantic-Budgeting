from anyio import sleep
from descriptors.testing import run
from embedding.pooling import pool
from utils import color_print

color_print("entering extraction part")
desc = run("""Ancient Egyptian civilization developed along the Nile River and lasted for over 3,000 years. Famous for pyramids, hieroglyphs, and pharaohs like Tutankhamun and Ramses II. The Egyptians made advances in mathematics, medicine, and engineering, and their culture continues to fascinate people today.""")
print(desc.model_dump_json(indent=2))

color_print("entering pooling part")

vec = pool(desc)
print(vec)  # e.g. [0.01, -0.02, ...]
print(vec.shape)  # e.g. (768,)