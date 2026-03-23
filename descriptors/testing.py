from descriptors.extractor import extract_descriptors
from descriptors.postprocessing import postprocess
from descriptors.schema import ChunkDescriptor


def run(chunk_text: str) -> ChunkDescriptor:
    raw = extract_descriptors(chunk_text)
    print("Raw output from model:")
    # print(raw)
    return postprocess(raw)

