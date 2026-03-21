from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from storage import store_vector

client = QdrantClient(url="http://localhost:6333")

client.delete_collection(collection_name="chunks")   # remove all data
store_vector.init_collection()                       # recreates collection (uses VECTOR_DIM)