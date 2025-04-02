import api
from openai import OpenAI
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore, RetrievalMode
import uuid
from tqdm import tqdm
import policy as pl

openai_client = OpenAI(api_key=api.OPENAI_API)
qdrant_client = QdrantClient(url=api.QDRANT_URL, api_key=api.QDRANT_API)

def dense_embed(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def sparse_embed(text):
    model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    sparse_embedding = model.embed(text)
    return list(sparse_embedding)[0]

def create_collection(collection_name, dense_embedding_dim):
    if qdrant_client.collection_exists(collection_name=collection_name):
        qdrant_client.delete_collection(collection_name=collection_name)
        print(f"Deleted old version collection {collection_name}")

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=dense_embedding_dim,
            distance=models.Distance.COSINE,
        ),
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )
    print(f"Collection {collection_name} initialized.")

def add_chunk(chunks, collection_name):
    for chunk in tqdm(chunks, desc=f"Embedding chunks and storing the embeddings."):
        content = chunk.content
        dense_embedding = dense_embed(content)
        sparse_embedding = sparse_embed(content)

        # dense
        doc_id = f"{uuid.uuid4()}" 
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    payload=chunk.__dict__,
                    vector=dense_embedding
                ),
            ]
        )
        # sparse
        doc_id = f"{uuid.uuid4()}"
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    payload=chunk.__dict__,
                    vector={
                        "sparse": {  
                            "indices": list(sparse_embedding.indices),
                            "values": list(sparse_embedding.values)
                        }
                    },
                ),
            ]
        )
    print(f"All chunks are saved to {collection_name}")

def add_policies(policies:dict, collection_name):
    for item in tqdm(policies, desc=f"Embedding policies and storing the embeddings."):
        policy = pl.Policy.from_dict(item)
        content = policy.policy + policy.effect
        dense_embedding = dense_embed(content)
        sparse_embedding = sparse_embed(content)

        # dense
        doc_id = f"{uuid.uuid4()}" 
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    payload=policy.__dict__,
                    vector=dense_embedding
                ),
            ]
        )
        # sparse
        doc_id = f"{uuid.uuid4()}"
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    payload=policy.__dict__,
                    vector={
                        "sparse": {  
                            "indices": list(sparse_embedding.indices),
                            "values": list(sparse_embedding.values)
                        }
                    },
                ),
            ]
        )
    print(f"All chunks are saved to {collection_name}")

def add_knowledges(knowledges:dict, collection_name):
    for knowledge in tqdm(knowledges, desc=f"Embedding knowledges and storing the embeddings."):
        content = knowledge['summary']
        dense_embedding = dense_embed(content)
        sparse_embedding = sparse_embed(content)

        # dense
        doc_id = f"{uuid.uuid4()}" 
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    payload=None,
                    vector=dense_embedding
                ),
            ]
        )
        # sparse
        doc_id = f"{uuid.uuid4()}"
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    payload=None,
                    vector={
                        "sparse": {  
                            "indices": list(sparse_embedding.indices),
                            "values": list(sparse_embedding.values)
                        }
                    },
                ),
            ]
        )
    print(f"All chunks are saved to {collection_name}")

def get_collection(collection_name, dense_embedding_function, sparse_embedding_function, retrieval_mode=RetrievalMode.HYBRID):
    collection = QdrantVectorStore.from_existing_collection(
        embedding=dense_embedding_function,
        sparse_embedding=sparse_embedding_function,
        collection_name=collection_name,
        url=api.QDRANT_URL,
        api_key=api.QDRANT_API,
        retrieval_mode=retrieval_mode,
        sparse_vector_name="sparse"
    )
    return collection

def retrieve_payload(document, collection):
    point_id = document.metadata["_id"]
    point = qdrant_client.retrieve(
        collection_name=collection.collection_name,
        ids=[point_id],
        with_payload=True,  
        with_vectors=False  
    )
    payload = point[0].payload
    return payload

