import numpy as np
from sentence_transformers import CrossEncoder
import vector_store as vs

class Retriever:
    def __init__(self, query, collection):
        self.query = query
        self.collection = collection
        self.found_docs = []
        self.reranked_docs = []
        self.filtered_embeddings = []
        self.filtered_contents = []

    def similarity_search(self, k):
        self.found_docs = self.collection.similarity_search(self.query, k=k, with_vectors=True)

    def similarity_search_with_filter(self, k, filter):
        self.found_docs = self.collection.similarity_search(self.query, k=k, filter=filter, with_vectors=True)
    
    def rerank(self, attribute, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        cross_encoder = CrossEncoder(model_name)
        rerank_input = []
        for document in self.found_docs:
            payload = vs.retrieve_payload(document, self.collection)
            if isinstance(attribute, list):
                content = ""
                for item in attribute:
                    value = payload.get(item, "")
                    content += value
            else:
                content = payload.get(attribute, "")
            rerank_input.append((self.query, content))
        scores = cross_encoder.predict(rerank_input)

        for i, doc in enumerate(self.found_docs):
            doc.metadata["cross_score"] = scores[i]

        self.reranked_docs = sorted(
            self.found_docs,
            key=lambda x: x.metadata["cross_score"],
            reverse=True
        )

    def cos_filtering(self, attribute, threshold, k):
        for doc in self.reranked_docs:
            payload = vs.retrieve_payload(doc, self.collection)
            if isinstance(attribute, list):
                content = ""
                for item in attribute:
                    value = payload.get(item, "")
                    content += value
            else:
                content = payload.get(attribute, "")
            embedding = vs.dense_embed(content)  

            if cosine_similarity_filter(embedding, self.filtered_embeddings, threshold):
                self.filtered_embeddings.append(embedding)
                self.filtered_contents.append(content)

            if len(self.filtered_contents) >= k:
                break
    
def cosine_similarity_filter(candidate, selected_vectors, threshold):
    if not selected_vectors :
        return True
    
    for vector in selected_vectors:
        dot_product = np.dot(candidate, vector)
        norm_candidate = np.linalg.norm(candidate)
        norm_vector = np.linalg.norm(vector)
        similarity =  dot_product / (norm_candidate * norm_vector)
        if similarity > threshold:
            return False
    return True

