from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np

model = SentenceTransformer('all-MiniLM-L12-v2')


def embed_kg_as_triplets(kg):
    for edge in kg.edges(data=True):
        source_node = edge[0]
        target_node = edge[1]
        relation = edge[2]['label']
        
        # Get the embeddings for the source node, target node, and relation
        triplet_text = kg.nodes[source_node]['label']  + " " + relation + " " + kg.nodes[target_node]['label']
        # Embedd the text
        triplet_embedding = model.encode(triplet_text)
        
        # Store the triplet embedding in the current edge
        kg.edges[source_node, target_node]['embedding'] = triplet_embedding
        kg.edges[source_node, target_node]['triplet_text'] = triplet_text

    return kg