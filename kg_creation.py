import networkx as nx
import json
import re

from ollama_inference import run_ollama_inference




def parse_json(json_str):
    # Step 1: Try parsing directly
        try:
            relationships = json.loads(json_str)
            if isinstance(relationships, list):
                print("Successfully parsed JSON!")
                return relationships
        except json.JSONDecodeError:
            print("Initial JSON parsing failed! Trying cleanup...")

        # Step 2: Extract the JSON part using regex
        json_match = re.search(r"\[.*\]", json_str, re.DOTALL)
        if json_match:
            try:
                cleaned_json = json_match.group(0)
                relationships = json.loads(cleaned_json)
                if isinstance(relationships, list):
                    print("Successfully recovered JSON from raw output!")
                    return relationships
            except json.JSONDecodeError:
                print("JSON extraction failed.")
                raise ValueError("Failed to parse JSON from the response.")

        # print("Final attempt: Returning an empty list due to parsing failure.")
        # return []


def inference_kg_complete(text,model,bio_text=True):
    
    prompt_bio = f"""
You are a knowledge graph generation system. I want to use you to extract important entities and their relationships from a text. The knowledge graph should reflect all relevant information from the text.

Extract the key entities (such as for example genes, diseases, biological processes, or other relevant concepts) and their relationships (such as associations, roles, effects, etc.). Make sure your response covers all the important aspects of the text.

Text from which to extract the entities and relationships:
{text}

IMPORTANT:
- Extract all important entities and their relationships from the text.
- Include multiple relationships where applicable, representing different interactions or roles between entities.
- Try to capture a variety of relationships (not just 1-to-1 mappings) between two entities. 
- Extract as many relationships as possible, based on the text. I want the knowledge graph to be as detailed as possible
- Only use the text as a source for the relationships. Do not add any additional information or context.

Return the knowledge graph in the following format as a JSON array of dictionaries. Each dictionary must contain exactly 3 key-value pairs: 'entity1', 'relationship', and 'entity2'. Ensure that:

Example format:
[
    {{"entity1": "SNP", "relationship": "associated with", "entity2": "disease"}},
    {{"entity1": "Gene", "relationship": "plays a role in", "entity2": "disease 2"}}
]

Be sure to cover all important aspects of the text and avoid deviating from the specified format. ONLY return JSON code in the above format. Do not include any additional text or explanations. 

"""
    
    
    prompt_general = f"""
You are a knowledge graph generation system. I want to use you to extract important entities and their relationships from a news article text. The knowledge graph should reflect all relevant information from the text.

Given the following text, extract the key entities (such as people, places, organizations, events, topics, or other relevant concepts) and their relationships (such as associations, roles, effects, etc.).

Return the knowledge graph in the following format as a JSON array of dictionaries. Each dictionary must contain exactly 3 key-value pairs: 'entity1', 'relationship', and 'entity2'. Ensure that:
- Each entity and relationship reflects an aspect of the text.
- Include multiple relationships where applicable, representing different interactions or roles between entities.
- Capture a variety of relationships between multiple entities (not just 1-to-1 mappings) I want an interconnected graph.
- Make sure to include all important information, concepts and their interactions. The graph should have as many relationships as possible.

Example JSON format:
[
    {{"entity1": "Elon Musk", "relationship": "CEO of", "entity2": "Tesla"}},
    {{"entity1": "Paris", "relationship": "hosted", "entity2": "the 2024 Olympics"}}
]

Do not deviate from the example JSON format. ONLY return the JSON code in the above format. Do not include any additional text or explanations.

Text from which to extract the entities and relationships:
{text}
"""
  
    if bio_text:
        prompt = prompt_bio
    else:
        prompt = prompt_general

    
    res = run_ollama_inference(model,prompt)

    
    dict_list = parse_json(res)
    print("Relationships extracted:", len(dict_list))

    return dict_list




# Function to build the knowledge graph
def build_knowledge_graph(dict_list):
    print("Building knowledge graph...")
    G = nx.DiGraph()

    keys = list(dict_list[0].keys())
    print("Keys in the JSON response:", keys)

    for item in dict_list:
        try:
            entity1 = item.get(keys[0])
            relation = item.get(keys[1])
            entity2 = item.get(keys[2])
        except IndexError:
            print("Error: Expected 3 keys in the dictionary, but got:", item.keys())
            raise IndexError("Expected 3 keys in the dictionary, but got:", item.keys())
            continue

        if entity1 and entity2:
            G.add_node(entity1,label=entity1)
            G.add_node(entity2,label = entity2)
            G.add_edge(entity1, entity2, label=relation)
    
    return G


# def visualize_graph(G):
#     import matplotlib.pyplot as plt
#     print("Visualizing knowledge graph...")
#     plt.figure(figsize=(10, 6))
#     pos = nx.spring_layout(G)
#     labels = nx.get_edge_attributes(G, 'label')
#     nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#     plt.show()


def create_kg(text,model,bio_text=True):
    print("Creating knowledge graph...")
    # entities =  inference_kg_entities(text)
    # entitiy_list = []
    # for item in entities:
    #     entitiy_list.append(item['entity'])

    # relationships = inference_kg_relationships(text, entitiy_list)

    relationships = inference_kg_complete(text,model,bio_text)

    
    G = build_knowledge_graph(relationships)

    return G


###### GRAPH VISUALIZATION ######
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(G):
    print("Visualizing knowledge graph...")
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'label')

    # Draw the graph with smaller font size for node labels
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1', font_size=8)
    
    # Draw edge labels with smaller font size
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)

    plt.show()


if __name__ == "__main__":
    from kg_visualization import visualize_kg

    text = """
    Single nucleotide polymorphisms (SNPs) are the most common type of genetic variation in humans, occurring when a single nucleotide in the genome differs between individuals. 
    These genetic variations can have significant effects on health, influencing susceptibility to diseases and drug response. 
    For instance, the SNP rs1801133 in the MTHFR gene has been associated with an increased risk of cardiovascular diseases due to its impact on folate metabolism. 
    Similarly, rs7903146 in the TCF7L2 gene is strongly linked to type 2 diabetes, as it affects insulin secretion and glucose metabolism. Another well-studied SNP, rs16969968 in the CHRNA5 gene, has been connected to nicotine addiction and an increased risk of lung cancer. 
    Moreover, certain SNPs in the APOE gene, such as rs429358, play a crucial role in the risk of developing Alzheimer’s disease by affecting lipid metabolism and amyloid plaque formation in the brain. 
    Understanding these genetic variations helps researchers develop personalized medicine approaches, where treatments can be tailored to an individual’s genetic profile to maximize efficacy and minimize adverse effects.
    """


    G = create_kg(text,model="mistral:latest")

    visualize_kg(G)