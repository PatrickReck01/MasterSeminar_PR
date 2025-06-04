from kg_creation import create_kg
from kg_embedding import embed_kg_as_triplets

from colorama import init, Fore, Style
init(autoreset=True)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np




#============= Graph Comparison =============#
# Compute cosine similarity between two embeddings
def compute_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

def calculate_triplet_similarity(kg1, kg2,threshold=0.8):
    
    matched = 0
    for edge_kg1 in kg1.edges(data=True):
        triplet_embedding_kg1 = edge_kg1[2]['embedding']
        triplet_text_kg1 = edge_kg1[2]['triplet_text']

        max_similarity = 0

        for edge_kg2 in kg2.edges(data=True):
            triplet_embedding_kg2 = edge_kg2[2]['embedding']
            triplet_text_kg2 = edge_kg2[2]['triplet_text']

            # Calculate cosine similarity between the triplet embeddings
            similarity = compute_cosine_similarity(triplet_embedding_kg1, triplet_embedding_kg2)

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = triplet_text_kg2
        
        if max_similarity >= threshold:
            matched += 1
            print(f"Matched triplet from kg1: {triplet_text_kg1} with triplet from kg2: {best_match} (similarity: {max_similarity})")

    # Calculate the similarity score as the ratio of matched triplets to total triplets in kg1
    total_triplets = len(kg1.edges)
    similarity_score = (matched / total_triplets)

    print(f"Similarity score: {similarity_score} / Matched triplets: {matched} / Total triplets {total_triplets}")

    return similarity_score
def compare_kgs(kg_text, kg_sum):

    ## Compute score
    print(Fore.RED + "## LOG: Computing score...")
    coverage = calculate_triplet_similarity(kg_text, kg_sum,threshold=0.77) # Option 4
    alignment = calculate_triplet_similarity(kg_sum, kg_text,threshold=0.77) # Option 4


    # Filter out NaN values
    if np.isnan(coverage) or np.isnan(alignment):
        raise ValueError("Score is NaN. Please check the input data.")
    
    print(Fore.GREEN + "## LOG: Score: ", f"Coverage: {coverage}, Alignment: {alignment}")


    return alignment, coverage





def create_embedded_kg(text:str,model,bio_text=True):
    ## Create KG from text and summary
    print(Fore.RED + "## LOG: Creating kg from text:")
    kg_text = create_kg(text,model,bio_text)
    print(Fore.RED + "## LOG: KG from text created.")


    ## Create embeddings for both KGs
    print(Fore.RED + "## LOG: Computing embeddings for both")
    embedded_kg_text = embed_kg_as_triplets(kg_text)


    return embedded_kg_text




def KG_SCORE(text:str,summary:str,model:str,bio_text=True):

    embedded_kg_text = create_embedded_kg(text,model,bio_text)

    embedded_kg_sum = create_embedded_kg(summary,model,bio_text)
    
    alignment, coverage = compare_kgs(embedded_kg_text, embedded_kg_sum)
    
    
    return alignment, coverage


if __name__ == "__main__":
    text = """
Single nucleotide polymorphisms (SNPs) are the most common type of genetic variation in humans, occurring when a single nucleotide in the genome differs between individuals. 
These genetic variations can have significant effects on health, influencing susceptibility to diseases and drug response. 
For instance, the SNP rs1801133 in the MTHFR gene has been associated with an increased risk of cardiovascular diseases due to its impact on folate metabolism. 
Similarly, rs7903146 in the TCF7L2 gene is strongly linked to type 2 diabetes, as it affects insulin secretion and glucose metabolism. Another well-studied SNP, rs16969968 in the CHRNA5 gene, has been connected to nicotine addiction and an increased risk of lung cancer. 
Moreover, certain SNPs in the APOE gene, such as rs429358, play a crucial role in the risk of developing Alzheimer’s disease by affecting lipid metabolism and amyloid plaque formation in the brain. 
Understanding these genetic variations helps researchers develop personalized medicine approaches, where treatments can be tailored to an individual’s genetic profile to maximize efficacy and minimize adverse effects.
    """

    summary = """
Single nucleotide polymorphisms (SNPs) are the most common genetic variations in humans, occurring when a single nucleotide differs between individuals. 
These variations can influence disease susceptibility and how individuals respond to medications. 
Specific SNPs, such as rs1801133 in the MTHFR gene and rs7903146 in the TCF7L2 gene, are linked to cardiovascular diseases and type 2 diabetes, respectively. 
Others, like rs16969968 in CHRNA5 and rs429358 in APOE, are associated with nicotine addiction, lung cancer, and Alzheimer’s disease. 
Understanding SNPs is essential for advancing personalized medicine, allowing treatments to be tailored to a person’s genetic makeup.
"""

    score = KG_SCORE(text,summary,model='mistral:latest',bio_text=True)
    print("Score: ", score)
    