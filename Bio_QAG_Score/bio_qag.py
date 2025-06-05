import json
import re

from ollama_inference import run_ollama_inference
try:
    from qag_utils import calc_QAG_metric, extract_info_from_qag_metric
except ModuleNotFoundError:
    from Bio_QAG_Score.qag_utils import calc_QAG_metric, extract_info_from_qag_metric


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


def get_assesment_questions(text,model,num_questions=5):
    print("Generating assessment questions...")
    print("Using model:", model)
    prompt = f"""
TASK:
Generate exactly {num_questions} closed-ended questions (yes/no questions) based ONLY on the provided biomedical text.
- Focus on the key biomedical aspects (e.g., SNPs and their effects). These questions will be used to assess the quality of a summary of the text, so make sure they cover all the important aspects of the text.
- The questions must be clear, concise, and based only on information from the text.
- The questions will be used to evaluate a summary of the text, so they should cover the main aspects of the text.
- DO NOT answer the questions.
- DO NOT provide explanations or commentary.
- DO NOT output anything other than the questions.

OUTPUT FORMAT:
Return ONLY a JSON list of questions, with each question as a string.

Example:
[
    "Is SNP rs123 associated with an increased risk of disease X?",
    "Does mutation Y affect gene Z expression?",
    "Is protein A involved in metabolic pathway B?",
    "Was a significant correlation found between SNP rs456 and condition C?",
    "Is treatment D effective for patients with mutation E?"
]

IMPORTANT:
- Return ONLY the raw JSON list. 
- DO NOT include any other text (no introductions, no labels like 'Here are the questions:', no markdown formatting, no explanations).

Biomedical Text:
{text}

"""
    
    prompt_2 =f"""You are a biomedical research evaluator. Your task is to generate a list of {num_questions} close-ended factual questions that can be answered with the original text.

Each question should:
- Be answerable with "yes" or "no".
- Be phrased clearly, specifically, and unambiguously.
- Focus on facts that a high-quality summary should preserve.


Typical types of information to cover include (if present in the text):
- Key findings or conclusions
- Experimental results
- Clinical or biological associations
- Molecular mechanisms or pathways
- Mentioned genes, SNPs, or variants
- Study population or methodology
- Limitations or scope of the study
The questions should cover all parts of the text.

Return the result as a JSON list with each questions as a string in that list. Do not answer the questions, just provide the list of questions in JSON format as shown in the below example:

[
  "question1",
  "question2",
  "question3",
]

Now, here is the biomedical summary from which to generate the questions:

{text}

Generate the list of assessment questions and answers in JSON format only.
"""


    
    response = run_ollama_inference(model, prompt_2)
    print("Response from LLM:", response)

    try:
        questions = parse_json(response)
    except Exception as e: # Often times the model returns a list of questions without the last bracket
        response = response + ']'
        questions = parse_json(response)
    if questions is None:
        response = response + ']'
        questions = parse_json(response)

    assert isinstance(questions, list), "No questions generated. Please check the model output."
    if questions:
        print("Questions generated:", len(questions))

    return questions


def calc_bio_qag_score(original_text, summary,model="llama3.2:latest"):
    """
    Calculate the QAG score based on the generated questions and the original text.
    This is a placeholder function and should be implemented based on specific requirements.
    """

    # Generate assessment questions from the original text
    questions = get_assesment_questions(original_text,model=model,num_questions=10)
    print('--' * 40)
    print('# LOG: Generated assessment questions:') 
    for question in questions:
        print(question)
    print('--' * 40)




    ## Calculate the QAG score based on the generated questions and the summary
    metric = calc_QAG_metric(original_text=original_text,
                              sum_text=summary,
                              assessment_questions=questions,)

    info = extract_info_from_qag_metric(metric)

    # print('QAG Score Details:')
    # print(info)

    return info['final_score'], info['alignement_score'], info['coverage_score']
    




if __name__ == "__main__":
    

    # Example usage
    text = """
Single Nucleotide Polymorphisms (SNPs) are the most common type of genetic variation found in the human genome. A SNP occurs when a single nucleotide — the building blocks of DNA — is altered, potentially affecting the function of genes and, ultimately, the traits they control. These minute changes, though subtle, can have significant implications for human health and disease.

In biomedicine, SNPs are increasingly recognized as key players in understanding genetic predispositions to various conditions, including cancer, cardiovascular diseases, and neurological disorders. By identifying specific SNPs associated with disease, researchers can gain insights into the underlying molecular mechanisms and develop more effective diagnostic tools and treatments.

Additionally, SNPs are crucial in personalized medicine. By mapping individual SNP profiles, doctors can tailor treatment plans based on genetic susceptibility, ensuring more effective and targeted therapies. For instance, SNPs can influence how patients metabolize medications, leading to more precise dosing and reduced side effects.

In recent years, advances in genomic technologies, such as genome-wide association studies (GWAS), have accelerated the identification of disease-associated SNPs. As biomedicine continues to evolve, the study of SNPs promises to revolutionize healthcare by providing more personalized, predictive, and preventative strategies for patient care.
"""


    summ = """
Single Nucleotide Polymorphisms (SNPs) are common genetic variations where a single nucleotide in the DNA sequence is altered. These variations play a crucial role in biomedicine, influencing susceptibility to diseases like cancer, cardiovascular issues, and neurological disorders. SNPs are key in personalized medicine, allowing for tailored treatments based on genetic profiles. Advances in genomic technologies, such as genome-wide association studies (GWAS), have helped identify SNPs linked to diseases. The ongoing research into SNPs holds great potential for improving diagnostics, treatments, and prevention strategies, ultimately enhancing personalized healthcare and patient outcomes.
"""

    score, alignment, coverage = calc_improved_qag_score(text, summ,model="mistral:latest")
    print("QAG Score:", score)
    print("Coverage Score:", coverage)
    print("Alignment Score:", alignment)

    