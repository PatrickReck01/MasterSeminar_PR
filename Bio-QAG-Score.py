from ollama_inference import run_ollama_inference

import json
import re

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
             
def get_assesment_questions(text,model,num_questions=5):
    print("Generating assessment questions...")
    print("Using model:", model)
    
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


from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric

def calc_QAG_metric(original_text, sum_text,assessment_questions=None):
    test_case = LLMTestCase(
        input=original_text, 
        actual_output=sum_text,
    )

    # OpenAI API key is set in the environment variable
    # if assessment_questions is None:
    #     summarization_metric = SummarizationMetric(model=model)
    # else:
    #     summarization_metric = SummarizationMetric(model=model,
    #                                                assessment_questions=assessment_questions,)


    # Ollama
    if assessment_questions is None:
        summarization_metric = SummarizationMetric(async_mode=False)
    else:
        summarization_metric = SummarizationMetric(assessment_questions=assessment_questions,
                                                   async_mode=False)

    summarization_metric.measure(test_case=test_case)

    return summarization_metric



def extract_info_from_qag_metric(metric:SummarizationMetric):
    
    final_score = metric.score
    alignement_score = metric.score_breakdown['Alignment']
    coverage_score = metric.score_breakdown['Coverage']

    assesment_questions = metric.assessment_questions
    coverage_verdicts = metric.coverage_verdicts
    alignment_verdicts = metric.alignment_verdicts

    claims = metric.claims

    truths = metric.truths

    res = {
        'final_score': final_score,
        'alignement_score': alignement_score,
        'coverage_score': coverage_score,
        'assesment_questions': assesment_questions,
        'coverage_verdicts': coverage_verdicts,
        'alignment_verdicts': alignment_verdicts,
        'claims': claims,
        'truths': truths
    }
    return res




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
    # IMPORTANT:
    # Make sure to run the following command in the terminal with your conda environment activated to use local ollama model for evaluation: deepeval set-ollama mistral:latest

    # Example text and summary
    text = """
Single Nucleotide Polymorphisms (SNPs) are the most common type of genetic variation found in the human genome. A SNP occurs when a single nucleotide — the building blocks of DNA — is altered, potentially affecting the function of genes and, ultimately, the traits they control. These minute changes, though subtle, can have significant implications for human health and disease.

In biomedicine, SNPs are increasingly recognized as key players in understanding genetic predispositions to various conditions, including cancer, cardiovascular diseases, and neurological disorders. By identifying specific SNPs associated with disease, researchers can gain insights into the underlying molecular mechanisms and develop more effective diagnostic tools and treatments.

Additionally, SNPs are crucial in personalized medicine. By mapping individual SNP profiles, doctors can tailor treatment plans based on genetic susceptibility, ensuring more effective and targeted therapies. For instance, SNPs can influence how patients metabolize medications, leading to more precise dosing and reduced side effects.

In recent years, advances in genomic technologies, such as genome-wide association studies (GWAS), have accelerated the identification of disease-associated SNPs. As biomedicine continues to evolve, the study of SNPs promises to revolutionize healthcare by providing more personalized, predictive, and preventative strategies for patient care.
"""


    summ = """Single Nucleotide Polymorphisms (SNPs) are common genetic variations where a single nucleotide in the DNA sequence is altered. These variations play a crucial role in biomedicine, influencing susceptibility to diseases like cancer, cardiovascular issues, and neurological disorders. SNPs are key in personalized medicine, allowing for tailored treatments based on genetic profiles. Advances in genomic technologies, such as genome-wide association studies (GWAS), have helped identify SNPs linked to diseases. The ongoing research into SNPs holds great potential for improving diagnostics, treatments, and prevention strategies, ultimately enhancing personalized healthcare and patient outcomes."""

    score, alignment, coverage = calc_bio_qag_score(text, summ,model="mistral:latest")
    print("Bio-QAG Score:", score)
    print("Coverage Score:", coverage)
    print("Alignment Score:", alignment)