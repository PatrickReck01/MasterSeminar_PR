import json
from KG_Score import KG_SCORE

import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

from scipy.stats import pearsonr, spearmanr

from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import torch.nn.functional as F

class BARTScorer:
    def __init__(self, model_name="facebook/bart-large-cnn", device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def score(self, source_texts, target_texts):
        source_texts = [source_texts]
        target_texts = [target_texts]
        """Compute BARTScore: log-likelihood of target given source."""
        scores = []
        for src, tgt in zip(source_texts, target_texts):
            input_ids = self.tokenizer.encode(src, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            labels = self.tokenizer.encode(tgt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=labels)
                # Sum the log-likelihood across all tokens
                log_likelihood = -outputs.loss.item() * labels.size(1)
                scores.append(log_likelihood)
        return scores [0]


def add_benchmark_scores_to_summEval(input_file, output_file):
    

    def compute_rouge(reference_text: str, summary_text: str) -> tuple:
        """
        Compute the F1 scores of ROUGE-1, ROUGE-2, and ROUGE-L between a reference text and a summary.

        Parameters:
        - reference_text (str): The original text.
        - summary_text (str): The generated summary.

        Returns:
        - tuple: (ROUGE-1 F1, ROUGE-2 F1, ROUGE-L F1)
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_text, summary_text)
        return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure
        

    
    def compute_bleu(summary, reference):

        return sentence_bleu([reference.split()], summary.split())
    
    def compute_meteor(reference_text: str, generated_summary: str) -> float:
        """
        Computes the METEOR score between a reference text and a generated summary.

        Args:
            reference_text (str): The ground truth text or reference summary.
            generated_summary (str): The LLM-generated summary to evaluate.

        Returns:
            float: METEOR score (between 0 and 1).
        """
        # Tokenize reference and hypothesis
        reference_tokens = word_tokenize(reference_text.lower())
        summary_tokens = word_tokenize(generated_summary.lower())

        # meteor_score expects list of reference tokens and a single hypothesis string
        return meteor_score([reference_tokens], summary_tokens)


    def compute_bert_score(summary, reference):
        P, R, F1 = bert_score([summary], [reference], lang="en", verbose=False)
        return float(P[0]),float(R[0]),float(F1[0])  # Return the precision score
    
    

    # Load the data
    data_list = []
    with open(input_file, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    print(f"Loaded {len(data_list)} entries from {input_file}")

    bart_scorer = BARTScorer()

    counter = 0
    skipped = 0
    for entry in data_list:
        try:
            text = entry['text']
            summary = entry['decoded']

            scores = entry.get('scores', {})
            
            # Compute the benchmark scores
            rouge_1, rouge_2, rouge_L = compute_rouge(summary, text)
            bleu = compute_bleu(summary, text)
            meteor = compute_meteor(text, summary)
            bert_p, bert_r, bert_f1 = compute_bert_score(summary, text)
            bart_score = bart_scorer.score(text, summary)

            scores['ROUGE-1'] = rouge_1
            scores['ROUGE-2'] = rouge_2
            scores['ROUGE-L'] = rouge_L
            scores['BLEU'] = bleu
            scores['METEOR'] = meteor
            scores['BERTScore_f1'] = bert_f1
            scores['BERTScore_p'] = bert_p
            scores['BERTScore_r'] = bert_r
            scores['BARTScore'] = bart_score

            entry['scores'] = scores
            
        except Exception as e:
            print(f"Error processing entry {counter}: {e}")
            skipped += 1
            continue


    
    # Save the results to a new JSONL file
    with open(output_file, 'w') as f:
        for entry in data_list:
            f.write(json.dumps(entry) + '\n')   

def add_kg_score_to_summEval(input_file, output_file, model):

    # Load the data
    data_list = []
    with open(input_file, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    print(f"Loaded {len(data_list)} entries from {input_file}")

    counter = 0
    skipped = 0
    for entry in data_list:
        try:
            text = entry['text']
            summary = entry['decoded']
            
            # Compute the KG_SCORE
            alignment, coverage = KG_SCORE_v2(text, summary, model=model,bio_text=False)
            entry['scores'] = {'kg_v2_mistral_alignment': alignment, 'kg_v2_mistral_coverage': coverage}
            print(f"Processed entry {counter}: KG_SCORE alignment={alignment}, coverage={coverage}")
            counter += 1
        except Exception as e:
            print(f"Error processing entry {counter}: {e}")
            skipped += 1
            continue
    
    # Save the results to a new JSONL file
    with open(output_file, 'w') as f:
        for entry in data_list:
            f.write(json.dumps(entry) + '\n')   


from Bio_QAG_Score.bio_qag import calc_bio_qag_score, get_assesment_questions, calc_QAG_metric
from Bio_QAG_Score.qag_utils import extract_info_from_qag_metric

def add_bio_qag_scores_to_summEval(input_file, output_file):
    # Load the data
    data_list = []
    with open(input_file, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    print(f"Loaded {len(data_list)} entries from {input_file}")

    counter = 0
    skipped = 0
    for entry in data_list:
        try:
            print(f"Processing entry {counter}...")
            text = entry['text']
            summary = entry['decoded']

            scores = entry['scores']
            
            # Calculate the QAG score
            qag_metric = calc_QAG_metric(original_text=text, sum_text=summary)

            info = extract_info_from_qag_metric(qag_metric)

            qag_alignment_score = info['alignement_score']
            qag_coverage_score = info['coverage_score']
            qag_total_score = (qag_alignment_score + qag_coverage_score) / 2

            scores['bio_qag__alignment'] = qag_alignment_score
            scores['bio_qag__coverage'] = qag_coverage_score
            scores['bio_qag__total'] = qag_total_score

            entry['scores'] = scores
            print(f"Processed entry {counter}: QAG alignment={qag_alignment_score}, coverage={qag_coverage_score}, total={qag_total_score}")
            counter += 1

        except Exception as e:
            print(f"Error processing entry {counter}: {e}")
            skipped += 1
            continue
    
    # Save the results to a new JSONL file
    with open(output_file, 'w') as f:
        for entry in data_list:
            f.write(json.dumps(entry) + '\n')







def calc_correlataion(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    human_total_scores = []
    human_consistency_scores = []
    human_relevance_scores = []

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    meteor_scores = []  
    bert_scores_f1 = []  
    bert_scores_precision = []
    bert_scores_recall = []
    bart_scores = []

    # qag_llama32_total_scores = []
    # qag_llama32_alignment_scores = []
    # qag_llama32_coverage_scores = []
    # kg_v2_deepseek7b_total_scores = []
    # kg_v2_deepseek7b_alignment_scores = []
    # kg_v2_deepseek7b_coverage_scores = []

    kg_v2_mistral_total_scores = []
    kg_v2_mistral_alignment_scores = []
    kg_v2_mistral_coverage_scores = []

    bio_qag_total_scores = []
    bio_qag_alignment_scores = []
    bio_qag_coverage_scores = []
    
    skipped_entries = 0
    for entry in data:

        if 'scores' not in entry.keys():
            print(f"Skipping entry {entry['id']} due to missing 'scores' key.")
            skipped_entries += 1
            continue

        scores = entry['scores']

        if 'kg_v2_mistral_alignment' not in scores.keys() or 'kg_v2_mistral_coverage' not in scores.keys():
            print(f"Skipping entry {entry['id']} due to missing 'kg_v2_mistral_alignment' or 'kg_v2_mistral_coverage' in scores.")
            skipped_entries += 1
            continue

        # Extract human scores
        try:
            avg_human_consistency = entry['avg_consistency']
            avg_human_relevance = entry['avg_relevance']
            avg_human_total = (avg_human_consistency + avg_human_relevance) / 2
        
            rouge1 = scores['ROUGE-1']
            rouge2 = scores['ROUGE-2']
            rougeL = scores['ROUGE-L']
            bleu = scores['BLEU']
            meteor = scores['METEOR']
            bert_f1 = scores['BERTScore_f1']
            bert_p = scores['BERTScore_p']
            bert_r = scores['BERTScore_r']
            bart  = scores['BARTScore']

            kg_v2_mistral_alignment = scores['kg_v2_mistral_alignment'] 
            kg_v2_mistral_coverage = scores['kg_v2_mistral_coverage']
            kg_v2_mistral_total = (kg_v2_mistral_alignment + kg_v2_mistral_coverage) / 2

            bio_qag_alignment = scores['bio_qag__alignment']
            bio_qag_coverage = scores['bio_qag__coverage']
            bio_qag_total = scores['bio_qag__total']
        except Exception as e:
            print(f"Skipping entry {entry['id']} due to missing benchmark scores: {e}")
            skipped_entries += 1
            continue

        human_total_scores.append(avg_human_total)
        human_consistency_scores.append(avg_human_consistency)
        human_relevance_scores.append(avg_human_relevance)

        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
        rougeL_scores.append(rougeL)
        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        bert_scores_f1.append(bert_f1)
        bert_scores_precision.append(bert_p)
        bert_scores_recall.append(bert_r)
        bart_scores.append(bart)

        kg_v2_mistral_total_scores.append(kg_v2_mistral_total)
        kg_v2_mistral_alignment_scores.append(kg_v2_mistral_alignment)
        kg_v2_mistral_coverage_scores.append(kg_v2_mistral_coverage)

        bio_qag_total_scores.append(bio_qag_total)
        bio_qag_alignment_scores.append(bio_qag_alignment)
        bio_qag_coverage_scores.append(bio_qag_coverage)

    # Calculate correlations


    results = {
        "rouge1": (pearsonr(rouge1_scores, human_total_scores), spearmanr(rouge1_scores, human_total_scores)),
        "rouge2": (pearsonr(rouge2_scores, human_total_scores), spearmanr(rouge2_scores, human_total_scores)),
        "rougeL": (pearsonr(rougeL_scores, human_total_scores), spearmanr(rougeL_scores, human_total_scores)),
        "bleu": (pearsonr(bleu_scores, human_total_scores), spearmanr(bleu_scores, human_total_scores)),
        "meteor": (pearsonr(meteor_scores, human_total_scores), spearmanr(meteor_scores, human_total_scores)),
        "bert_f1": (pearsonr(bert_scores_f1, human_total_scores), spearmanr(bert_scores_f1, human_total_scores)),
        "bart": (pearsonr(bart_scores, human_total_scores), spearmanr(bart_scores, human_total_scores)),

        "kg_v2_mistral_total": (pearsonr(kg_v2_mistral_total_scores, human_total_scores), spearmanr(kg_v2_mistral_total_scores, human_total_scores)),
        "kg_v2_mistral_alignment": (pearsonr(kg_v2_mistral_alignment_scores, human_consistency_scores), spearmanr(kg_v2_mistral_alignment_scores, human_consistency_scores)), # IMPORTANT WHICH SCORES ARE USED
        "kg_v2_mistral_coverage": (pearsonr(kg_v2_mistral_coverage_scores, human_relevance_scores), spearmanr(kg_v2_mistral_coverage_scores, human_relevance_scores)), # IMPORTANT WHICH SCORES ARE USED

        "bio_qag_total": (pearsonr(bio_qag_total_scores, human_total_scores), spearmanr(bio_qag_total_scores, human_total_scores)),
        "bio_qag_alignment": (pearsonr(bio_qag_alignment_scores, human_consistency_scores), spearmanr(bio_qag_alignment_scores, human_consistency_scores)), # IMPORTANT WHICH SCORES ARE USED
        "bio_qag_coverage": (pearsonr(bio_qag_coverage_scores, human_relevance_scores), spearmanr(bio_qag_coverage_scores, human_relevance_scores)), # IMPORTANT WHICH SCORES ARE USED

    }


    # Print results
    print(f"Skipped {skipped_entries} entries due to missing 'scores' key.")
    print(f"{'Metric':<10} {'Pearson':>10} {'Spearman':>10}")
    print("-" * 32)
    for metric, ((pearson_val, _), (spearman_val, _)) in results.items():
        print(f"{metric:<10} {pearson_val:>10.4f} {spearman_val:>10.4f}")
    print("-" * 32)





if __name__ == "__main__":
    input_file = '/Users/patrickreck/Documents/SeminarSoSe25/Datasets/SummEval/SummEVAL_normalised_data_v2_110sampled.jsonl'
    output_file = '/Users/patrickreck/Documents/SeminarSoSe25/Datasets/SummEval/SummEVAL_normalised_data_v2_110sampled_kg_v2_mistral.jsonl'
    model = 'mistral:latest'  # Adjust the model as needed

    # Add benchmark scores to SummEval
    #add_benchmark_scores_to_summEval('/Users/patrickreck/Documents/SeminarSoSe25/Datasets/SummEval/SummEVAL_normalised_data_v2_110sampled_kg_v2_mistral.jsonl', '/Users/patrickreck/Documents/SeminarSoSe25/Datasets/SummEval/SummEVAL_normalised_data_v2_110sampled_kg_v2_mistral_benchmarks.jsonl')

    #add_kg_score_to_summEval(input_file, output_file, model)
    #add_bio_qag_scores_to_summEval('/Users/patrickreck/Documents/SeminarSoSe25/Datasets/SummEval/SummEVAL_normalised_data_v2_110sampled_kg_v2_mistral_benchmarks.jsonl','/Users/patrickreck/Documents/SeminarSoSe25/Datasets/SummEval/SummEVAL_normalised_data_v2_110sampled_kg_v2_mistral_benchmarks_bio_qag.jsonl')

    calc_correlataion('SummEval_dataset/SummEVAL_110sampled_kg_v2_mistral_benchmarks_bio_qag.jsonl')
