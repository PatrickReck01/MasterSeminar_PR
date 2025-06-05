from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric


def calc_QAG_metric(original_text, sum_text,assessment_questions=None):
    test_case = LLMTestCase(
        input=original_text, 
        actual_output=sum_text,
    )


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























if __name__ == "__main__":
    
    text_paper="""
Abstract

Alzheimer's disease (AD) is a multifactorial neurodegenerative disorder characterized by progressive cognitive decline. It is the most common form of dementia, accounting for 60-80% of all cases. Despite significant advances in understanding the genetic and environmental factors contributing to AD, the underlying mechanisms remain poorly understood. One of the most promising avenues of research involves investigating the role of Single Nucleotide Polymorphisms (SNPs) in AD susceptibility. SNPs are the most common form of genetic variation among humans and have been implicated in the development of various diseases, including neurodegenerative disorders. This review aims to provide an overview of the role of SNPs in Alzheimer's disease, examining key SNPs identified in large-scale genetic studies and their potential contribution to disease pathogenesis.

Introduction

Alzheimer’s disease (AD) is a complex disorder primarily affecting the elderly, characterized by cognitive impairments such as memory loss, language difficulties, and a decline in executive function. The pathophysiology of AD involves the accumulation of amyloid-beta plaques, tau tangles, and neuroinflammation, but the genetic underpinnings of the disease are multifactorial. Several genetic and environmental factors have been associated with AD risk, with the APOE gene being the most well-known risk factor. However, the discovery of SNPs associated with AD has provided valuable insights into the disease's molecular basis.

SNPs are variations in a single nucleotide in the genome, where a single base pair is substituted by another. These genetic variations can occur in both coding and non-coding regions of the genome and may influence gene expression, protein function, and disease susceptibility. In this context, SNPs have become key targets in understanding the genetic basis of AD.

Genetic Basis of Alzheimer's Disease

The first major breakthrough in the genetic study of Alzheimer's disease was the discovery of the APOE ε4 allele, which has been consistently associated with an increased risk of developing AD. APOE is involved in lipid metabolism and neuronal repair, and the ε4 allele is thought to influence amyloid-beta processing, accelerating plaque formation. However, the effect of APOE ε4 on AD risk is not absolute, and not all carriers of the allele develop the disease. This has led researchers to explore other genetic variants that may modulate AD risk.

The advent of genome-wide association studies (GWAS) has significantly expanded our understanding of the genetic landscape of Alzheimer's disease. GWAS has identified numerous SNPs associated with AD risk, many of which map to genes involved in immune response, lipid metabolism, and neuronal function. One of the most notable findings is the association of SNPs in the TREM2 gene, a key regulator of microglial activation. Microglia are the resident immune cells in the brain, and their dysfunction is believed to play a central role in AD pathology.

Key SNPs and Their Contribution to Alzheimer's Disease

APOE ε4 Allele
The APOE gene is located on chromosome 19 and encodes a protein involved in the transport of cholesterol and other lipids. The APOE ε4 allele has been identified as the most significant genetic risk factor for late-onset Alzheimer's disease. Studies have shown that individuals carrying one copy of the ε4 allele have an increased risk of developing AD, while those with two copies face an even greater risk. The exact mechanism by which APOE ε4 contributes to AD is not fully understood, but it is thought to promote the accumulation of amyloid-beta plaques, impair synaptic function, and disrupt neuronal repair mechanisms.

TREM2 (Triggering Receptor Expressed on Myeloid Cells 2)
TREM2 is a gene located on chromosome 6 that encodes a receptor involved in microglial activation and response to injury. Recent studies have shown that variants of TREM2, particularly the R47H SNP, are associated with an increased risk of developing Alzheimer's disease. The R47H mutation impairs the normal function of TREM2, leading to defective microglial activation and an inability to clear amyloid-beta plaques efficiently. This results in chronic neuroinflammation, a key feature of AD pathology.

CLU (Clusterin)
The CLU gene, located on chromosome 8, encodes a protein involved in various cellular processes, including protein folding, apoptosis, and lipid transport. SNPs in the CLU gene, particularly rs11136000, have been associated with an increased risk of AD in several large-scale GWAS. Clusterin is believed to play a role in the clearance of amyloid-beta plaques and may modulate the immune response in the brain. The association of CLU with AD highlights the importance of cellular stress response pathways in the pathogenesis of the disease.

PICALM (Phosphatidylinositol Binding Clathrin Assembly Protein)
The PICALM gene, located on chromosome 11, encodes a protein involved in endocytosis and synaptic vesicle recycling. SNPs in PICALM, particularly rs3851179, have been associated with AD risk in multiple GWAS studies. PICALM is thought to influence the trafficking and clearance of amyloid-beta, and alterations in its function may lead to an accumulation of amyloid plaques in the brain. Additionally, PICALM is involved in the regulation of synaptic function, and disruptions in this pathway could contribute to the cognitive decline observed in AD.

CR1 (Complement Receptor 1)
The CR1 gene, located on chromosome 1, encodes a receptor involved in the complement system, which plays a crucial role in immune defense. Variants of CR1, particularly rs6656401, have been associated with an increased risk of AD in multiple studies. CR1 is thought to modulate the clearance of amyloid-beta by microglia through the complement pathway. The identification of CR1 as a genetic risk factor for AD has highlighted the importance of immune-related pathways in the disease.

Mechanisms of SNP Action in Alzheimer's Disease

The functional consequences of SNPs associated with Alzheimer's disease can be complex and may involve changes in gene expression, protein function, or cellular processes. For example, SNPs in the TREM2 gene result in reduced microglial function, impairing the ability of these cells to clear amyloid-beta plaques and leading to neuroinflammation. Similarly, SNPs in the CLU and PICALM genes may affect the clearance of amyloid-beta and disrupt synaptic function.

In addition to these effects on amyloid-beta metabolism, many of the SNPs identified in AD GWAS are involved in immune signaling, lipid metabolism, and cellular stress response pathways. These pathways are thought to interact in a complex network that influences the development and progression of the disease. Understanding how these SNPs contribute to AD pathogenesis may lead to the identification of novel therapeutic targets.

Conclusion

The discovery of SNPs associated with Alzheimer's disease has provided valuable insights into the genetic basis of this complex disorder. While the APOE ε4 allele remains the most significant genetic risk factor, other SNPs, such as those in the TREM2, CLU, PICALM, and CR1 genes, have been identified as important contributors to disease risk. These findings underscore the importance of immune-related pathways, lipid metabolism, and cellular stress responses in the pathogenesis of AD. As research continues, the identification of additional SNPs and the exploration of their functional consequences will be crucial in developing personalized treatments for Alzheimer's disease.

Acknowledgments

The authors would like to acknowledge the contributions of all researchers and participants involved in the large-scale genetic studies of Alzheimer's disease. Funding for this work was provided by the Alzheimer's Research Foundation.

References

Corder, E. H., et al. (1993). "Gene dose of apolipoprotein E type 4 and the risk of Alzheimer’s disease in late onset families." Science.
Guerreiro, R., et al. (2013). "TREM2 variants in Alzheimer's disease." The New England Journal of Medicine.
Lambert, J.-C., et al. (2013). "Genome-wide association study identifies variants at CLU and PICALM associated with Alzheimer's disease." Nature Genetics.
Hollingworth, P., et al. (2011). "Common variants at ABCA7, APOE, and other loci are associated with Alzheimer’s disease." Nature.
"""

    text_summary="""
This paper reviews the role of Single Nucleotide Polymorphisms (SNPs) in Alzheimer's disease (AD), a neurodegenerative disorder causing cognitive decline. While the APOE ε4 allele is the most well-established genetic risk factor for AD, genome-wide association studies (GWAS) have identified several other SNPs associated with the disease. These SNPs primarily affect genes involved in immune response, lipid metabolism, and neuronal function.

The paper discusses key SNPs, including those in the TREM2 gene, which regulates microglial activation. Variants such as R47H impair microglial function, preventing the efficient clearance of amyloid-beta plaques and leading to neuroinflammation. Similarly, SNPs in the CLU gene, which encodes a protein involved in protein folding and lipid transport, are linked to AD risk and may influence amyloid-beta clearance. Other important SNPs are found in PICALM, involved in synaptic vesicle recycling, and CR1, a gene associated with the complement system that helps clear amyloid-beta through immune pathways.

The mechanisms through which these SNPs contribute to AD include alterations in amyloid-beta metabolism, synaptic function, and immune signaling. These findings emphasize the complexity of AD's genetic basis and suggest that immune-related pathways and lipid metabolism play a significant role in disease pathogenesis.

Overall, the paper highlights the importance of SNPs in advancing the understanding of Alzheimer's disease and suggests that further genetic research could lead to novel therapeutic targets for the disease.
"""


    # Assuming metric is already populated with data
    metric = calc_QAG_metric(text_paper, text_summary)
    result = extract_info_from_qag_metric(metric)

    print('Final Score:', result['final_score'])
    