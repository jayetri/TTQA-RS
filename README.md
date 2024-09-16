# TTQA-RS- A break-down prompting approach for Multi-hop Table-Text Question Answering with Reasoning and Summarization
This repository contains the code for - [TTQA-RS- A break-down prompting approach for Multi-hop Table-Text Question Answering with Reasoning and Summarization](https://arxiv.org/abs/2406.14732v1) 
## Chain-of-Thought
All CoT examples are in the examples folder ( HybridQA/examples and OTTQA/examples ), including 1-4 shot of: CoT, Standard Prompting, Full Model with Summary, and Full Model without Summary.

## HybridQA:

### We obtain the raw data from:
HybridQA: [HybridQA GitHub](https://github.com/wenhuchen/HybridQA/tree/master)
WikiTables: [WikiTables](https://github.com/wenhuchen/WikiTables-WithLinks)

### Get traced data from retriever:

Inside HybridQA/Retriever
    First convert table row to summary using bart.
    Then generate key node for each row and wiki-link in training set.
    Fine-tuning retriever on pretrained DPR. Batch size: 16, epoch: 8.
    Compute text similarities based on pretrained DPR and row scores.

### Run Summary, Subanswers, and Entity Types:

Generating Summary:
    We used a 0-shot LLAMA-3 70B model to generate summaries. 100 examples are contained in HybridQA/sample_summaries folder.

Decomposing Questions:
    We used a 2-shot LLAMA-3 70B model to decompose questions. Pre-made decomposed questions are provided in HybridQA/question_dev.json and HybridQA/question_test.json.

Identifying Entity Types:
    We used SpaCy to identify entity types in both the main questions and subquestions.

### Run Full Model or Full Model without Summary:

Applicability:
    This process applies to both the development (dev) and test sets.

Generating Subanswers:
    First, load the model to generate subanswers for the subquestions, and save the results to the outputs folder. During this process, entity types and summaries are included to achieve better results for the subquestions.

Answering Main Questions:
    Use the subanswers, along with the entity types from the main questions and the summaries(summary can be commented out for no-summary version) , to help answer the main questions.

### Evaluation Method:

To evaluate our models, we employ the same rigorous evaluation metrics used by other studies utilizing the HybridQA dataset. Specifically, we calculate the Exact Match (EM) and F1 Score:

Exact Match (EM):
This metric measures the percentage of predictions that exactly match the ground truth answers. It is a strict evaluation metric that requires the predicted answer to be identical to the correct answer in terms of both content and format.

F1 Score:
The F1 Score is a harmonic mean of precision and recall, providing a balanced measure of the model's accuracy. Precision measures the proportion of correctly predicted answers out of all predicted answers, while recall measures the proportion of correctly predicted answers out of all ground truth answers. The F1 Score thus captures both the completeness and exactness of the model's predictions.


## OTTQA:
   
### We obtain the raw data from:
OTTQA: [OTTQA GitHub](https://github.com/wenhuchen/OTT-QA)
    
### Experiment setup:
We only used the dev set for the experiment.
We used HybridQA's retriever to retrieve the wiki links of dev set: [HybridQA GitHub](https://github.com/wenhuchen/HybridQA/tree/master)

### Run Summary, Subanswers, and Entity Types:

Generating Summary:
    We used a 0-shot LLAMA-3 70B model to generate summaries. 100 examples are contained in OTTQA/sample_summary folder.

Decomposing Questions:
    We used a 2-shot LLAMA-3 70B model to decompose questions. Pre-made decomposed questions are provided in OTTQA/question_dev.json.

Identifying Entity Types:
    We used SpaCy to identify entity types in both the main questions and subquestions.
    
### Run Full Model or Full Model without Summary:

Applicability:
    This process applies to the development (dev) set.

Generating Subanswers:
    First, load the model to generate subanswers for the subquestions, and save the results to the outputs folder. During this process, entity types and summaries are included to achieve better results for the subquestions.

Answering Main Questions:
    Use the subanswers, along with the entity types from the main questions and the summaries(summary can be commented out for no-summary version) , to help answer the main questions.

### Evaluation Method:

To evaluate our models, we employ the same rigorous evaluation metrics used by other studies utilizing the OTTQA dataset. Specifically, we calculate the Exact Match (EM) and F1 Score:

Exact Match (EM):
This metric measures the percentage of predictions that exactly match the ground truth answers. It is a strict evaluation metric that requires the predicted answer to be identical to the correct answer in terms of both content and format.

F1 Score:
The F1 Score is a harmonic mean of precision and recall, providing a balanced measure of the model's accuracy. Precision measures the proportion of correctly predicted answers out of all predicted answers, while recall measures the proportion of correctly predicted answers out of all ground truth answers. The F1 Score thus captures both the completeness and exactness of the model's predictions.

