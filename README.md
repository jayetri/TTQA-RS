# TTQA-RS- A break-down prompting approach for Multi-hop Table-Text Question Answering with Reasoning and Summarization

##HybridQA:

Raw data:
    HybridQA: [HybridQA GitHub](https://github.com/wenhuchen/HybridQA/tree/master)
    WikiTables: [WikiTables](https://github.com/wenhuchen/WikiTables-WithLinks)

Get traced data:
    For HybridQA, we used the text retriever from HybridQA: [HybridQA GitHub](https://github.com/wenhuchen/HybridQA/tree/master)
    We used the table retriever from S3HQA: [S3HQA GitHub](https://github.com/lfy79001/S3HQA/tree/main)

Run summary, subanswers, and entity types:
    We have pre-made decomposed questions: HybridQA/question_dev.json and HybridQA/question_test.json

Run full model or full model without summary:
    This applies to both the dev and test sets.

##OTTQA:
   
    Raw data:
        OTTQA: [OTTQA GitHub](https://github.com/wenhuchen/OTT-QA)
    
    Experiment setup:
        We only used the dev set for the experiment.
        We used Hybrider as the retriever: [HybridQA GitHub](https://github.com/wenhuchen/HybridQA/tree/master)

    Decomposed questions:
        Similar to HybridQA, we have QTTQA/question_dev.json

    Run full model or full model without summary:
        This is applied to the dev set.
