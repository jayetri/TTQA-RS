# TTQA-RS- A break-down prompting approach for Multi-hop Table-Text Question Answering with Reasoning and Summarization


HybridQA:

    Get traced data:
        For HybridQA, we used the text retriever from HybridQA: [HybridQA GitHub](https://github.com/wenhuchen/HybridQA/tree/master)
        We used the table retriever from S3HQA: S3HQA GitHub

    Run summary, subanswers, and entity types:
        We have pre-made decomposed questions: question_dev.json and question_test.json

    Run full model or full model without summary:
        This applies to both the dev and test sets.

OTTQA:

    Experiment setup:
        We only used the dev set for the experiment.
        We used Hybrider as the retriever: HybridQA GitHub

    Decomposed questions:
        Similar to HybridQA, we have question_dev.json

    Run full model or full model without summary:
        This is applied to the dev set.
