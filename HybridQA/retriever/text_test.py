import torch
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import json
import os

# Load models
model_name = 'facebook/dpr-question_encoder-single-nq-base'
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
question_encoder = DPRQuestionEncoder.from_pretrained(model_name)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

def encode_text(text, tokenizer, encoder):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        embedding = encoder(**inputs).pooler_output
    return embedding

def calculate_similarity(emb1, emb2):
    return torch.nn.functional.cosine_similarity(emb1, emb2).item()

# Extract all wiki links from answer-node
def extract_wiki_links(wikilink):
    wiki_links = []
    if not wikilink or not isinstance(wikilink, list):
        print("Warning: answer_node is empty or not a list")
        return wiki_links
    
    for answer in wikilink:
        if not isinstance(answer, (list, tuple)) or len(answer) < 3:
            print(f"Warning: Unexpected answer format: {answer}")
            continue
        
        wiki_link = answer[2]
        if not isinstance(wiki_link, str):
            print(f"Warning: wiki_link is not a string: {wiki_link}")
            continue
        
        if wiki_link.startswith("/wiki/"):
            wiki_links.append(wiki_link)
    
    print(f"Extracted wiki links: {wiki_links}")
    return wiki_links

# Retrieve context data from request_tok
def retrieve_context_from_wiki_links(wiki_links, request_tok_dir, table_id):
    context = []
    context_links = []  # New list to store corresponding wiki links
    request_file_path = os.path.join(request_tok_dir, f"{table_id}.json")
    
    if not os.path.exists(request_file_path):
        print(f"Request file not found: {request_file_path}")
        return context, context_links

    with open(request_file_path, 'r', encoding='utf-8') as request_file:
        request_data = json.load(request_file)

    for link in wiki_links:
        if link in request_data:
            context.append(request_data[link])
            context_links.append(link)  # Store the corresponding link
        else:
            print(f"Link '{link}' not found in request_data keys: {list(request_data.keys())}")

    return context, context_links

def write_result_to_json(output_dir, table_id, result):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{table_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def load_question_and_rows(input_json_path, row_retrieve_dir, request_tok_dir, output_dir, num_questions=None):
    if not all(isinstance(path, str) for path in [input_json_path, row_retrieve_dir, request_tok_dir, output_dir]):
        raise TypeError("All directory and file path arguments must be strings")
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    if num_questions is None:
        num_questions = len(questions_data)
    else:
        num_questions = min(num_questions, len(questions_data))

    for i, item in enumerate(questions_data[:num_questions]):
        try:
            table_id = item['table_id']
            question = item['question']

            retrieve_file_path = os.path.join(row_retrieve_dir, f"{table_id}_retrieve.json")
            if not os.path.exists(retrieve_file_path):
                print(f"File not found: {retrieve_file_path}")
                continue

            with open(retrieve_file_path, 'r', encoding='utf-8') as retrieve_file:
                retrieve_data = json.load(retrieve_file)

            rows = []
            row_scores = []
            for j in range(1, 4):  # Assume maximum of 3 retrieve_content
                content_key = f"retrieve_content{j}"
                score_key = f"score{j}"
                if content_key in retrieve_data and score_key in retrieve_data:
                    rows.append(retrieve_data[content_key])
                    row_scores.append(retrieve_data[score_key])

            wiki_links = extract_wiki_links(item.get('answer-node', []))
            contexts, context_links = retrieve_context_from_wiki_links(wiki_links, request_tok_dir, table_id)

            if question and rows and contexts:
                question_embedding = encode_text(question, question_tokenizer, question_encoder)
                row_embeddings = [encode_text(row, question_tokenizer, question_encoder) for row in rows]
                context_embeddings = [encode_text(context, context_tokenizer, context_encoder) for context in contexts]

                alpha = 0.2
                combined_embeddings = [alpha * question_embedding + (1 - alpha) * row_embedding for row_embedding in row_embeddings]

                final_similarities = []
                for j, (row_score, combined_embedding) in enumerate(zip(row_scores, combined_embeddings)):
                    context_similarities = [torch.nn.functional.cosine_similarity(combined_embedding, context_emb).item() for context_emb in context_embeddings]
                    max_context_similarity = max(context_similarities)
                    max_context_idx = context_similarities.index(max_context_similarity)
                    final_similarity = row_score * max_context_similarity
                    final_similarities.append((final_similarity, j, max_context_idx))

                best_similarity, best_row_idx, best_context_idx = max(final_similarities)
                
                result = {
                    "question": question,
                    "row": rows[best_row_idx],
                    "row_similarity": row_scores[best_row_idx],
                    "context": contexts[best_context_idx],
                    "context_link": context_links[best_context_idx],  # Add the corresponding wiki link
                    "combined_similarity": best_similarity / row_scores[best_row_idx],
                    "final_similarity": best_similarity
                }
                
                write_result_to_json(output_dir, table_id, result)
            else:
                print(f"Skipping question {i+1} due to missing data")
        
        except Exception as e:
            print(f"Error processing question {i+1}: {str(e)}")
            continue

    print(f"Processed {num_questions} questions.")

if __name__ == "__main__":
    # Usage example
    input_json_path = "train.json"
    row_retrieve_dir = "row_retrieve"
    request_tok_dir = "WikiTables-WithLinks-master/request_tok"
    output_dir = "retrieve"

    # Get user input to determine the number of questions to process
    num_questions = input("Enter the number of questions to process (press Enter for all): ").strip()
    num_questions = int(num_questions) if num_questions else None

    # Call the function to process questions and table row data
    load_question_and_rows(input_json_path, row_retrieve_dir, request_tok_dir, output_dir, num_questions)