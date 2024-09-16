import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import (
    DPRQuestionEncoder, DPRContextEncoder, 
    DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer,
    AdamW, get_linear_schedule_with_warmup
)

# Set paths
BASE_PATH = "HybridQA"
PREPARED_DATA_FILE = os.path.join(BASE_PATH, "prepared_dpr_training_data.json")
TEST_DATA_FILE = os.path.join(BASE_PATH, "prepared_dpr_test_data.json")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DPRDataset(Dataset):
    def __init__(self, data, question_tokenizer, context_tokenizer, max_length=512):
        self.data = data
        self.question_tokenizer = question_tokenizer
        self.context_tokenizer = context_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        positive_row = ' '.join(str(v) for v in item['positive_row'].values())
        negative_row = ' '.join(str(v) for v in item['negative_row'].values())

        question_encoding = self.question_tokenizer(question, truncation=True, padding='max_length', 
                                                    max_length=self.max_length, return_tensors='pt')
        positive_encoding = self.context_tokenizer(positive_row, truncation=True, padding='max_length', 
                                                   max_length=self.max_length, return_tensors='pt')
        negative_encoding = self.context_tokenizer(negative_row, truncation=True, padding='max_length', 
                                                   max_length=self.max_length, return_tensors='pt')

        return {
            'question': {k: v.squeeze(0) for k, v in question_encoding.items()},
            'positive': {k: v.squeeze(0) for k, v in positive_encoding.items()},
            'negative': {k: v.squeeze(0) for k, v in negative_encoding.items()}
        }

class DPRTestDataset(Dataset):
    def __init__(self, data, question_tokenizer, context_tokenizer, max_length=512):
        self.data = data
        self.question_tokenizer = question_tokenizer
        self.context_tokenizer = context_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        rows = item['rows']
        
        question_encoding = self.question_tokenizer(question, truncation=True, padding='max_length', 
                                                    max_length=self.max_length, return_tensors='pt')
        
        row_encodings = [self.context_tokenizer(' '.join(str(v) for v in row.values()), 
                                                truncation=True, padding='max_length',
                                                max_length=self.max_length, return_tensors='pt')
                         for row in rows]
        
        return {
            'question': {k: v.squeeze(0) for k, v in question_encoding.items()},
            'rows': [{k: v.squeeze(0) for k, v in encoding.items()} for encoding in row_encodings],
            'table_id': item['table_id'],
            'correct_index': item['correct_row_index']
        }

def encode_data(question_encoder, context_encoder, dataset, device):
    question_encoder.eval()
    context_encoder.eval()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    question_embeddings = []
    positive_embeddings = []
    negative_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            q_emb = question_encoder(
                input_ids=batch['question']['input_ids'].to(device),
                attention_mask=batch['question']['attention_mask'].to(device)
            ).pooler_output
            p_emb = context_encoder(
                input_ids=batch['positive']['input_ids'].to(device),
                attention_mask=batch['positive']['attention_mask'].to(device)
            ).pooler_output
            n_emb = context_encoder(
                input_ids=batch['negative']['input_ids'].to(device),
                attention_mask=batch['negative']['attention_mask'].to(device)
            ).pooler_output
            
            question_embeddings.extend(q_emb.cpu().numpy())
            positive_embeddings.extend(p_emb.cpu().numpy())
            negative_embeddings.extend(n_emb.cpu().numpy())
    
    return np.array(question_embeddings), np.array(positive_embeddings), np.array(negative_embeddings)

def train_epoch(question_encoder, context_encoder, train_dataloader, optimizer, scheduler, device, epoch):
    question_encoder.train()
    context_encoder.train()
    total_loss = 0
    
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()
        
        q_emb = question_encoder(
            input_ids=batch['question']['input_ids'].to(device),
            attention_mask=batch['question']['attention_mask'].to(device)
        ).pooler_output
        p_emb = context_encoder(
            input_ids=batch['positive']['input_ids'].to(device),
            attention_mask=batch['positive']['attention_mask'].to(device)
        ).pooler_output
        n_emb = context_encoder(
            input_ids=batch['negative']['input_ids'].to(device),
            attention_mask=batch['negative']['attention_mask'].to(device)
        ).pooler_output
        
        # Calculate similarity
        pos_score = torch.sum(q_emb * p_emb, dim=1)
        neg_score = torch.sum(q_emb * n_emb, dim=1)
        
        # Calculate loss
        loss = nn.functional.margin_ranking_loss(
            pos_score, neg_score, 
            torch.ones_like(pos_score).to(device), 
            margin=0.1
        )
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_dataloader)

def test_model(question_encoder, context_encoder, test_dataset, device):
    question_encoder.eval()
    context_encoder.eval()
    results = []
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            batch = test_dataset[idx]
            q_emb = question_encoder(
                input_ids=batch['question']['input_ids'].unsqueeze(0).to(device),
                attention_mask=batch['question']['attention_mask'].unsqueeze(0).to(device)
            ).pooler_output
            
            row_embeddings = []
            for row in batch['rows']:
                r_emb = context_encoder(
                    input_ids=row['input_ids'].unsqueeze(0).to(device),
                    attention_mask=row['attention_mask'].unsqueeze(0).to(device)
                ).pooler_output
                row_embeddings.append(r_emb)
            
            row_embeddings = torch.cat(row_embeddings, dim=0)
            
            similarities = torch.matmul(q_emb, row_embeddings.t()).squeeze()
            predicted_index = similarities.argmax().item()
            
            results.append({
                'table_id': batch['table_id'],
                'label': predicted_index,
                'row': test_dataset.data[idx]['rows'][predicted_index],
                'answer_node': batch['correct_index']
            })
    
    return results

def main():
    # Load pre-trained DPR models and tokenizers
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

    # Load training data
    print(f"Loading prepared training data from {PREPARED_DATA_FILE}")
    with open(PREPARED_DATA_FILE, "r") as f:
        training_data = json.load(f)
    print(f"Loaded {len(training_data)} prepared training samples.")

    # Create training dataset and dataloader
    train_dataset = DPRDataset(training_data, question_tokenizer, context_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)
    
    # Set up optimizer and learning rate scheduler
    optimizer = AdamW(list(question_encoder.parameters()) + list(context_encoder.parameters()), lr=2e-5)
    total_steps = len(train_dataloader) * 3  # Assuming 3 epochs of training
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train the model
    num_epochs = 3
    for epoch in range(num_epochs):
        avg_loss = train_epoch(question_encoder, context_encoder, train_dataloader, optimizer, scheduler, device, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # Encode training data
    print("Encoding training data...")
    question_embeddings, positive_embeddings, negative_embeddings = encode_data(question_encoder, context_encoder, train_dataset, device)

    # Save encoded vectors
    np.save(os.path.join(BASE_PATH, "question_embeddings.npy"), question_embeddings)
    np.save(os.path.join(BASE_PATH, "positive_embeddings.npy"), positive_embeddings)
    np.save(os.path.join(BASE_PATH, "negative_embeddings.npy"), negative_embeddings)

    print("Encoded data saved.")

    # Load test data
    print(f"Loading test data from {TEST_DATA_FILE}")
    with open(TEST_DATA_FILE, "r") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples.")

    # Create test dataset
    test_dataset = DPRTestDataset(test_data, question_tokenizer, context_tokenizer)

    # Test the model
    test_results = test_model(question_encoder, context_encoder, test_dataset, device)

    # Output test results
    print("\nTest Results:")
    for result in test_results:
        print(f"Table ID: {result['table_id']}")
        print(f"Predicted Row: {result['label']}")
        print(f"Correct Row: {result['answer_node']}")
        print(f"Retrieved Row Content: {result['row']}")
        print("---")

    # Save test results
    with open("retrieval_result.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print("Test results saved to retrieval_result.json")

if __name__ == "__main__":
    main()