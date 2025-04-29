import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import accuracy_score
from seqeval.metrics import classification_report
from collections import defaultdict
import numpy as np
from datasets import load_dataset
from torch import cuda

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []
    sentence = sentence.strip()
    for word, label in zip(sentence.split(), text_labels.split(",")):
        tokenized_word = tokenizer.tokenize(word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * len(tokenized_word))
    return tokenized_sentence, labels

class NERDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        sentence = self.data.sentence.iloc[index]
        word_labels = self.data.word_labels.iloc[index]
        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        labels = ["O"] + labels + ["O"]

        if len(tokenized_sentence) > self.max_len:
            tokenized_sentence = tokenized_sentence[:self.max_len]
            labels = labels[:self.max_len]
        else:
            pad_len = self.max_len - len(tokenized_sentence)
            tokenized_sentence += ['[PAD]'] * pad_len
            labels += ["O"] * pad_len

        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        label_ids = [label2id[label] for label in labels]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len

def train_model(model, optimizer, scheduler, training_loader, max_grad_norm):
    model.train()
    total_loss = 0
    total_steps = 0
    for idx, batch in enumerate(training_loader):
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)

        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        loss = outputs.loss
        total_loss += loss.item()
        total_steps += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    if scheduler:
        scheduler.step()
    avg_loss = total_loss / total_steps
    print(f"Average training loss: {avg_loss}")
    return avg_loss

def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    total_steps = 0
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in data_loader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
            for i in range(ids.size(0)):
                true_seq = []
                pred_seq = []
                for j in range(ids.size(1)):
                    if mask[i][j] == 1:
                        true_seq.append(id2label[targets[i][j].item()])
                        pred_seq.append(id2label[predictions[i][j].item()])
                all_true.append(true_seq)
                all_preds.append(pred_seq)
    avg_loss = total_loss / total_steps
    print(f"Validation Loss: {avg_loss}")
    report = classification_report(all_true, all_preds, output_dict=True)
    print("Classification Report:", report)
    return avg_loss, report



def main(args):
    df_selected = pd.read_json(args.comparisons_file)
    selected_texts = set(df_selected["text"])

    full_dataset = load_dataset("conll2003")["train"].to_pandas()
    df_remaining = full_dataset[~full_dataset["sentence"].isin(selected_texts)]
    print("Number of previously selected sentences:", len(selected_texts))
    print("Number of remaining sentences:", df_remaining.shape[0])

    if args.group == 'A':
        training_df = df_remaining.sample(n=args.num_samples, random_state=42)
    elif args.group == 'B':
        training_df = pd.read_json(args.distilled_file, lines=True).sample(n=args.num_samples, random_state=42)
    else:
        raise ValueError("Group must be either 'A' or 'B'.")

    train_df = training_df.sample(frac=0.8, random_state=42)
    val_df = training_df.drop(train_df.index)

    global label2id, id2label
    all_labels = set()
    for labels in train_df['word_labels']:
        for lab in labels.split(","):
            all_labels.add(lab)
    label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
    id2label = {idx: label for label, idx in label2id.items()}

    train_dataset = NERDataset(train_df, tokenizer, args.max_len)
    val_dataset = NERDataset(val_df, tokenizer, args.max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=0)

    for run in range(args.num_runs):
        print(f"----- Run {run + 1} -----")
        model = BertForTokenClassification.from_pretrained('bert-base-uncased',
                                                             num_labels=len(id2label),
                                                             id2label=id2label,
                                                             label2id=label2id)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=args.lr_decay) if args.lr_decay < 1.0 else None

        for epoch in range(args.epochs):
            print(f"Training epoch: {epoch + 1}")
            train_model(model, optimizer, scheduler, train_loader, args.max_grad_norm)

        _, report = evaluate_model(model, val_loader)
        print("Run", run+1, "Evaluation Report:", report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Training Script for Group A/B")
    parser.add_argument("--group", type=str, choices=["A", "B"], default="A",
                        help="Group to train: 'A' for original, 'B' for distilled")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to use for training")
    parser.add_argument("--comparisons_file", type=str, default="energy_ner_llm/energy_ner_llm/src/mod_distil_by_cot/comparisons.json",
                        help="Path to the comparisons JSON file")
    parser.add_argument("--distilled_file", type=str, default="path/to/distilled.json",
                        help="Path to the distilled data JSON file (used for Group B)")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Validation batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-05, help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=10, help="Max gradient norm for clipping")
    parser.add_argument("--lr_decay", type=float, default=0.95, help="Learning rate decay gamma (set <1.0 to use decay)")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of independent training runs")
    args = parser.parse_args()
    main(args)
