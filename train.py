import argparse
from Config import IDPTConfig, PTConfig
from datasets import load_dataset
from Text2TextDataset import Text2TextDataset
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler
import torch
from torch.optim import AdamW
from transformers import GPT2Tokenizer
from GPTIDPromptTuning import GPT2IDPromptTuningLM
from GPTPromptTuning import GPT2PromptTuningLM


# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


def load_imdb(tokenizer, batch_size=8, max_length=512, n_prompt_tokens=20, n_samples=50):
    dataset = load_dataset("imdb")
    batch_size = batch_size
    max_length = max_length
    n_prompt_tokens = n_prompt_tokens

    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token='<|pad|>')
    # tokenizer.pad_token = ?
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token='<|endoftext|>')

    train_df = pd.DataFrame({'text': dataset['train']['text'],
                             'label': dataset['train']['label']})
    train_df = train_df.sample(min(len(train_df), n_samples))
    test_df = pd.DataFrame({'text': dataset['test']['text'],
                            'label': dataset['test']['label']})
    test_df = test_df.sample(min(len(test_df), n_samples))

    train_dataset = Text2TextDataset(train_df, tokenizer, max_length=max_length - n_prompt_tokens)
    test_dataset = Text2TextDataset(test_df, tokenizer, max_length=max_length - n_prompt_tokens)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    return train_dataloader, test_dataloader


def train(id_pt=False):
    if id_pt:
        print("Performing ID-PT...\n")
        args = IDPTConfig()
    else:
        print("Performing PT...\n")
        args = PTConfig()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token='<|pad|>')
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token='<|endoftext|>')


    # get dataloaders
    train_dataloader, test_dataloader = load_imdb(tokenizer=tokenizer,
                                                  batch_size=args.batch_size,
                                                  max_length=args.max_length,
                                                  n_prompt_tokens=args.n_prompt_tokens,
                                                  n_samples=args.n_samples)

    if id_pt:
        model = GPT2IDPromptTuningLM.from_pretrained(args.model, n_tokens=args.n_prompt_tokens,
                                                     initialize_from_vocab=args.init_from_vocab)
    else:
        model = GPT2PromptTuningLM.from_pretrained(args.model, n_tokens=args.n_prompt_tokens,
                                                   initialize_from_vocab=args.init_from_vocab)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id  # add padding token

    # setting optimizer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n == "soft_prompt.weight"],
            "weight_decay": args.weight_decay,
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    num_training_steps = args.num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    model.train()

    for epoch in range(args.num_epochs):
        total_train_loss = 0

        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()
            progress_bar.update(1)
            batch_loss = loss.item()
            total_train_loss += batch_loss

        print(f'epoch: {epoch}, train loss: {total_train_loss / len(train_dataloader)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_pt', action='store_true')
    parser.add_argument('--no-id_pt', dest='id_pt', action='store_false')
    parser.set_defaults(feature=True)
    args = parser.parse_args()

    train(id_pt=args.id_pt)
