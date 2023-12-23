import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from sup.model import LinearProbe
from sup.dataset import LabelDataset
from sup.hack import PhiForCausalLM
from llm import LLM
from Define import PROMPT_POOL


def main(args):
    # load model
    phi_model = PhiForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, device_map="cuda", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token_id=tokenizer.eos_token_id
    llm = LLM(model=phi_model,tokenizer=tokenizer)

    # model and optimizer
    model = LinearProbe(llm=llm, n_cls=len(PROMPT_POOL))
    optimizer = torch.optim.Adam(model.build_optimized_model().parameters(), lr=args.lr)
    model.cuda()
    loss_func = nn.CrossEntropyLoss()

    # load data
    train_paths = [
        f"{args.root}/hotpotqa_train_processed_data.json",
        f"{args.root}/openbookqa_train_processed_data.json",
        f"{args.root}/strategyqa_train_processed_data.json",
        f"{args.root}/truthfulqa_train_processed_data.json",
    ]
    test_paths = [
        f"{args.root}/hotpotqa_validation_processed_data.json",
        f"{args.root}/openbookqa_test_processed_data.json",
        f"{args.root}/strategyqa_test_processed_data.json",
        f"{args.root}/truthfulqa_test_processed_data.json",
    ]
    train_dataset = LabelDataset(train_paths, llm.tokenizer)
    test_dataset = LabelDataset(test_paths, llm.tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn())

    # train loop
    print("Start training!")
    for t in range(args.n_epoch):
        losses = []
        # acc
        accs = []
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            predictions = model(batch["samples"])  # B, n_cls
            loss = loss_func(predictions, batch["labels"].cuda())
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # train acc
            acc = (predictions.argmax(dim=-1) == batch["labels"].cuda()).sum() / len(batch["labels"])
            accs.append(acc.item())

            if i == 0:
                print(predictions.argmax(dim=-1))
                print(batch["labels"])

            # log
            if args.log_step > 0 and (i + 1) % args.log_step == 0:
                print(f'[Epoch {t+1}] Train/Loss: {sum(losses) / len(losses):.4f}')
                losses = []
        
        if args.log_step < 0:
            # print(f'[Epoch {t+1}] Train/Loss: {sum(losses) / len(losses):.4f}')
            final_loss = sum(losses) / len(losses)
            final_acc = sum(accs) / len(accs)
            print(f'[Epoch {t+1}] Train/Acc: {final_acc*100:.2f}%, Train/Loss: {final_loss:.4f}')
        losses = []
        
        # validation loop
        model.eval()
        if (t + 1) % args.val_epoch == 0:
            with torch.no_grad():
                accs, losses = [], []
                for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
                    predictions = model(batch["samples"])  # B, n_cls
                    loss = loss_func(predictions, batch["labels"].cuda())
                    losses.append(loss.item())

                    acc = (predictions.argmax(dim=-1) == batch["labels"].cuda()).sum() / len(batch["labels"])
                    accs.append(acc.item())

                    if i == 0:
                        print(predictions.argmax(dim=-1))
                        print(batch["labels"])
                    
            # log
            final_loss = sum(losses) / len(losses)
            final_acc = sum(accs) / len(accs)
            print(f'[Epoch {t+1}] Val/Acc: {final_acc*100:.2f}%, Val/Loss: {final_loss:.4f}')
            os.makedirs(f"{args.exp_name}/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"{args.exp_name}/checkpoints/{t+1}.ckpt")
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="directory with data json paths", default="./_data")
    parser.add_argument("--exp_name", type=str, help="experiment name", default="model2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--n_epoch", type=int, default=100, help="total run epoch")
    parser.add_argument("--val_epoch", type=int, default=1)
    parser.add_argument("--log_step", type=int, default=-1, help="log every n step")
    
    args = parser.parse_args()
    main(args)
