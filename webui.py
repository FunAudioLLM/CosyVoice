import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import YourModel, YourTokenizer
import argparse
import os

def train(model, dataloader, optimizer, device, accumulation_steps=2):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Normalize loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()
        
        # Accumulate gradients
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--model_path', type=str, default='path_to_model', help='Path to your model')
    parser.add_argument('--use_fp16', action='store_true', help='Use mixed precision training')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    model = YourModel.from_pretrained(args.model_path).to(device)
    tokenizer = YourTokenizer.from_pretrained(args.model_path)
    
    # Data loading and preparation
    train_dataset = YourDataset() # Customize this
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer and learning rate scheduling
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None
    
    # Training loop
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        
        model.train()
        total_loss = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=args.use_fp16):
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # Scale loss for mixed precision training
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Average Loss: {avg_loss:.4f}')

if __name__ == '__main__':
    main()
