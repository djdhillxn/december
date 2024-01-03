import os
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import Word2Batch, HangmanGRUNet
import random

def evaluate_model(model, words, device, verbose=False):
    model.eval()
    success_count = 0
    total_loss = 0
    loss_func = nn.BCEWithLogitsLoss()

    for word in tqdm(words, desc='Evaluating', unit='word'):
        batch = Word2Batch(model=model, word=word, device=device)
        obscured_word, prev_guess, correct_response = batch.game_mimic(verbose=verbose)
        predict = model(obscured_word, prev_guess)
        predict = predict.squeeze(1)
        loss = loss_func(predict, correct_response)
        total_loss += loss.item()

        # Evaluate the success of the game
        if batch.lives_left > 0 and len(batch.remain_letters) == 0:
            success_count += 1
    
    model.train()
    avg_loss = total_loss / len(words)
    success_rate = success_count / len(words)
    print(f"Evaluation completed. Average Loss: {avg_loss:.4f}, Success Rate: {success_rate:.2f}")
    return avg_loss, success_rate


def train_model(model, train_data, val_data, epochs, learning_rate, device, checkpoint_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_func = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)
    num_words = len(train_data)

    start_epoch = 0
    best_val_accuracy = 0.0

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_accuracy = checkpoint['best_val_accuracy']
        print(f"Resuming from epoch {start_epoch} with best val accuracy {best_val_accuracy}")
    
    # Ensure 'models/' directory exists
    model_save_dir = 'models/'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    

    for n in range(start_epoch, epochs):
        print(f"Epoch {n+1} of {epochs} starting.")
        model.train()
        epoch_loss = 0
        random.shuffle(train_data)
        with tqdm(total=num_words, desc=f"Epoch {n+1}/{epochs}", unit='word') as pbar:
            for i, word in enumerate(train_data):
                if len(word) == 1:
                    continue
                new_batch = Word2Batch(word=word, model=model, device=device)
                obscured_word, prev_guess, correct_response = new_batch.game_mimic()
                optimizer.zero_grad()
                predict = model(obscured_word, prev_guess)
                predict = predict.squeeze(1)
                loss = loss_func(predict, correct_response)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.update(1)
            scheduler.step()
            model.eval() #validation 
            avg_val_loss, val_success_rate = evaluate_model(model, val_data, device, verbose=False)
            model.train()

            print(f"Epoch {n+1} of {epochs} completed. Training Loss: {epoch_loss / num_words:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Success Rate: {val_success_rate:.5f}")
            
            # Check and update the best model
            if val_success_rate > best_val_accuracy:
                best_val_accuracy = val_success_rate
                best_model_filename = f'{model_save_dir}best_hangman_model_epoch{n + 1}_trainlen{len(train_data)}_valacc{val_success_rate:.5f}_bestvalacc{best_val_accuracy:.5f}.pth'        
                torch.save({'epoch': n + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_accuracy': best_val_accuracy},
                        best_model_filename)
                print(f"New best model saved. Accuracy: {val_success_rate:.5f}, Epoch: {n + 1}")


            # Save the model for the current epoch
            epoch_model_filename = f'{model_save_dir}hangman_model_epoch{n + 1}_trainlen{len(train_data)}_valacc{val_success_rate:.5f}_bestvalacc{best_val_accuracy:.5f}.pth'
            torch.save({'epoch': n + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'current_val_accuracy': val_success_rate,
                        'best_val_accuracy': best_val_accuracy},
                    epoch_model_filename)
            print(f"Epoch {n + 1} model saved as '{epoch_model_filename}'.")

def load_data(filepath):
    print(f"Loading data from {filepath}")
    with open(filepath, 'r') as file:
        words = [line.strip() for line in file]
    print(f"Data loaded. Number of words: {len(words)}")
    return words

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_words = load_data('./data/words_250000_train.txt')
    test_words = load_data('./data/words_test.txt')

    print("Initializing model...")
    model = HangmanGRUNet(hidden_dim=512, gru_layers=2)
    model.to(device)
    print("Model initialized.")

    print("Model summary:")
    print(model)

    # Define the checkpoint path for resuming training
    #checkpoint_path = './models/hangman_model_epoch5_trainlen120_valacc0.30_bestvalacc0.36.pth'
    checkpoint_path = './models/hangman_model_epoch10_trainlen120_valacc0.25000_bestvalacc0.36667.pth'
    #checkpoint_path = None
    current_epoch = 0  # Default current epoch if no checkpoint is found

    # Check if checkpoint file exists and update the current epoch
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        current_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {current_epoch}.")
    else:
        print("No checkpoint found or specified. Starting training from scratch.")

    # Define the additional number of epochs to train
    additional_epochs = 10
    total_epochs = current_epoch + additional_epochs

    print("Starting training...")
    train_model(model, train_words[:120], test_words[:120], epochs=total_epochs, learning_rate=0.001, device=device, checkpoint_path=checkpoint_path)
    print("Training completed.")

if __name__ == '__main__':
    main()
