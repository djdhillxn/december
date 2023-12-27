import os
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import Word2Batch, HangmanGRUNet


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
    return avg_loss, success_rate


def train_model(model, train_data, val_data, epochs, learning_rate, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_func = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)
    num_words = len(train_data)
    
    # Ensure 'models/' directory exists
    model_save_dir = 'models/'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # start training
    tot_sample = 0
    for n in range(epochs):
        i = 0
        print(f"Starting Epoch {n+1}/{epochs}")
        model.train()
        epoch_loss = 0
        with tqdm(total=num_words, desc=f"Epoch {n+1}") as pbar:
            while tot_sample < (n + 1) * num_words:
                if i >= num_words:
                    break

                word = train_data[i]
                if len(word) == 1:
                    i += 1
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
                i += 1
                tot_sample += 1
                pbar.update(1)

            scheduler.step()
            # validation
            model.eval() 
            verbose_validation=False ##see here to change verbose type
            avg_val_loss, val_success_rate = evaluate_model(model, val_data, device, verbose=verbose_validation)
            model.train()
            print(f'Epoch {n + 1} - Training Loss: {epoch_loss / num_words:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Success Rate: {val_success_rate:.2f}')
            
            # Save model after each epoch with training data length and validation accuracy in the filename, in the models/ directory
            epoch_model_filename = f'{model_save_dir}hangman_model_trainlen{len(train_data)}_valacc{val_success_rate:.2f}_epoch{n + 1}.pth'
            torch.save(model.state_dict(), epoch_model_filename)
            print(f"Model saved as '{epoch_model_filename}'.")


def load_data(filepath):
    with open(filepath, 'r') as file:
        words = [line.strip() for line in file]
    return words


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading training data...")
    train_words = load_data('./data/words_250000_train.txt')
    print(f"Loaded {len(train_words)} words for training.")

    print("Loading testing data...")
    test_words = load_data('./data/words_test.txt')
    print(f"Loaded {len(test_words)} words for testing.")

    print("Initializing model...")
    model = HangmanGRUNet(hidden_dim=512, gru_layers=2)
    print("Model initialized.")

    """
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    """

    model.to(device)
    print(model)

    print("Starting training...")
    train_model(model, train_words[:100], test_words[:100], epochs=5, learning_rate=0.001, device=device)
    print("Training completed.")

    
if __name__ == '__main__':
    main()
