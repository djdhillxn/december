import os
import time
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from model import Word2Batch, RNN_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

def show_game(original_word, guesses, obscured_words_seen):
    print('Hidden word was "{}"'.format(original_word))
    for i in range(len(guesses)):
        word_seen = ''.join([chr(i + 97) if i != 26 else ' ' for i in obscured_words_seen[i].argmax(axis=1)])
        print('Guessed {} after seeing "{}"'.format(guesses[i], word_seen))


def get_all_words(file_location):
    with open(file_location, "r") as text_file:
        all_words = text_file.read().splitlines()
    return all_words


# get data
root_path = os.getcwd()
file_name = "./data/words_250000_train.txt"
file_path = os.path.join(root_path, file_name)
words = get_all_words(file_path)
words = words[:1000]
num_words = len(words)

# define model
model = RNN_model(target_dim=26, hidden_units=16).to(device)

# define hyper parameter
n_epoch = 2
lr = 0.001
record_step = 100  # output result every 100 words
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
loss_func = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)

# start training
start_time = time.perf_counter()
tot_sample = 0
for n in range(n_epoch):
    i = 0
    print(f"Starting Epoch {n+1}/{n_epoch}")

    with tqdm(total=num_words, desc=f"Epoch {n+1}") as pbar:
        while tot_sample < (n + 1) * num_words:
            if i >= len(words):
                break

            word = words[i]
            if len(word) == 1:
                i += 1
                continue

            new_batch = Word2Batch(word=word, model=model, device=device)
            obscured_word, prev_guess, correct_response = new_batch.game_mimic(model)

            
            optimizer.zero_grad()
            predict = model(obscured_word, prev_guess)
            predict = predict.squeeze(1)
            loss = loss_func(predict, correct_response)
            loss.backward()
            optimizer.step()
            

            i += 1
            tot_sample += 1
            pbar.update(1)

        scheduler.step()
