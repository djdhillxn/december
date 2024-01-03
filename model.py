import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributed as dist
import collections


class HangmanGRUNet(nn.Module):
    def __init__(self, hidden_dim, target_dim=26, gru_layers=1):
        super(HangmanGRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(27, hidden_dim, dropout=0.19, num_layers=gru_layers, bidirectional=True, batch_first=True) 
        self.fc = nn.Linear(hidden_dim * 2 + 26, target_dim)
    
    def forward(self, obscure_word, prev_guess, train=True):
        if len(obscure_word.size()) < 3:
            obscure_word = obscure_word.unsqueeze(0)
        if len(prev_guess.size()) < 2:
            prev_guess = prev_guess.unsqueeze(0)
        no_of_timesteps = obscure_word.shape[0]
        batch_size = obscure_word.shape[1]
        outputs = []
        for i in range(no_of_timesteps):
            gru_out, _ = self.gru(obscure_word[i].unsqueeze(0))
            final_gru_out = torch.cat((gru_out[:, -1, :self.hidden_dim], gru_out[:, 0, self.hidden_dim:]), dim=1)
            curr_prev_guess = prev_guess[i]
            curr_prev_guess = curr_prev_guess.unsqueeze(0) if curr_prev_guess.dim() == 1 else curr_prev_guess
            combined = torch.cat((final_gru_out, curr_prev_guess), dim=1)
            out = self.fc(combined)
            outputs.append(out)
        return torch.stack(outputs)


class Word2Batch:
    def __init__(self, model, word, lives=6, device='cpu'):
        self.device = device
        self.origin_word = word
        self.guessed_letter = set()  # each element should be a idx
        self.word_idx = [ord(i)-97 for i in word]
        self.remain_letters = set(self.word_idx)
        self.model = model
        self.lives_left = lives
        self.guessed_letter_each = []

        # the following is the dataset for variable to output
        self.obscured_word_seen = []  # n * 27, where n is the number of guesses
        self.prev_guessed = []  # n*26, where n is the number of guesses and each element is the normalized word idx
        self.correct_response = []  # this is the label, meaning self.prev_guess should be one of self.correct_response
    
    def encode_obscure_word(self):
        word = [i if i in self.guessed_letter else 26 for i in self.word_idx]
        obscured_word = np.zeros((len(word), 27), dtype=np.float32)
        for i, j in enumerate(word):
            obscured_word[i, j] = 1
        return torch.from_numpy(obscured_word).to(self.device)

    def encode_prev_guess(self):
        guess = np.zeros(26, dtype=np.float32)
        for i in self.guessed_letter:
            guess[i] = 1.0
        return torch.from_numpy(guess).to(self.device)

    def encode_correct_response(self):
        response = np.zeros(26, dtype=np.float32)
        for i in self.remain_letters:
            response[i] = 1.0
        response /= response.sum()
        return torch.from_numpy(response).to(self.device)

    def game_mimic(self, verbose=False):
        obscured_words_seen = []
        prev_guess_seen = []
        correct_response_seen = []

        # Print the actual word at the start of each game if verbose is True
        if verbose:
            print("-" * 30)
            print(f"Actual word: {self.origin_word}")
            print("-" * 30)

        while self.lives_left > 0 and len(self.remain_letters) > 0:
            obscured_word = self.encode_obscure_word()
            prev_guess = self.encode_prev_guess()
  
            self.model.eval()
            with torch.no_grad():
                guess_probabilities = self.model(obscured_word, prev_guess)
                guess_probabilities[0, :, list(self.guessed_letter)] = -float('inf')  # Set probabilities of guessed letters to negative infinity
                guess = torch.argmax(guess_probabilities, dim=2).item()            
            self.guessed_letter.add(guess)
            self.guessed_letter_each.append(chr(guess + 97))
            self.model.train()

            obscured_words_seen.append(obscured_word)
            prev_guess_seen.append(prev_guess) 

            
            current_state = self.decode_obscured_word(obscured_word)
            guessed_letter = chr(guess + 97) #if guess in self.remain_letters else '~'
            if verbose:
                game_status = "Game Won" if len(self.remain_letters) == 0 else ""
                print(f"Remaining lives: {self.lives_left}, Guessed letter: '{guessed_letter}', Word state: {current_state}, {game_status}".rstrip(", "))
            

            correct_response = self.encode_correct_response()
            correct_response_seen.append(correct_response)


            if guess in self.remain_letters:
                self.remain_letters.remove(guess)

            if correct_response_seen[-1][guess] < 0.0000001:
                self.lives_left -= 1

            
            if verbose and (len(self.remain_letters) == 0 or self.lives_left == 0):
                game_status = "Game Won" if self.lives_left > 0 else "Game Lost"
                final_word_state = self.origin_word if self.lives_left > 0 else current_state
                print(f"Remaining lives: {self.lives_left}, Guessed letter: '~', Word state: {final_word_state}, {game_status}")
                print("-" * 30)
                break


        return torch.stack(obscured_words_seen), torch.stack(prev_guess_seen), torch.stack(correct_response_seen)

    def decode_obscured_word(self, obscured_word):
        return ''.join([chr(i + 97) if i != 26 else '_' for i in obscured_word.argmax(axis=1)])
