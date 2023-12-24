import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributed as dist
import collections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def show_game(original_word, guesses, obscured_words_seen):
    print('Hidden word was "{}"'.format(original_word))

    for i in range(len(guesses)):
        word_seen = ''.join([chr(i + 97) if i != 26 else ' ' for i in obscured_words_seen[i].argmax(axis=1)])
        print('Guessed {} after seeing "{}"'.format(guesses[i], word_seen))

class HangmanGRUNet(nn.Module):
    def __init__(self, hidden_dim, target_dim=26, gru_layers=1, device='cpu'):
        super(HangmanGRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(27, hidden_dim, num_layers=gru_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_dim + 26, target_dim).to(device)
        self.device = device
    
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
            final_gru_out = gru_out[:, -1, :]

            # Ensure prev_guess is a 2D tensor for concatenation
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

    def game_mimic(self, model):
        model = model.to(device)
        obscured_words_seen = []
        prev_guess_seen = []
        correct_response_seen = []

        while self.lives_left > 0 and len(self.remain_letters) > 0:
            # store obscured word and previous guesses -- act as X for the label
            obscured_word = self.encode_obscure_word()
            #print("Shape of obscured_word:", obscured_word.shape)  # Add this line to check the shape

            prev_guess = self.encode_prev_guess()

            obscured_words_seen.append(obscured_word)
            prev_guess_seen.append(prev_guess)

            self.model.eval()
            guess = self.model(obscured_word, prev_guess)  # output of guess should be a 1 by 26 vector
            guess = torch.argmax(guess, dim=2).item()
            self.guessed_letter.add(guess)
            self.guessed_letter_each.append(chr(guess + 97))

            # store correct response -- act as label for the model
            correct_response = self.encode_correct_response()
            correct_response_seen.append(correct_response)

            # update letter remained and lives left
            if guess in self.remain_letters:  # only remove guess when the guess is correct
                self.remain_letters.remove(guess)

            if correct_response_seen[-1][guess] < 0.0000001:  # which means we made a wrong guess
                self.lives_left -= 1

        return torch.stack(obscured_words_seen), torch.stack(prev_guess_seen), torch.stack(correct_response_seen)

def gen_n_gram(word, n):
    n_gram = []
    for i in range(n, len(word)+1):
        if word[i-n:i] not in n_gram:
            n_gram.append(word[i-n:i])
    return n_gram

def init_n_gram(n):
    n_gram = {}
    full_dictionary = ["apple", "hhh", "genereate", "google", "abc", "googla"]
    for word in full_dictionary:
        single_word_gram = gen_n_gram(word, n)
        print(word, single_word_gram)
        if len(word) not in n_gram:
            n_gram[len(word)] = single_word_gram
        else:
            n_gram[len(word)].extend(single_word_gram)
    print(n_gram)
    res = {}
    for key in n_gram.keys():
        res[key] = collections.Counter(n_gram[key])
    return res

if __name__ == "__main__":
    word = "compluvia"
    print(init_n_gram(2))
    # print(gen_n_gram(word, 2))
    # target_size = 26
    # # model = HangManModel(target_dim=target_size)
    # model = RNN_model(target_dim=target_size, hidden_units=16)
    # new_batch = Word2Batch(model, word)
    # a, b, c = new_batch.game_mimic()
    # guess = new_batch.guessed_letter_each
    # new_model = RNN_model()
    # out = new_model(a, b)
    # out = out.squeeze(1)
    # loss_func = nn.BCEWithLogitsLoss()
    # loss = loss_func(out, c)
    # show_game(word, guess, a)






