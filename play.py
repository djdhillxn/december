import tqdm
import time
import pickle
import os
from hangman_api import HangmanAPI

API_STATE_FILE = 'api_state.pkl'

def load_api_state():
    if os.path.exists(API_STATE_FILE):
        with open(API_STATE_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def run_practice_games(api, num_games):
    wins = 0
    with tqdm.tqdm(total=num_games, position=0, leave=True) as pbar:
        for _ in range(num_games):
            # Get wins count before starting the game
            [_, _, _, prev_total_practice_successes] = api.my_status()

            # Start a new practice game
            api.start_game(practice=1, verbose=True)
            #verbose True or False set accordingly 

            # Get wins count after the game
            [_, _, _, post_total_practice_successes] = api.my_status()

            # Determine if the current game was won
            if post_total_practice_successes > prev_total_practice_successes:
                wins += 1

            pbar.update(1)
            pbar.set_postfix({'accuracy': wins / (pbar.n if pbar.n > 0 else 1)})
            time.sleep(0.5)  # To prevent too high frequency of requests

    return wins


def main():    
    num_games = 50  # Number of games to run; adjust as needed
    
    # Try to load the saved API state
    api = load_api_state()
    if api is None:
        print("No saved Hangman API state found. Please run initialize.py first.")
        return

    print("Using saved Hangman API state.")

    #Run the specified number of practice games and get the number of wins
    num_wins = run_practice_games(api, num_games)
    # Print the session-specific success rate
    if num_games > 0:
        success_rate = num_wins / num_games
        print(f'\nSuccess rate for this session: {success_rate:.3f} ({num_wins}/{num_games} games won)')
    else:
        print("No games were played in this session.")

    # Get and print game stats
    stats = api.my_status()
    total_practice_runs, total_recorded_runs, total_recorded_successes, total_practice_successes = stats
    practice_success_rate = total_practice_successes / total_practice_runs
    print(f'Completed {total_practice_runs} practice games out of an allotted 100,000.')
    print(f'Absolute Practice success rate so far: {practice_success_rate:.3f}')

if __name__ == "__main__":
    main()

