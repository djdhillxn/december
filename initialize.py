# initialize.py
import time
import pickle
from datetime import datetime, timedelta
from hangman_api import HangmanAPI

API_STATE_FILE = 'api_state.pkl'
LOG_FILE = 'initialization.log'

def initialize_api(access_token, timeout=2000):
    start_time = time.time()
    api = HangmanAPI(access_token=access_token, timeout=timeout)
    duration = time.time() - start_time

    # Calculate and format expiration time
    expiration_time = datetime.now() + timedelta(seconds=timeout)
    formatted_expiration = expiration_time.strftime("%Y-%m-%d %H:%M:%S")

    # Log initialization and expiration time
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"API Initialized: {time.ctime(start_time)}\n")
        log_file.write(f"Initialization Duration: {duration:.2f} seconds\n")
        log_file.write(f"API Expires: {formatted_expiration}\n")

    # Print the same information to the console
    print(f"API Initialized: {time.ctime(start_time)}")
    print(f"Initialization Duration: {duration:.2f} seconds")
    print(f"API Expires: {formatted_expiration}")

    # Save the API state
    with open(API_STATE_FILE, 'wb') as f:
        pickle.dump(api, f)

if __name__ == "__main__":
    access_token = "a89c46a38ba61895d3a1df501b490e"
    initialize_api(access_token)
