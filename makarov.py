import random

# Load the communication data from a file
with open('communication_data.txt', 'r') as f:
    communication_data = f.read().splitlines()

# Preprocess the data by splitting it into individual words
words = []
for message in communication_data:
    words.extend(message.split())

# Define the Markov chain model by calculating the transition probabilities
def get_transition_probabilities(words):
    transitions = {}
    for i in range(len(words) - 1):
        curr_word = words[i]
        next_word = words[i + 1]
        if curr_word not in transitions:
            transitions[curr_word] = {}
        if next_word not in transitions[curr_word]:
            transitions[curr_word][next_word] = 0
        transitions[curr_word][next_word] += 1
    probabilities = {}
    for curr_word, next_words in transitions.items():
        total_count = sum(next_words.values())
        probabilities[curr_word] = {next_word: count / total_count for next_word, count in next_words.items()}
    return probabilities

# Generate a new message using the Markov chain model
def generate_message(probabilities, max_length=20):
    message = []
    curr_word = random.choice(list(probabilities.keys()))
    while len(message) < max_length and curr_word in probabilities:
        message.append(curr_word)
        next_words = probabilities[curr_word]
        curr_word = random.choices(list(next_words.keys()), list(next_words.values()))[0]
    return ' '.join(message)

# Train the Markov chain model on the preprocessed data
probabilities = get_transition_probabilities(words)

# Generate 10 new messages using the Markov chain model
new_messages = []
for i in range(1000):
    message = generate_message(probabilities)
    new_messages.append(message)

# Append the new messages to the communication data file
with open('communication_data.txt', 'a') as f:
    for message in new_messages:
        f.write(message + '\n')    
