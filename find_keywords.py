import csv
import nltk

# Download the NLTK stopwords and punkt data
nltk.download('stopwords')
nltk.download('punkt')

# Define the CSV file path and the communication data column index
csv_file_path = "synthetic_dataset.csv"
comm_data_col_index = 4

# Open the CSV file and read the communication data column
with open(csv_file_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    comm_data_col = [row[comm_data_col_index] for row in reader]

# Join all communication data into a single string
comm_data_text = ' '.join(comm_data_col)

# Tokenize the communication data text into individual words
words = nltk.word_tokenize(comm_data_text)

# Filter out stop words (e.g. "the", "and", "to")
stop_words = set(nltk.corpus.stopwords.words('english'))
keywords = [word.lower() for word in words if word.lower() not in stop_words]

# Print the keywords
print(keywords)
