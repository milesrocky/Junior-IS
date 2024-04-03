import os
import music21
from transformers import BertTokenizer
from transformers import AutoTokenizer

# Specify the directory 
data_dir = r"C:\Users\miles\OneDrive\Desktop\Junior IS\Training data"

# Create lists to store data
data = []
target_data = []


# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data
input_ids = [tokenizer.encode(seq, add_special_tokens=True) for seq in data]

def preprocess_data(data_dir, tokenizer, max_seq_length):
    data = []
    target_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.mid') or filename.endswith('.xml'):
            score = music21.converter.parse(os.path.join(data_dir, filename))
            rhythmic_data = [str(note.quarterLength) for note in score.flatten().notes]
            data.append(' '.join(rhythmic_data))
            target_data.append([note.quarterLength for note in score.flatten().notes])

    input_ids = tokenizer.batch_encode_plus(
        data,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )['input_ids']

    return input_ids, target_data

