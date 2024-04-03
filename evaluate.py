import torch
from transformers import BertTokenizer, BertModel, BertConfig
from ModelDef import RhythmicModel

# Load the pre-trained BERT model
config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
bert_model = BertModel.from_pretrained('bert-base-uncased', config=config)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RhythmicModel(bert_model)
model.to(device)
model.load_state_dict(torch.load('trained_model.py', map_location=device))
model.eval()

# Load tokenizer and preprocess input data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_data = [r"C:\Users\miles\OneDrive\Desktop\Junior IS\Training data\4 piano whole notes.mid"]
input_ids = tokenizer.batch_encode_plus(
    input_data,
    max_length=4,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)['input_ids']
input_ids = input_ids.to(device)

# Run the model on the input data
with torch.no_grad():
    outputs = model(input_ids)

# Analyze the model's output
for i, output in enumerate(outputs.squeeze(-1)):
    print(f"Input: {input_data[i]}")
    print(f"Predicted note lengths: {output}")