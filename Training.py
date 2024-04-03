# train.py
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import BertTokenizer, AdamW
from DataPre import preprocess_data, target_data
from ModelDef import RhythmicModel

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and preprocess data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data_dir = r"C:\Users\miles\OneDrive\Desktop\Junior IS\Training data"
max_seq_length = 4
input_ids, target_data = preprocess_data(data_dir, tokenizer, max_seq_length)

# Move data to device
input_ids = input_ids.to(device)
target_data = torch.tensor(target_data, dtype=torch.float32).to(device)

# Load pre-trained BERT model and create an instance of the custom model
config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
bert_model = BertModel.from_pretrained('bert-base-uncased', config=config)
model = RhythmicModel(bert_model).to(device)

# Define loss function
criterion = nn.MSELoss()

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
num_epochs = 10
batch_size = 32
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[i:i+batch_size]
        batch_target_data = target_data[i:i+batch_size]

        # Forward pass
        outputs = model(batch_input_ids)
        loss = criterion(outputs.squeeze(-1), batch_target_data)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    # Update learning rate
    scheduler.step()

    print(f'Epoch {epoch+1} / {num_epochs}, Loss: {total_loss / len(input_ids)}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.py')