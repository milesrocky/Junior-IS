# ModelDef.py
import torch
from transformers import BertModel, BertConfig

# Load the pre-trained BERT model
config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
bert_model = BertModel.from_pretrained('bert-base-uncased', config=config)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RhythmicModel(torch.nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.classifier = torch.nn.Linear(bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        return logits

# Create an instance of the RhythmicModel with the pre-trained BERT model instance
model = RhythmicModel(bert_model).to(device)