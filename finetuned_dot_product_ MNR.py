
import logging
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)

dataset = load_dataset('csv', data_files={'train': 'C:/Users/busin/Downloads/updated_conversations.csv'})

# Create training examples from the dataset

sentences = [example['Conversation'] for example in dataset['train']]

train_examples = []
for i in range(0, len(sentences), 2):
    if i+1 < len(sentences):
        train_examples.append(InputExample(texts=[sentences[i], sentences[i+1]]))
        
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-dot-v1')

# Dataloader has to be prepared and loss function is used here
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

num_epochs = 2  # Adjust the number of epochs as needed
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up

# Start training
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps)

model.save('C:/Users/busin/Downloads/sienna-dot-finetune')

