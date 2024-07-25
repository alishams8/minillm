import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from config import *
from embedding import InputEmbeddings

# Dummy Classifier
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embeddings = InputEmbeddings()
        self.fc = nn.Linear(D_MODEL, NUM_CLASSES)

    def forward(self, x):
        x = self.input_embeddings(x)
        x = x.mean(dim=1)  # Simple pooling: mean over sequence length
        return self.fc(x)

# Tokenization and Vocabulary Setup
def tokenize(text):
    tokenizer = get_tokenizer('basic_english')
    return tokenizer(text)

def build_vocab(texts):
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        map(tokenizer, texts),
        specials=['<unk>', '<pad>']
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def text_to_indices(text, vocab, max_seq_len):
    tokens = tokenize(text)
    indices = [vocab[token] for token in tokens]
    if len(indices) < max_seq_len:
        indices += [vocab['<pad>']] * (max_seq_len - len(indices))
    else:
        indices = indices[:max_seq_len]
    return indices

# Create a real dataset
def create_real_data(texts, vocab, max_seq_len, batch_size):
    indices = [text_to_indices(text, vocab, max_seq_len) for text in texts]
    return torch.tensor(indices, dtype=torch.long).reshape(batch_size, max_seq_len)

# Example texts and labels
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Hello world, this is a sample text",
    "Another example of text data for classification",
    "More text data to train the model"
] * (BATCH_SIZE // 4)  # Adjust to get the desired batch size

def train_and_evaluate():
    # Initialize model, loss, and optimizer
    model = SimpleClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Build vocabulary
    vocab = build_vocab(texts)

    # Create real dataset
    num_batches = 10
    for epoch in range(2):  # 2 epochs for demonstration
        for _ in range(num_batches):
            inputs = create_real_data(texts, vocab, MAX_SEQ_LEN, BATCH_SIZE)
            targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))  # Random target labels for demo

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss
            print(f'Epoch [{epoch+1}/2], Loss: {loss.item():.4f}')

        # Evaluation
        model.eval()
        with torch.no_grad():
            inputs = create_real_data(texts, vocab, MAX_SEQ_LEN, BATCH_SIZE)
            targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == targets).float().mean().item()
            print(f'Epoch [{epoch+1}/2] Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    train_and_evaluate()
