import torch
from embedding import InputEmbeddings  # Make sure to adjust this import based on your file structure
from config import D_MODEL, VOCAB_SIZE, MAX_SEQ_LEN, FF_DROPOUT

# Initialize the embedding model
input_embeddings = InputEmbeddings()

# Example input tensor: batch of 2 sequences with 4 tokens each
sample_input = torch.tensor([
    [1, 5, 8, 2],
    [3, 7, 6, 4]
])

# Get the embeddings
embedded_output = input_embeddings(sample_input)

# Print the output shape
print("Embedded Output Shape:", embedded_output.shape)
# Output: (2, 4, 512) assuming the configuration values
