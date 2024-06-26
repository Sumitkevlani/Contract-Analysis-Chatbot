## Model Overview
This model uses a decoder-based transformer architecture to predict the outputs of questions related to a PDF containing contracts between employers and vendors. The model is trained to understand the context and provide relevant responses based on the training data extracted from the PDF.

## Model Architecture

The model is based on the transformer architecture and includes the following components:

### Embedding Layers:

- Token Embeddings: Converts input tokens into dense vectors of a specified dimension (n_embd).
- Position Embeddings: Adds information about the position of each token in the sequence, which helps the model understand the order of tokens.
- Transformer Blocks: The model consists of multiple transformer blocks, each containing:

### Multi-Head Self-Attention:
- Heads: The self-attention mechanism allows the model to focus on different parts of the input sequence simultaneously. Each attention head operates independently, projecting the input into query, key, and value vectors.
- Attention Mechanism: The attention scores are computed as the dot product of query and key vectors, scaled by the square root of the dimension, followed by a softmax function. These scores are then used to weight the value vectors.
- Concatenation and Projection: The outputs from all heads are concatenated and projected back into the original embedding dimension.
- Feed-Forward Neural Networks:
- Linear Layers: Consists of two linear layers with a ReLU activation in between, providing non-linearity to the model.
- Dropout: Applied after the linear layers to prevent overfitting.
- Layer Normalization: Applied before each sub-layer (self-attention and feed-forward) to normalize the inputs, improving training stability.
- Residual Connections: Adds the input of each sub-layer to its output, allowing gradients to flow through the network more easily and helping in training deeper networks.

### Output Layer:

- Linear Layer: Projects the hidden states of the last transformer block back to the vocabulary size, producing logits for the next token prediction.
- Softmax: Converts logits to probabilities, used during inference to sample the next token.


### Active Learning:

- Uncertainty Sampling: The model incorporates an uncertainty sampling mechanism to select new training samples that are most uncertain. This enhances the learning process over time by focusing on examples where the model is least confident.
- Training Procedure
The training procedure includes the following steps:

- Data Preparation:

1. Extract text from the PDF using PyPDF2.
2. Tokenize the text and create train and validation splits.
3. Map characters to integers (tokenization) and create input-output pairs for the model.

- Batch Generation:

1. Generate batches of data for training and validation, where each input sequence is paired with its corresponding target sequence (shifted by one position).

- Model Training:

1. Initialize the model, optimizer (AdamW), and loss function (Cross-Entropy Loss).
2. For each training iteration:
Sample a batch of data.
Perform a forward pass through the model to compute logits and loss.
Perform a backward pass to compute gradients.
Update model parameters using the optimizer.
Periodically evaluate the model on the validation set to monitor training progress and prevent overfitting.

3. Active Learning Iterations:

At regular intervals, perform uncertainty sampling to select new samples for training.
Label these new samples and include them in the training process to improve model performance.

## Installation:
```bash
  python -m venv cuda
  ```
```bash  
  cd env/Scripts
  ```  
```bash  
  .\activate
  ```  
```bash
  pip install PyPDF2 torch
 ```

## References:
- Research Paper: [Attention is All you Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- Video Explanation: [Transformer Model](https://www.youtube.com/live/SMZQrJ_L1vo?si=PXTreQ1YRCbCwf68)
- Code Explanation: [Create your own LLM](https://youtu.be/UU1WVnMk4E8?si=ExJu4wqBkFfkPgEl)