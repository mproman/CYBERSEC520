---
title: Class 05 - Sequential Models for Cybersecurity
subject: Lecture Notes
subtitle: From RNNs to Transformers to Large Language Models
short_title: Class 05
authors:
  - name: Michael Roman
    affiliations:
      - Duke University
    email: michael.roman@duke.edu
license: CC-BY-4.0
keywords: [Sequential Models, RNN, LSTM, Transformers, Attention Mechanism, BERT, Large Language Models, Phishing Detection, Text Classification]
abstract: |
  We explore sequential neural network architectures and their application to cybersecurity problems. Starting with a quick recap of MLPs and FastAI, we progress through Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, and Transformers. Each architecture is demonstrated on phishing URL and email detection tasks. We compare traditional fine-tuned approaches with modern Large Language Model APIs, examining tradeoffs in accuracy, latency, and computational cost. The class culminates with a comprehensive assignment requiring implementation of all three approaches on email phishing detection.
---

:::{caution}
:class: dropdown
This is an experimental format. The content of this page was adapted from Professor Roman's lecture in Fall 2025. Claude 4.5 was used to clean up the audio transcripts and adapt them for this format. AI may make mistakes. If you run into any issues please open an `Issue` or send an email to the TA [Sasank](mailto:sasank.g@duke.edu)
:::

# Class 5: Sequential Models for Cybersecurity

Welcome to Class 5. Today we're introducing sequential models, and this is where things get really interesting for cybersecurity applications. The first four weeks gave you a solid foundation in machine learning fundamentals. Now we're going to apply those concepts to more sophisticated problems where the order of data matters.

## Quick Recap: MLPs with PyTorch and FastAI

Before we jump into sequential models, let's do a quick review of what we covered last week. We built Multi-Layer Perceptron (MLP) models for port scan detection using both PyTorch and FastAI. This recap is important because it reinforces the workflow you'll use throughout the rest of the course.

### The Port Scan Detection Problem

Remember our dataset: network traffic features from the CICIDS dataset, specifically Friday afternoon port scan activity. We had 79 features including destination port, flow duration, packet counts, and various statistical measures of network behavior. The task was binary classification: is this traffic benign or a port scan?

### Building MLPs with PyTorch

Let me show you the PyTorch approach we used. First, we imported the necessary libraries and loaded our data:

```{code-cell} python
:tags: [hide-input]
:label: pytorch-recap

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Data preprocessing
X = df.drop(' Label', axis=1).values
y = df[' Label'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
```

The key insight here is that PyTorch requires data in tensor format. The `StandardScaler` normalizes our features to have zero mean and unit variance, which helps training stability. The `LabelEncoder` converts our text labels into numeric values: 0 for benign, 1 for port scan.

Next, we defined our MLP architecture as a Python class:

```{code-cell} python
:tags: [hide-input]
:label: mlp-architecture

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize model
model = MLP(X_train.shape[1], 64, len(encoder.classes_))
```

This architecture is straightforward: we have an input layer that takes our 79 features, a hidden layer with 64 neurons, and an output layer with 2 neurons for binary classification. The ReLU activation function introduces non-linearity between the layers.

The training loop requires several hyperparameters:

```{code-cell} python
:tags: [hide-input]
:label: training-hyperparameters

learning_rate = 0.001
num_epochs = 10
batch_size = 64

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

Cross-entropy loss is perfect for classification tasks because it penalizes confident wrong predictions more heavily than uncertain predictions. The Adam optimizer adapts the learning rate for each parameter, helping us avoid getting stuck in local minima.

The training loop processes data in batches:

```{code-cell} python
:tags: [hide-input]
:label: training-loop

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])
    
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_x)
        loss = loss_function(outputs, batch_y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
```

The key steps: zero out previous gradients, compute predictions, calculate loss, backpropagate to compute gradients, and update weights. We achieved nearly 100% accuracy on this dataset, which shows that even simple neural networks can excel at port scan detection.

### FastAI: A Higher-Level Abstraction

PyTorch gives you fine-grained control, but FastAI provides a more convenient API. Here's the same problem solved with FastAI:

```{code-cell} python
:tags: [hide-input]
:label: fastai-approach

from fastai.tabular.all import *

# Create data splits
splits = RandomSplitter(valid_pct=0.2)(range_of(df))

# Create TabularDataLoaders
data = TabularDataLoaders.from_df(
    df,
    procs=[Categorify, FillMissing, Normalize],
    cont_names=list(df.drop(' Label', axis=1).columns),
    y_names=" Label",
    splits=splits,
    bs=64
)

# Create learner
learn = tabular_learner(
    data,
    layers=[64],
    metrics=accuracy,
    loss_func=CrossEntropyLossFlat()
)

# Train
learn.fit_one_cycle(1, 0.001)
```

FastAI automatically handles data preprocessing, creates train-validation splits, and provides convenient training methods like `fit_one_cycle`. The `CrossEntropyLossFlat()` is FastAI's version that handles shape inconsistencies automatically.

The `tabular_learner` abstracts away the architecture definition. You specify the layer sizes, and FastAI builds the network with ReLU activations between layers. This is great for rapid prototyping and experimentation.

:::{tip}
**When to use PyTorch vs FastAI**: Use PyTorch when you need fine-grained control over architecture or training loops. Use FastAI when you want to quickly experiment with standard architectures. For research and production systems, PyTorch is standard. For exploration and learning, FastAI accelerates development.
:::

## Why Sequential Models Matter

Multi-Layer Perceptrons work great for tabular data where each feature is independent. But what happens when the order of your data matters?

Consider these cybersecurity scenarios:

**Phishing URLs**: The URL `paypal.com/login` is legitimate, but `suspicious-site.com/paypal.com/login` is likely phishing. The order of the characters matters. An MLP treats each character position independently, missing the sequential structure.

**Log file analysis**: A sequence of failed login attempts followed by successful access might indicate credential stuffing. The temporal order is critical for detection.

**Network traffic patterns**: A port scan touches sequential ports (22, 23, 24, 25...). The sequence reveals the attack pattern.

**Malware behavior**: API call sequences distinguish legitimate software from malware. The order of operations matters more than which APIs are called.

MLPs have a fundamental limitation: they treat all inputs as independent features. If you reshape a sequence into a fixed-size vector, you lose the temporal structure. If you feed sequences of varying lengths, MLPs can't handle the variable dimensions.

This is where sequential models come in. They're designed to process ordered data, maintain memory of previous inputs, and handle variable-length sequences naturally.

## Recurrent Neural Networks (RNNs)

The key idea behind Recurrent Neural Networks is simple but powerful: maintain a hidden state that gets updated as you process each element of the sequence.

### The Basic RNN Architecture

Think of an RNN as a neural network with memory. At each time step $t$, the network receives an input $x_t$ and produces an output $y_t$. But critically, it also maintains a hidden state $h_t$ that captures information from all previous inputs.

The hidden state update follows this formula:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

Where:
- $h_t$ is the hidden state at time $t$
- $h_{t-1}$ is the previous hidden state
- $x_t$ is the current input
- $W_{hh}$ are weights connecting the previous hidden state to the current state
- $W_{xh}$ are weights connecting the input to the hidden state
- $b_h$ is the bias term
- $\tanh$ is the activation function, squashing values to $[-1, 1]$

The output at each step is computed as:

$$y_t = W_{hy} h_t + b_y$$

This architecture has elegant properties. The same weights are reused at every time step, making the model parameter-efficient. The hidden state accumulates information from the sequence, providing context for each prediction.

### Processing a Sequence

Let's trace through a concrete example. Suppose we're analyzing a URL character by character to detect phishing:

**Input sequence**: `"p", "a", "y", "p", "a", "l"`

**Step 1**: Process `"p"`
- Input: embedding of character "p"
- Hidden state: $h_1 = \tanh(W_{hh} h_0 + W_{xh} x_1 + b_h)$
- $h_0$ is initialized (often to zeros)

**Step 2**: Process `"a"`
- Input: embedding of character "a"
- Hidden state: $h_2 = \tanh(W_{hh} h_1 + W_{xh} x_2 + b_h)$
- Note: $h_2$ now contains information about both "p" and "a"

This continues through the sequence. By the final step, $h_6$ has processed the entire string "paypal" and can be used for classification.

### The Vanishing Gradient Problem

RNNs have a critical weakness. During backpropagation through time, gradients must flow backward through many time steps. At each step, the gradient is multiplied by the weight matrix and the derivative of the activation function.

The gradient for a parameter with respect to the loss at time $t$ involves a product:

$$\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W}$$

Each $\frac{\partial L_t}{\partial W}$ involves products of Jacobian matrices across time steps. If these matrices have eigenvalues less than 1, the gradient vanishes exponentially. If they have eigenvalues greater than 1, the gradient explodes.

The $\tanh$ activation function has derivatives in the range $(0, 1]$. Multiply many such values together, and you get vanishingly small gradients. This means the network can't learn long-range dependencies.

**Practical impact for cybersecurity**: If you're trying to detect a command injection attack where the malicious payload comes after 100 benign characters, a basic RNN will struggle. The gradient signal from the attack pattern can't propagate back through 100 time steps effectively.

:::{warning}
**RNN limitations in practice**: Basic RNNs work for sequences up to 10-20 steps. Beyond that, vanishing gradients severely limit their ability to learn dependencies. For most cybersecurity text processing tasks (URLs, log entries, code snippets), you need something more powerful.
:::

## Long Short-Term Memory Networks (LSTMs)

LSTMs solve the vanishing gradient problem through a clever architecture that maintains two separate pathways for information: a hidden state (like RNNs) and a cell state with gated access.

### The LSTM Architecture

An LSTM cell has four main components:

**1. Forget Gate**: Decides what information to discard from the cell state

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**2. Input Gate**: Decides what new information to add to the cell state

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**3. Cell State Update**: Combines forget and input operations

$$C_t = f_t \ast C_{t-1} + i_t \ast \tilde{C}_t$$

**4. Output Gate**: Decides what to output based on the cell state

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \ast \tanh(C_t)$$

The key insight: the cell state $C_t$ flows through the network with only minor linear interactions (element-wise multiplications). Gradients can flow backward through the cell state without vanishing, because there are no repeated matrix multiplications with small values.

The gates use the sigmoid activation function $\sigma$, which outputs values in $(0, 1)$. These act as "soft switches" that control information flow. A gate value near 0 blocks information; near 1 allows it through.

### How Gates Work Together

Let's trace through an example of processing the URL fragment `"paypal.com/suspicious-site"`:

**Processing "paypal"**: The LSTM builds up context in its cell state, learning that this looks like a legitimate domain name. The input gate allows this information in.

**Processing ".com"**: The forget gate maintains the "paypal" context. The cell state now contains information about the full domain "paypal.com".

**Processing "/suspicious"**: Here's where it gets interesting. The model needs to remember the "paypal" context while also processing the suspicious path. The forget gate doesn't discard the domain information—it's still relevant. The input gate allows the new path information in.

**Final classification**: The output gate produces a hidden state that considers both the legitimate-looking domain and the suspicious path structure. This combination is key to detecting URL-based phishing attacks.

### AWD-LSTM: A Production-Ready Architecture

For our phishing URL detection task, we use AWD-LSTM (ASGD Weight-Dropped LSTM), a variant optimized to prevent overfitting:

**Weight Dropping**: Random connections between LSTM layers are dropped during training, similar to dropout but applied to recurrent connections. This forces the network to be robust and prevents it from memorizing specific sequences.

**ASGD (Average Stochastic Gradient Descent)**: Instead of using just the current weight values, ASGD maintains an average of recent weight vectors. This often leads to better generalization and faster convergence.

### Language Model Pre-training

One powerful technique is to first train the LSTM as a language model—predicting the next character in a sequence—before fine-tuning for classification. This teaches the model general patterns in URLs before specializing for phishing detection.

```{code-cell} python
:tags: [hide-input]
:label: lstm-language-model

from fastai.text.all import *

# Create language model data loaders
dls_lm = TextDataLoaders.from_df(
    sample_df, 
    path='', 
    text_col='URL', 
    is_lm=True,  # Language model mode
    valid_pct=0.2, 
    bs=16
)

# Create language model learner
learn_lm = language_model_learner(
    dls_lm, 
    AWD_LSTM, 
    metrics=[accuracy, Perplexity()]
).to_fp16()

# Train language model
learn_lm.fit_one_cycle(3, 1e-2)
```

The `is_lm=True` parameter tells FastAI to create training data for language modeling: given a sequence of characters, predict the next character. Perplexity measures how well the model predicts the next character (lower is better).

After pre-training, we fine-tune for classification:

```{code-cell} python
:tags: [hide-input]
:label: lstm-classifier

# Create classifier data loaders
dls_clas = TextDataLoaders.from_df(
    sample_df,
    path='',
    text_col='URL',
    label_col='Label',  # Now we have labels
    valid_pct=0.2,
    text_vocab=dls_lm.vocab  # Reuse language model vocabulary
)

# Create classifier
learn_clas = text_classifier_learner(
    dls_clas, 
    AWD_LSTM, 
    drop_mult=0.2, 
    metrics=accuracy
)

# Find optimal learning rate
learn_clas.lr_find()

# Train classifier
learn_clas.fit_one_cycle(8, 1e-2)
```

The learning rate finder is crucial. It runs through a range of learning rates and plots the loss. You want to choose a learning rate where the loss is decreasing most rapidly but hasn't started to diverge.

### LSTM Results on Phishing URLs

On our phishing URL dataset (2,000 samples, 80/20 train-test split), the AWD-LSTM achieved approximately 77% accuracy. The confusion matrix revealed an interesting pattern:

```{figure} #lstm-confusion-matrix
:name: fig-lstm-confusion
:width: 80%

LSTM confusion matrix showing strong performance on benign URLs (247 true negatives) but moderate false positive rate (38 false positives) and notable false negative rate (54 false negatives). The model correctly identified 96 phishing URLs.
```

The false negatives are concerning from a security perspective—these are phishing URLs that slipped through. The false positives would lead to blocking legitimate URLs. For production deployment, you'd want to adjust the classification threshold based on your tolerance for false positives versus false negatives.

:::{important}
**Key insight about LSTMs**: They excel at capturing patterns in sequences but require substantial training data and computational resources. The language model pre-training helps significantly when you have limited labeled data—the model learns general URL structure from unlabeled data before specializing for phishing detection.
:::

## Transformers and the Attention Mechanism

LSTMs process sequences step by step, maintaining hidden state as they go. Transformers take a fundamentally different approach: process all positions in parallel using attention mechanisms.

### The Attention Idea

The core insight behind attention is simple: when processing a word or character, the model should focus on relevant parts of the input, regardless of their position.

Consider the sentence: "The bank by the river has steep cliffs." When processing the word "steep," attention allows the model to focus on "bank" and "cliffs" while downweighting "river." This is much more flexible than an LSTM's sequential processing.

### Self-Attention Mathematics

The attention mechanism computes a weighted combination of all positions in the sequence. For each position, we compute three vectors:

**Query (Q)**: What am I looking for?
**Key (K)**: What do I have to offer?
**Value (V)**: What is my actual content?

For an input sequence with positions $1, ..., n$, we compute:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let's break this down:

1. **$QK^T$**: We compute similarity scores between the query at each position and the keys at all positions. This is a matrix of size $n \times n$ where entry $(i,j)$ measures how much position $i$ should attend to position $j$.

2. **$\frac{1}{\sqrt{d_k}}$**: We scale by the square root of the key dimension. Without this, the dot products can become very large, pushing the softmax into regions with extremely small gradients.

3. **$\text{softmax}$**: We normalize the scores so they sum to 1 at each position. This converts raw similarity scores into attention weights.

4. **Multiply by $V$**: We compute a weighted sum of the value vectors, where the weights are the attention scores.

The result: for each position, we get a vector that combines information from all relevant positions in the sequence.

### Multi-Head Attention

Transformers use multiple attention mechanisms in parallel, called "heads." Each head learns to attend to different aspects of the input:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

Why multiple heads? Different heads can specialize. In a URL, one head might focus on the domain structure, another on the path components, and another on parameter patterns. This parallel processing captures complex relationships.

### Positional Encoding

Attention is permutation-invariant: if you shuffle the input positions, you get shuffled outputs. But we care about order! URLs are sequences where position matters.

Transformers solve this by adding positional encodings to the input embeddings:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$

$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

Where $pos$ is the position in the sequence and $i$ is the dimension. These sinusoidal functions create unique patterns for each position that the model can learn to use.

The positional encoding is added to the input embedding:

$$\text{input}_{\text{final}} = \text{embedding}(x) + PE_{pos}$$

This allows the model to distinguish between "paypal.com/suspicious" and "suspicious.com/paypal" even though they contain the same tokens.

### Why Transformers Outperform LSTMs

**Parallelization**: LSTMs must process sequences step by step. Transformers process all positions simultaneously, making them much faster to train on modern GPUs.

**Long-range dependencies**: LSTMs struggle with dependencies beyond 50-100 steps. Transformers can attend to any position directly, regardless of distance.

**Representational capacity**: Multi-head attention allows the model to learn multiple types of relationships simultaneously. An LSTM has a single hidden state that must encode everything.

**Training stability**: The attention mechanism avoids the vanishing gradient problem that plagues RNNs. Gradients flow through the attention weights without degradation.

For cybersecurity applications, this means transformers can effectively process entire log entries, email bodies, or code snippets without the limitations that constrain LSTMs.

## BERT and DistilBERT

BERT (Bidirectional Encoder Representations from Transformers) revolutionized natural language processing by demonstrating that large transformer models pre-trained on unlabeled text could be fine-tuned for specific tasks with relatively little labeled data.

### The BERT Architecture

BERT is built from transformer encoder layers stacked on top of each other. The base model has 12 layers, while BERT-large has 24. Each layer contains:

- Multi-head self-attention (12 heads)
- Feed-forward neural networks
- Layer normalization
- Residual connections

The key innovation: bidirectional context. Earlier models (like GPT) only looked at previous words. BERT looks at both previous and following context simultaneously, giving it a deeper understanding of language.

### DistilBERT: A Smaller, Faster Variant

DistilBERT is a distilled version of BERT—it achieves 97% of BERT's performance with only 40% of the parameters. This makes it much faster and more practical for production use.

The distillation process trains a smaller "student" model to mimic a larger "teacher" model (BERT). The student learns not just to match the teacher's final predictions, but also its internal representations.

For phishing detection, DistilBERT provides an excellent balance: strong enough to capture complex patterns, small enough to deploy efficiently.

### Fine-tuning DistilBERT for Phishing URLs

Let's walk through the implementation using HuggingFace transformers:

```{code-cell} python
:tags: [hide-input]
:label: distilbert-setup

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset

# Load pre-trained tokenizer and model
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, 
    num_labels=2
)
```

The tokenizer converts text into tokens that the model can process. DistilBERT uses WordPiece tokenization, which breaks words into subwords. For example, "suspicious" might become ["sus", "##picious"].

This subword approach handles several challenges:

- Unknown words: Even if the model hasn't seen "cryptomining", it can process "crypto" and "##mining" separately
- Spelling variations: "phishing" and "phising" share subword tokens
- Efficient vocabulary: Instead of millions of words, we only need ~30,000 subword tokens

### Tokenization Process

```{code-cell} python
:tags: [hide-input]
:label: tokenization

# Prepare datasets
train_df, test_df = train_test_split(
    hf_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=hf_df['label']
)

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# Tokenization function
max_len = 128

def tokenize(batch):
    return tokenizer(
        batch['text'], 
        truncation=True, 
        padding='max_length', 
        max_length=max_len
    )

# Apply tokenization
train_ds_tok = train_ds.map(tokenize, batched=True, remove_columns=['text'])
test_ds_tok = test_ds.map(tokenize, batched=True, remove_columns=['text'])

# Set format for PyTorch
train_ds_tok.set_format(
    type='torch', 
    columns=['input_ids', 'attention_mask', 'label']
)
test_ds_tok.set_format(
    type='torch', 
    columns=['input_ids', 'attention_mask', 'label']
)
```

The `max_length=128` parameter limits sequences to 128 tokens. Most URLs fit comfortably within this limit. Longer sequences are truncated, shorter ones are padded.

The attention mask tells the model which tokens are real (1) and which are padding (0). This prevents the model from attending to padding tokens.

### Training Configuration

```{code-cell} python
:tags: [hide-input]
:label: training-config

args = TrainingArguments(
    output_dir="/content/distilbert_phishing",
    eval_strategy="epoch",
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    logging_strategy="epoch",
    logging_steps=5,
    logging_first_step=True,
    report_to=[]  # Disable WandB logging
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds_tok,
    eval_dataset=test_ds_tok
)

# Train the model
trainer.train()
```

The learning rate (2e-5) is quite small. This is intentional: we're fine-tuning a pre-trained model, not training from scratch. Aggressive learning rates would destroy the useful representations BERT already learned.

Weight decay (0.01) provides regularization by penalizing large weights. This helps prevent overfitting on our relatively small dataset.

### DistilBERT Results

After 5 epochs of training, DistilBERT achieved impressive results on the phishing URL dataset:

**Final Metrics**:
- Training loss: 0.043
- Validation loss: 0.229
- Accuracy: 95%

The confusion matrix tells the detailed story:

```{figure} #distilbert-confusion-matrix
:name: fig-distilbert-confusion
:width: 80%

DistilBERT confusion matrix showing excellent performance: 283 true negatives (correctly identified benign URLs) and 96 true positives (correctly identified phishing URLs), with only 6 false positives and 15 false negatives.
```

Compare this to the LSTM results:
- **LSTM**: 77% accuracy, 54 false negatives
- **DistilBERT**: 95% accuracy, 15 false negatives

The transformer's bidirectional attention mechanism captures URL patterns that the LSTM missed. It better understands the relationship between domain and path components.

### Classification Report

```
              precision    recall  f1-score   support

    Good URL       0.95      0.98      0.96       289
     Bad URL       0.94      0.86      0.90       111

    accuracy                           0.95       400
   macro avg       0.95      0.92      0.93       400
weighted avg       0.95      0.95      0.95       400
```

The precision-recall tradeoff is favorable. High precision (0.94) on phishing URLs means few false alarms. Good recall (0.86) means we catch most attacks, though 15 phishing URLs still slip through.

:::{tip}
**Production deployment consideration**: For a security application, you might lower the classification threshold to increase recall at the cost of precision. It's better to flag a few extra legitimate URLs for manual review than to miss actual phishing attacks.
:::

## Large Language Models via API

The latest approach to text classification is to use large language models through API calls. Instead of fine-tuning your own model, you send requests to a hosted model like GPT-4, Claude, or Gemini.

### Zero-Shot Classification

LLMs can perform classification without task-specific training. You provide a prompt describing the task, and the model generates predictions based on its general language understanding.

For phishing URL detection:

```python
prompt = [
    "Is this URL likely to be phishing (bad) or safe (good)?:",
    url
]
```

The model hasn't seen labeled examples of phishing URLs during training (hence "zero-shot"), but its general knowledge about web security patterns allows it to make reasonable predictions.

### Structured Outputs with Gemini

We use Google's Gemini 2.5 Flash model, which is optimized for low latency and cost. The key is to constrain outputs to a specific format:

```{code-cell} python
:tags: [hide-input]
:label: gemini-api

import google.generativeai as genai
from google.colab import userdata

# Configure API
API_KEY = userdata.get('GeminiAPI')
genai.configure(api_key=API_KEY)
flash = genai.GenerativeModel('gemini-2.5-flash')

def classify_url_flash(url):
    prompt = [
        "Is this URL likely to be phishing (bad) or safe (good)?:", 
        url
    ]
    
    # Constrain output to enum
    cfg = genai.GenerationConfig(
        response_mime_type='text/x.enum',
        response_schema={
            'type': 'STRING',
            'enum': ['good', 'bad']
        }
    )
    
    resp = flash.generate_content(prompt, generation_config=cfg)
    return resp.text.strip()
```

The `response_schema` with `enum` is crucial. Without this constraint, the model might respond with "This URL appears to be phishing because..." We need just "good" or "bad" for automated classification.

### Measuring Performance and Latency

```{code-cell} python
:tags: [hide-input]
:label: llm-evaluation

import time
from sklearn.metrics import f1_score

num_runs = 20
predictions = []
true_labels = []
response_times = []

def map_prediction_to_label(pred):
    pred = pred.strip().lower()
    if pred == 'good':
        return 0
    elif pred == 'bad':
        return 1
    else:
        return -1

for i in range(num_runs):
    url = test_df['text'].iloc[i]
    true_label = test_df['label'].iloc[i]
    
    start = time.time()
    prediction = classify_url_flash(url)
    predicted_label = map_prediction_to_label(prediction)
    elapsed = time.time() - start
    
    response_times.append(elapsed)
    predictions.append(predicted_label)
    true_labels.append(true_label)

# Calculate metrics
f1 = f1_score(true_labels, predictions, pos_label=1, average='binary')
avg_response = sum(response_times) / num_runs

print(f"F1-score over {num_runs} samples: {f1:.2%}")
print(f"Average API response time: {avg_response:.2f} seconds")
```

Results on 20 samples:
- **F1-score**: 26.09%
- **Average latency**: 0.86 seconds per request

The F1-score is surprisingly low given the model's capabilities. This is a zero-shot baseline—the model hasn't been optimized for this specific task.

### Why Zero-Shot Underperforms

The low accuracy reveals several issues:

1. **Task ambiguity**: "Phishing" is complex. Is `paypal-secure-login.com` phishing? It depends on whether it's the real PayPal login or a spoof. The model needs more context.

2. **URL structure vs. content**: The model sees only the URL string, not the actual webpage content. Many phishing sites use legitimate-looking URLs.

3. **No task-specific training**: DistilBERT was fine-tuned on thousands of labeled phishing URLs. Gemini has general web knowledge but no specific phishing detection training.

### Improving LLM Performance

In the next class, we'll explore prompt engineering strategies:

**Few-shot learning**: Provide examples in the prompt:
```
Good URL examples:
- paypal.com/login
- amazon.com/checkout

Bad URL examples:
- paypa1.com/secure-login
- amaz0n-secure.com/verify

Now classify: paypal-verify.suspicious-site.com
```

**Chain-of-thought prompting**: Ask the model to explain its reasoning before answering, which improves accuracy.

**Structured reasoning**: Break the task into steps (identify domain, check path, analyze parameters) rather than asking for a direct classification.

## Comparing the Three Approaches

We've implemented phishing URL detection using three different architectures. Let's compare them systematically:

| Approach | Accuracy | Training Time | Inference Latency | Model Size | Fine-tuning Required |
|----------|----------|---------------|-------------------|------------|---------------------|
| AWD-LSTM | 77% | ~15 min | <10ms | ~50MB | Yes |
| DistilBERT | 95% | ~5 min | ~20ms | ~250MB | Yes |
| Gemini API | 26% (zero-shot) | None | ~860ms | N/A (API) | No |

### When to Use Each Approach

**LSTM (AWD-LSTM)**:
- **Best for**: Resource-constrained environments, embedded systems
- **Strengths**: Small model size, fast inference, good for sequences up to ~200 tokens
- **Weaknesses**: Lower accuracy than transformers, requires language model pre-training for best results
- **Cybersecurity applications**: IoT security, edge device monitoring, mobile security apps

**Transformer (DistilBERT)**:
- **Best for**: Production systems where accuracy is critical
- **Strengths**: State-of-the-art accuracy, handles long sequences well, captures complex patterns
- **Weaknesses**: Larger model size, slightly slower inference than LSTMs
- **Cybersecurity applications**: Email filtering systems, malware detection, advanced threat protection

**LLM API (Gemini/GPT)**:
- **Best for**: Rapid prototyping, low-volume applications, exploratory analysis
- **Strengths**: No training required, can be improved with prompt engineering, extremely flexible
- **Weaknesses**: High latency, API costs, requires internet connection, lower accuracy without fine-tuning
- **Cybersecurity applications**: Security research, threat intelligence analysis, manual investigation assistance

### Error Analysis

Looking at the failure modes reveals interesting patterns:

**URLs that fool all models**:
- `legitimate-looking-domain.com/malicious-path`
- IDN homograph attacks: `pаypal.com` (contains Cyrillic 'а')
- Newly registered domains that haven't been seen in training

**URLs where DistilBERT excels over LSTM**:
- Long URLs with complex path structures
- URLs with multiple subdomains
- Parameter-heavy URLs (common in phishing kits)

**URLs where even zero-shot LLM succeeds**:
- Obvious misspellings: `g00gle.com`
- Well-known phishing patterns: `verify-account-suspended.com`

### Cost Considerations

For a production system processing 1 million URLs daily:

**LSTM**:
- Infrastructure: ~$50/month (small CPU instance)
- Training: One-time cost, ~$5 on Google Colab
- Total monthly: ~$50

**DistilBERT**:
- Infrastructure: ~$100/month (small GPU instance)
- Training: One-time cost, ~$10 on Google Colab
- Total monthly: ~$100

**Gemini API**:
- API costs: ~$0.10 per 1K requests
- 1M requests/day × 30 days = 30M requests/month
- Total monthly: ~$3,000

The API cost is prohibitive at scale. LLM APIs make sense for low-volume applications (<10K requests/day) or when rapid deployment is more important than cost.

:::{important}
**Hybrid approach for production**: Use DistilBERT for real-time filtering (high volume, low cost), and route uncertain cases (prediction confidence between 0.4 and 0.6) to an LLM API for secondary analysis. This balances accuracy, latency, and cost.
:::

## Assignment: Phishing Email Detection with Three Approaches

Now it's your turn to implement these architectures. You'll build and compare all three approaches on email phishing detection—a more challenging task than URLs because emails contain longer, more varied text.

### Objective

Build, evaluate, and compare the following three approaches for phishing email detection using the [Kaggle Phishing Emails dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset):

1. Classic RNN or LSTM architecture
2. A transformer-based classifier
3. LLM API approach

### Dataset Preparation

**Data source**: Use the Kaggle Phishing Emails dataset. Download and load the data into your notebook.

**Label mapping**: Treat the email body/text column as input. Map labels to binary targets where:
- `0` = safe/good (legitimate emails)
- `1` = phishing/bad (phishing emails)

This convention matches the class examples and makes your code consistent with the provided notebooks.

**Data splitting**: Perform a reproducible train/validation/test split:

```python
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
```

Use `random_state=42` throughout to ensure reproducibility.

**Exploratory Data Analysis**: In your notebook, report:
- Class balance (percentage of phishing vs. legitimate)
- Token length statistics (min, max, mean, median)
- A few representative examples of each class

Understanding your data distribution helps interpret model performance later.

### Model 1: RNN/LSTM

Implement a supervised neural sequence classifier using PyTorch, Keras, or FastAI.

**Architecture requirements**:
- Document your tokenization approach (character-level, word-level, or subword)
- Specify vocabulary size or subword encoding strategy
- Define maximum sequence length
- Describe embedding strategy (pre-trained or learned)
- Provide training schedule (learning rate, batch size, epochs)

**Training**:
- Train on the training set
- Monitor validation performance to detect overfitting
- Use early stopping if validation loss stops improving

**Evaluation**:
- Test on the held-out test set
- Report: accuracy, precision, recall, F1-score
- Visualize: confusion matrix
- Discuss: Signs of overfitting/underfitting? Training instabilities?

**Code example starter**:

```python
from fastai.text.all import *

# Language model pre-training
dls_lm = TextDataLoaders.from_df(
    train_df,
    text_col='email_body',
    is_lm=True,
    valid_pct=0.1
)

learn_lm = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()])
learn_lm.fit_one_cycle(3, 1e-2)

# Classification fine-tuning
dls_clas = TextDataLoaders.from_df(
    train_df,
    text_col='email_body',
    label_col='label',
    valid_pct=0.1,
    text_vocab=dls_lm.vocab
)

learn_clas = text_classifier_learner(dls_clas, AWD_LSTM, metrics=accuracy)
learn_clas.fit_one_cycle(8, 1e-2)
```

### Model 2: Transformer Classifier

Fine-tune a transformer model (DistilBERT recommended) on the email dataset.

**Architecture requirements**:
- Use DistilBERT or another distilled model (due to size constraints)
- Document `max_seq_length` (suggest 256 or 512 for emails)
- Specify learning rate (2e-5 is a good starting point)
- Define batch sizes (16 for training, 32 for evaluation)
- Set number of epochs (3-5 typically sufficient)

**Training**:
- Use HuggingFace `Trainer` or equivalent training loop
- Keep the 0/1 label convention
- Monitor validation metrics during training

**Evaluation**:
- Evaluate on the held-out test set
- Include classification report with precision/recall per class
- Visualize confusion matrix

**Code example starter**:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)

# Tokenize datasets
def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

train_ds_tok = train_ds.map(tokenize, batched=True, remove_columns=['text'])
test_ds_tok = test_ds.map(tokenize, batched=True, remove_columns=['text'])

# Training
args = TrainingArguments(
    output_dir="./distilbert_email_phishing",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    learning_rate=2e-5
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds_tok, eval_dataset=test_ds_tok)
trainer.train()
```

### Model 3: LLM API Call

Implement phishing classification using an LLM API with structured output.

**API requirements**:
- Use Google Gemini-2.5-Flash (recommended for low latency and cost)
- Alternative: Any LLM API is acceptable (GPT, Claude, etc.)
- Constrain output to enum: `["Safe Email", "Phishing Email"]`
- Map responses to 0/1 labels

**Evaluation**:
- Test on at least 50 held-out samples (more if API budget allows)
- Measure per-request latency using timing loop from class
- Calculate accuracy, precision, recall, F1-score

**Prompt engineering**:
Create at least two prompt variants:
1. **Zero-shot**: Just the classification task description
2. **Few-shot**: Include 2-3 examples of each class

Compare accuracy, false positives/negatives, and latency across prompts.

**Code example starter**:

```python
import google.generativeai as genai
import time

genai.configure(api_key=YOUR_API_KEY)
flash = genai.GenerativeModel('gemini-2.5-flash')

def classify_email(email_text, prompt_type='zero-shot'):
    if prompt_type == 'zero-shot':
        prompt = [
            "Is this email safe or phishing? Respond with exactly 'Safe Email' or 'Phishing Email':",
            email_text
        ]
    elif prompt_type == 'few-shot':
        prompt = [
            "Examples of Safe Emails:\n- 'Your order has shipped...'\n- 'Meeting tomorrow at 2pm...'\n\n"
            "Examples of Phishing Emails:\n- 'Verify your account immediately...'\n- 'You've won a prize, click here...'\n\n"
            "Classify this email as 'Safe Email' or 'Phishing Email':",
            email_text
        ]
    
    cfg = genai.GenerationConfig(
        response_mime_type='text/x.enum',
        response_schema={'type': 'STRING', 'enum': ['Safe Email', 'Phishing Email']}
    )
    
    start = time.time()
    resp = flash.generate_content(prompt, generation_config=cfg)
    latency = time.time() - start
    
    return resp.text.strip(), latency

# Evaluate
results = []
for i in range(50):
    email = test_df.iloc[i]['email_body']
    true_label = test_df.iloc[i]['label']
    pred, lat = classify_email(email, prompt_type='zero-shot')
    results.append({'pred': pred, 'true': true_label, 'latency': lat})
```

:::{note}
**API Key Setup**: Make sure you setup your API key for the LLM provider. We'll discuss prompt engineering strategies in the next class to improve performance. For now, focus on getting the basic implementation working and measuring baseline performance.
:::

### Comparative Analysis

Compare all three approaches on the same test set:

**Metrics comparison table**:
- Accuracy
- Precision (per class)
- Recall (per class)
- F1-score
- Average inference latency
- Model size (for local models)
- Training time

**Error analysis**:
- Identify precision/recall tradeoffs
- Document typical false positives (legitimate emails flagged as phishing)
- Document typical false negatives (phishing emails that evade detection)
- Analyze consistent failure modes across models

**Practical considerations**:
- Which approach would you deploy in production? Why?
- How would you handle the speed vs. accuracy tradeoff?
- What are the cost implications at scale?

### Deliverables

Submit a single notebook (or multiple linked notebooks) containing:

1. **Exploratory Data Analysis**:
   - Dataset statistics
   - Class distribution
   - Sample examples

2. **RNN/LSTM Implementation**:
   - Architecture definition
   - Training code
   - Evaluation metrics
   - Confusion matrix

3. **Transformer Implementation**:
   - Model fine-tuning
   - Training code
   - Evaluation metrics
   - Confusion matrix

4. **LLM API Implementation**:
   - API setup and authentication
   - Prompt variants (zero-shot and few-shot)
   - Latency measurements
   - Accuracy comparison

5. **Comparative Analysis**:
   - Metrics comparison table
   - At least two figures (bar chart of metrics, confusion matrices)
   - Written discussion (2-3 paragraphs) comparing approaches
   - Error analysis with examples

**Code quality requirements**:
- Cells run top-to-bottom without errors
- Fixed random seeds for reproducibility (`random_state=42`, `torch.manual_seed(42)`)
- Clear comments explaining key steps
- Consistent with class practices (0/1 label convention, train/val/test splits)

### Hints and Best Practices

**For RNN/LSTM**:
- Keep architectures lightweight (single LSTM layer with 64-128 hidden units)
- Use early stopping to prevent overfitting
- Language model pre-training significantly improves results
- Monitor training loss—if it plateaus quickly, increase model capacity

**For Transformers**:
- Start with 3 epochs and increase if validation loss is still decreasing
- Use learning rate of 2e-5 to 5e-5
- If you run out of memory, reduce batch size or sequence length
- Save checkpoints in case training is interrupted

**For LLM API**:
- Use strict enum schema to ensure parseable outputs
- Implement error handling for API failures
- Cache results to avoid redundant API calls during development
- Monitor API costs—stop if you're exceeding budget

**General**:
- Save intermediate artifacts (tokenizers, label maps, model checkpoints)
- Create functions for repeated operations (evaluation, confusion matrix plotting)
- Document your hyperparameter choices
- If results seem too good or too bad, double-check your label mapping

## Final Project Overview

The final project applies everything you've learned to a cybersecurity problem of your choice. This is your opportunity to demonstrate deep understanding of machine learning applied to real-world security challenges.

### Project Structure

**Problem Identification**: Choose a cybersecurity problem where machine learning can make an impact. Examples:
- Malware classification from static features
- Network intrusion detection from traffic patterns
- Vulnerability prediction from code features
- Threat intelligence from social media data
- Anomaly detection in system logs

**Dataset Selection**: Find appropriate data. Sources include:
- Kaggle datasets
- CICIDS network traffic datasets
- VirusTotal malware samples
- GitHub security data
- Public CTF datasets

**Multiple Model Comparison**: Apply at least three different approaches:
- Traditional ML (Random Forest, SVM, etc.)
- Neural networks (MLPs, CNNs, RNNs, Transformers)
- Compare architectures within a family (LSTM vs. GRU vs. Transformer)

**Performance Benchmarking**: 
- Comprehensive evaluation metrics
- Confusion matrices
- ROC curves (if appropriate)
- Precision-recall tradeoffs
- Computational cost analysis

### Presentation Format

The final presentation is an executive briefing, not a technical deep-dive. Imagine you're presenting to a Chief Information Security Officer (CISO) or Board of Directors to justify R&D spending.

**Target audience**: Category 2 executives (technical background but not ML experts)

**Time limit**: 5-10 minutes

**Required elements**:

1. **Problem Statement** (1 minute):
   - What security challenge are you addressing?
   - Why does this matter? What's the business impact?
   - Current state: How is this handled today?

2. **Approach** (2-3 minutes):
   - What dataset did you use?
   - What models did you evaluate?
   - Why did you choose these approaches?

3. **Results** (2-3 minutes):
   - Performance comparison across models
   - Key metrics (accuracy, false positive rate, etc.)
   - Best-performing approach and why

4. **Recommendation** (1-2 minutes):
   - Which approach would you deploy?
   - What's the expected ROI?
   - What are the limitations and risks?
   - Should the company invest further?

**Visual requirements**:
- Clear charts comparing model performance
- Confusion matrix or ROC curve
- Cost/benefit analysis (if applicable)
- No code in slides—show results, not implementation

### Evaluation Criteria

Your project will be evaluated on:

**Technical execution** (40%):
- Appropriate model selection
- Correct implementation
- Thorough evaluation
- Valid train/test methodology

**Analysis quality** (30%):
- Insightful comparison of approaches
- Understanding of tradeoffs
- Error analysis and failure modes
- Realistic assessment of limitations

**Presentation clarity** (30%):
- Clear problem motivation
- Appropriate for audience level
- Professional visual design
- Compelling recommendation

### Tips for Success

**Choose problems with clear metrics**: Binary classification (malware vs. benign) is easier to evaluate than regression (severity scoring). Avoid problems where ground truth is ambiguous.

**Start simple**: Implement a baseline model (logistic regression) first. This gives you a performance floor and helps debug your data pipeline.

**Budget your time**: Don't spend 80% of time on data collection and 20% on modeling. Find datasets early and allocate time for thorough evaluation.

**Tell a story**: Your presentation should have a narrative arc: "Here's a problem → Here's what I tried → Here's what worked → Here's what we should do."

**Practice your pitch**: Rehearse your presentation. 5-10 minutes goes quickly. Focus on the most important results, not every detail.

**Be honest about limitations**: If your best model only achieves 70% accuracy, explain why. Real-world security problems are hard. Demonstrating understanding of limitations shows maturity.

## Looking Ahead

We've covered three major architecture families in this class: RNNs, LSTMs, and Transformers. Each has its place in the cybersecurity toolkit.

In the next class, we'll dive deeper into prompt engineering for LLMs. You'll learn techniques to dramatically improve the 26% F1-score we saw with zero-shot classification:

- Few-shot learning with strategic example selection
- Chain-of-thought prompting for complex reasoning
- Structured output formats for reliable parsing
- Temperature and sampling strategies
- Prompt injection defenses (critical for security applications)

You'll also see how to use LLMs for tasks beyond classification: threat intelligence summarization, security policy generation, and automated incident response.

For now, focus on completing the assignment. Getting hands-on experience with all three approaches will give you intuition about their strengths and weaknesses. This practical knowledge is invaluable when deciding which architecture to deploy in production.

## Resources and References

### Documentation
- [FastAI Documentation](https://docs.fast.ai/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Google Gemini API](https://ai.google.dev/tutorials/python_quickstart)

### Research Papers
- Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
- Vaswani et al. (2017): "Attention Is All You Need"
- Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers"
- Merity et al. (2017): "Regularizing and Optimizing LSTM Language Models" (AWD-LSTM)

### Datasets
- [CICIDS Network Traffic](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Kaggle Phishing Emails](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
- [Kaggle Phishing URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)

### Interactive Tools
- [TensorFlow Playground](https://playground.tensorflow.org/) - Visualize neural network training
- [LSTM Visualizer](http://lstm.seas.harvard.edu/) - See LSTM gates in action
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) - Interactive transformer visualization
