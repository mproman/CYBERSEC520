---
title: Class 04 - Introduction to Neural Networks
subject: Lecture Notes
subtitle: Understanding Deep Learning Fundamentals
short_title: Class 04 - Introduction to Neural Networks
authors:
  - name: Michael Roman
    affiliations:
      - Duke University
    email: michael.roman@duke.edu
license: CC-BY-4.0
keywords: neural networks, deep learning, activation functions, ReLU, gradient descent, universal approximation theorem, backpropagation, regularization, cybersecurity
abstract: |
  This lecture introduces neural network fundamentals through both theory and hands-on exploration. We cover network architecture, how individual neurons process information, activation functions, and the Universal Approximation Theorem through interactive demonstrations. The training process via gradient descent and backpropagation is explained with practical examples, followed by discussion of regularization techniques. An extended hands-on lab using TensorFlow Playground builds intuition about network parameters and their effects on learning.
kernelspec:
  display_name: 'Python 3'
  language: python3
  name: python3
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.0
---

:::{caution}
:class: dropdown
This is an experimental format. The content of this page was adapted from Professor Roman's lecture in Fall 2025. Claude 4.5 was used to clean up the audio transcripts and adapt them for this format. AI may make mistakes. If you run into any issues please open an `Issue` or send an email to the TA [Sasank](mailto:sasank.g@duke.edu)
:::

# Class 04 - Introduction to Neural Networks

Welcome to Class 4. Today we're getting into the foundational architecture that underpins modern AI: neural networks. We'll explore both theory and practice through an introduction to neural network fundamentals followed by a hands-on exploration lab. The lab focuses on building intuition rather than coding—you'll experiment with different parameters to understand how model architecture and hyperparameters impact training and performance.


## Today's Plan

Here's what we're covering:
1. Introduction to neural network architecture and mechanics
2. Hands-on exploration lab using TensorFlow Playground (~45 minutes)

We'll cover sequential models next week when we build a phishing detection model using natural language processing. We'll start it in class and you'll finish it as homework.

## Revisiting Machine Learning

Traditional machine learning algorithms learn patterns from data. We select the algorithm and objective (the loss function), then let the machine learn the optimal parameters—weights and biases—that minimize loss across the dataset. The fundamental limitation of these traditional approaches is their reliance on manual feature engineering. In previous classes, we've seen this challenge firsthand: carefully selecting features, applying domain knowledge to transform them, and iteratively refining our feature sets.

The biggest breakthroughs in AI—including the rise of Large Language Models—followed the advent of neural networks and deep learning. What makes deep learning transformative is its ability to automatically learn representations from data, essentially eliminating the need for manual feature engineering.

:::{important}
The power of deep learning is grounded in the **Universal Approximation Theorem**: a feedforward neural network with a single hidden layer containing sufficient neurons can approximate any continuous function to any desired degree of accuracy.
:::

We'll prove this theorem today through interactive demonstration.

## What is a Neural Network?

The concept of artificial neural networks dates back to the 1940s. The 2024 Nobel Prize in Physics was awarded to John Hopfield and Geoffrey Hinton for "foundational discoveries and inventions that enable machine learning with artificial neural networks." Hopfield's work on associative memory networks demonstrated that these systems could learn and retrieve patterns. Hinton's research with Boltzmann machines extended Hopfield networks to enable genuine machine learning rather than mere instruction-following. However, it took decades of engineering advances—in both hardware and algorithms—to realize the potential we see today.

:::{note}
Hinton also pioneered influential contributions to backpropagation and convolutional neural networks.
:::

The core idea is to design computational systems inspired by how the brain processes information. In biological systems, different neurons activate in response to specific stimuli. When you see a cat, certain neural pathways fire; when you see a dog, different patterns emerge. These neurons respond to learned patterns in the input data.

```{figure} https://assets.ibm.com/is/image/ibm/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork:16x9?fmt=png-alpha&dpr=on%2C2&wid=1536&hei=864
:name: NNarchitecture

Neural Network Architecture showing input layer, multiple hidden layers, and output layer. *Image credit: IBM*
```

### Neural Network Architecture

As shown in {numref}`NNarchitecture`, neural networks consist of three fundamental components:

**Input Layer**: This represents your data. In the CICIDS 2017 dataset we've been working with, each input node corresponds to one feature. For images, these nodes receive pixel values (grayscale or RGB channels). For text, they receive tokenized representations—which we'll explore next week when we build the phishing detector.

**Hidden Layers**: These intermediate layers are what make networks "deep." Traditional machine learning uses a shallow architecture: inputs $x$ pass through a function $f(x)$ to produce outputs $y$. Deep learning introduces hidden layers where outputs from one layer become inputs to the next. Mathematically, if the first layer applies function $f(x)$ producing $y_1$, and the second layer applies $g(y_1)$, the final output is $y_2 = g(y_1) = g(f(x))$. This composition of functions continues through all layers until reaching the classification stage.

**Output Layer**: This produces the final prediction. For binary classification (malicious vs. benign), you might use a single output node. For multi-class problems—classifying different threat types—you expand to multiple output nodes, each representing a different class.

## Why Neural Networks for Cybersecurity?

Neural networks offer several distinct advantages for cybersecurity applications:

**Complex Threat Patterns**: Cyber attacks exhibit non-linear, sophisticated patterns that challenge rule-based detection systems. Think about advanced persistent threats (APTs)—they don't follow simple signatures. Individual actions might look benign, but the sequence and timing reveal malicious intent. Neural networks excel at identifying these complex relationships across multiple features and time periods.

**Automatic Feature Discovery**: Rather than manually encoding attack signatures—the traditional approach used by antivirus companies for decades—neural networks automatically learn distinguishing characteristics of malicious activity. They detect subtle statistical anomalies that human analysts might overlook. Instead of spending weeks engineering features like "average packet size divided by connection duration when port is 443," the network discovers which combinations of raw features actually matter for detection.

**Adaptability to Evolving Threats**: The threat landscape changes daily. New exploits, malware families, and attack techniques emerge constantly. Neural networks can adapt through retraining, staying current with evolving attack vectors including zero-day exploits. This is critical in an adversarial environment where attackers are constantly trying to evade detection.

The multilayer perceptron architecture we're studying today is one of the earliest neural network designs, dating to the 1950s. While fundamental, it forms the basis for understanding more sophisticated modern architectures like convolutional neural networks, recurrent networks, and transformers.

## The Input Layer in Detail

The input layer is essentially the entry point into the network. Each neuron represents one feature, one data dimension. In cybersecurity, this can be packet size, flow rate, connection frequency, protocol type, destination port—all the features you want the network to consider.

**Data Preprocessing**: You're still going to do normalization, scaling, and encoding—the same preprocessing we covered for traditional ML. But here it's less about measuring distance (like in KNN) and more about ensuring equal dimensionality of the weights. You don't want one input with values in the thousands and another with values between 0 and 1, because that complicates how hard it is to learn those parameters. Normalizing makes the importance of features more balanced initially, letting the network learn which ones actually matter through training.

**Dimensionality**: This is the size of your input layer. This can be something small—like 78 features in our CICIDS dataset—or tens of thousands of input nodes for image data or high-dimensional text embeddings. This is a very customizable architecture. You can adjust the number of layers, the nodes in each hidden layer, the input size, the number of layers themselves. There are a lot of different parameters you can tune to tailor the model to your specific problem.

## Hidden Layers: Where the Magic Happens

The hidden layers—that's where the magic happens. Think about trying to detect whether an image contains a cat. The first layer might activate to detect an angular edge. Another node might detect a circular shape. Then you start putting them together—we have angular edges here at this distance from circles here, which looks like cat ears. You resolve more complex patterns as you go deeper through the network.

In a cybersecurity context, this hierarchical learning is powerful:

**Low-Level Feature Extraction**: The first hidden layers might identify basic patterns like unusual port numbers, abnormal packet timing intervals, high connection frequency—relatively simple combinations of your input features.

**Intermediate Pattern Recognition**: Middle layers start to form more complex signatures. These might recognize sequences that look like port scanning behavior (many connections to different ports in rapid succession), or data exfiltration patterns (large outbound transfers to unusual destinations), or reconnaissance activity (systematic probing of network services).

**High-Level Concept Formation**: By the deep layers—the ones close to the output—you're recognizing sophisticated attack patterns. We're talking about coordinated multi-stage campaigns, specific TTPs (Tactics, Techniques, and Procedures) that characterize advanced persistent threats, behavioral patterns that may not be immediately identifiable from those lower-level features alone.

The key insight: individual features like port numbers and packet counts are just normal numbers. What matters is the **combination and interaction** of those features that allows you to distinguish malicious from benign activity. The network learns these combinations automatically through training.

## The Output Layer

The output layer produces your final prediction:

**Binary Classification**: For simple malicious vs. benign decisions, you can use a single output neuron that produces a probability between 0 and 1. A value of 0.85 means "85% confident this is malicious." You'd typically set a threshold (like 0.5) where anything above is classified as malicious.

**Multi-Class Detection**: When you need to identify specific threat types, you use multiple output neurons. Maybe one neuron activates for ransomware, another for botnet activity, another for phishing, another for data exfiltration. Each outputs a probability, and you typically take the highest one as your prediction (though you could also flag multiple threats if several probabilities are high).

## How Does One Neuron Work?

Let's zoom in on a single neuron and understand exactly what it does. Once you understand one neuron, you understand the entire network—it's just this same operation repeated thousands or millions of times.

Every neuron performs a three-step process:

### Step 1: Receive Inputs and Multiply by Weights

The neuron receives inputs either from the input layer (if it's in the first hidden layer) or from neurons in the previous hidden layer. Each input gets multiplied by a learned weight value.

Think of weights as importance scores. A large positive weight means "this input is highly relevant to my decision." A small weight means "this input barely matters." Negative weights indicate inverse relationships—when the input increases, this neuron's activation should decrease.

Here's a concrete example. Imagine a neuron learning to detect port scanning behavior:
- Input 1: Connection frequency (connections per minute)
- Input 2: Number of unique destination ports accessed
- Input 3: Average connection duration

During training, this neuron might learn weights like:
- Weight 1: **+0.8** (high positive) - more connections per minute strongly suggests scanning
- Weight 2: **+0.9** (high positive) - accessing many different ports is the key port scanning signature  
- Weight 3: **-0.6** (negative) - longer connection durations are less suspicious; port scans are quick probes

The weighted sum would be: $(0.8 \times \text{freq}) + (0.9 \times \text{ports}) - (0.6 \times \text{duration})$

If you've ever heard of "weights and biases"—these are your weights. (There's also a company called Weights & Biases that makes ML tools, but conceptually, we're talking about the fundamental parameters here.)

### Step 2: Sum Everything Up

You add up all those weighted inputs, then add a bias term. The bias is important—it allows the neuron to shift its activation threshold. Think of it like the $b$ in the equation $y = mx + b$. Just like that $b$ shifts a line up or down, the bias shifts when this neuron starts to activate.

$$z = \sum_{i=1}^{n} w_i x_i + b$$

where:
- $w_i$ are the weights (learned during training)
- $x_i$ are the inputs  
- $b$ is the bias (also learned during training)

Each layer has these weights and biases for every neuron. That's what the network is learning—the optimal values for all these parameters that minimize error on the training data.

### Step 3: Apply the Activation Function

This is the key to everything. You pass that weighted sum through a non-linear activation function. Without this non-linearity, even a deep network would just collapse down to a fancy linear classifier.

Look at the math: $z = \sum w_i x_i + b$. That's a linear operation. If you stack multiple layers of just linear operations, you could mathematically compress the entire network into a single linear function. Linear models can't learn complex patterns like "this AND this BUT NOT that" or curved decision boundaries.

The non-linear activation function transforms the weighted sum at every single step, allowing the network to learn arbitrarily complex functions. This is what gives neural networks their power.

## Activation Functions

There are several different activation functions you'll encounter. Let's talk about the main ones and when you'd use each.

### ReLU (Rectified Linear Unit)

ReLU is the most popular activation function in modern neural networks. The formula is embarrassingly simple:

$$f(z) = \max(0, z)$$

If the input is negative, output zero. If the input is positive, output the input unchanged. That's it.

**Why ReLU is so popular:**

1. **Computationally Efficient**: It's just a comparison operation—extremely fast to compute. When you're processing thousands of network events per second in a real-time intrusion detection system, this computational efficiency compounds significantly. You're running this operation millions of times, so speed matters.

2. **No Saturation**: For positive values, ReLU doesn't "saturate"—the gradient doesn't vanish. Other activation functions cause gradients to become very small during training, essentially stopping learning in deep networks. ReLU maintains strong gradient flow for positive activations, allowing effective learning even with many layers.

3. **Sparse Activation**: Many neurons output zero, creating efficient representations where only relevant neurons activate for specific patterns. This is actually similar to biological neural networks—most neurons in your brain are inactive most of the time, conserving energy and reducing noise. Same principle here.

**The Dying ReLU Problem**: There is one issue to be aware of. If a neuron's weighted sum is consistently negative across all training examples, it always outputs zero. The gradient is zero, so the weights never update. The neuron "dies"—it stops learning permanently. You can mitigate this with proper weight initialization (Xavier or He initialization), appropriate learning rates, and ensuring training data diversity. If dying ReLUs become problematic, you can switch to Leaky ReLU.

You can't use ReLU at the final output layer for classification because it's unbounded at the top—you need outputs between 0 and 1 for probability interpretation. But for hidden layers, ReLU is your default choice.

:::{tip}
**Default Choice**: Start with ReLU for hidden layers. It's simple, effective, and well-understood. Only switch to other activation functions if you have a specific reason (like dying ReLUs becoming problematic, or needing outputs in a specific range).
:::

### Sigmoid

Sigmoid was very popular in earlier neural networks:

$$f(z) = \frac{1}{1 + e^{-z}}$$

It squashes any input to a value between 0 and 1, making it perfect for probability interpretation.

**Pros:**
- Output values range between 0 and 1, which naturally maps to probabilities
- Smooth and differentiable everywhere (nice mathematical properties)
- Great for the output layer in binary classification

**Cons:**
- **Vanishing Gradient Problem**: This is the killer. For very large or very small inputs, the sigmoid function flattens out. Look at the curve—at the extremes, it's nearly horizontal. That means the gradient (the slope) becomes tiny. During backpropagation, when you're computing how to update weights in early layers, you multiply gradients from all subsequent layers. If multiple layers use sigmoid, you're multiplying many small numbers together—the gradient vanishes exponentially as you go deeper. Early layers learn very slowly or stop learning entirely.

Since modern cybersecurity threats require deep networks to capture their complexity (remember: hierarchical patterns from simple anomalies to sophisticated attack campaigns), sigmoid in hidden layers cripples your model. Use it for the final output in binary classification, but avoid it in hidden layers.

### Tanh (Hyperbolic Tangent)

Tanh is sigmoid's zero-centered cousin:

$$f(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**Pros:**
- Outputs range between -1 and 1, centering data around zero
- Zero-centered outputs often lead to better training dynamics than sigmoid
- Unlike ReLU (only positive outputs), tanh allows both positive and negative activations

**Cons:**
- Still susceptible to vanishing gradients for large inputs—it asymptotes toward -1 and 1 just like sigmoid asymptotes toward 0 and 1

Tanh is better than sigmoid for hidden layers, but ReLU has largely replaced both in modern practice.

### Leaky ReLU

Leaky ReLU addresses the dying ReLU problem:

$$f(z) = \max(\alpha z, z)$$

where $\alpha$ is typically 0.01

Instead of outputting zero for negative inputs, Leaky ReLU outputs a small negative value ($0.01 \times \text{input}$). This keeps gradient flow alive even for negative activations, preventing neurons from dying.

**When to use it**: If you're experiencing significant dying ReLU problems—many neurons permanently outputting zero despite training—Leaky ReLU is worth trying. Some variants (Parametric ReLU / PReLU) learn the optimal value of $\alpha$ during training rather than fixing it at 0.01.

## Universal Approximation Theorem

Alright, now I want to show you something really cool. Can neural networks learn any function? The answer comes from the **Universal Approximation Theorem**—one of the most important theoretical results in neural network research.

**The theorem states**: Any continuous function can be approximated by a neural network with just a single hidden layer, as long as that hidden layer has enough neurons (possibly arbitrarily many) and the activation functions are non-linear (like ReLU, sigmoid, tanh).

Let me emphasize this: theoretically, a **single hidden layer is sufficient** for universal approximation capability. You don't need deep networks at all—in principle.

But here's the catch: while one layer is theoretically sufficient, you might need an astronomical number of neurons in that layer to approximate complex functions well. With multiple layers, you can approximate the same complex functions with far fewer total neurons. Deep networks are more efficient than shallow networks for representing complex functions. That's why we use depth in practice.

:::{note}
**Cybersecurity Implications**

The Universal Approximation Theorem means neural networks can theoretically learn:
- Any malware behavioral pattern, no matter how complex
- Any user activity baseline for insider threat detection
- Any multi-stage attack sequence signature
- Any relationship between network features and threat classification

This is why neural networks are so powerful for detecting sophisticated attacks that traditional rule-based systems miss.
:::

### Demonstrating Universal Approximation with ReLU

Let me show you how this actually works through interactive demonstration. We'll use ReLU functions because they're simple and clearly illustrate the principle.

#### Starting with a Single ReLU

A single ReLU creates a simple "hockey stick" function:

$$f(x) = \max(0, mx + b)$$

It's zero for negative inputs (the flat part), then linear for positive inputs with slope $m$ (the rising part). That's your basic building block. Not very interesting by itself—just a bent line.

In an interactive demo, you can adjust the slope $m$ and offset $b$. The slope controls how steep the rising part is, and $b$ shifts where the "bend" occurs horizontally. Play around with it and you'll see it's pretty limited—just a simple hockey stick shape.

But the real power comes when you combine multiple ReLUs.

#### Combining Two ReLUs

Now let's combine two ReLU functions:

$$f(x) = \max(0, m_1x + b_1) + \max(0, m_2x + b_2)$$


```{code-cell} python
:tags: [hide-input]
import altair as alt
import pandas as pd
import numpy as np

def relu(x):
    return np.maximum(0, x)

def double_relu(m1, b1, m2, b2, x):
    return relu(m1*x + b1) + relu(m2*x + b2)

# Create parameter sliders
m1_slider = alt.binding_range(min=-3, max=3, step=0.1, name='Slope 1: ')
m1_param = alt.param(name='m1', value=-2.2, bind=m1_slider)

b1_slider = alt.binding_range(min=-3, max=3, step=0.1, name='Offset 1: ')
b1_param = alt.param(name='b1', value=-2.2, bind=b1_slider)

m2_slider = alt.binding_range(min=-3, max=3, step=0.1, name='Slope 2: ')
m2_param = alt.param(name='m2', value=2.2, bind=m2_slider)

b2_slider = alt.binding_range(min=-3, max=3, step=0.1, name='Offset 2: ')
b2_param = alt.param(name='b2', value=2.2, bind=b2_slider)

# Generate base data
x_vals = np.linspace(-2, 2, 1000)

# Compute y for all parameter combinations would require pre-computing
# Instead, use expressions and transforms for dynamic computation
base_df = pd.DataFrame({'x': x_vals})

chart = alt.Chart(base_df).mark_line(
    color='#2E86AB',
    strokeWidth=3
).encode(
    x=alt.X('x:Q', title='Input (x)', scale=alt.Scale(domain=[-2, 2])),
    y=alt.Y('y:Q', title='Output (y)', scale=alt.Scale(domain=[-2, 6]))
).transform_calculate(
    # Compute relu1 and relu2 using parameter values
    relu1=f'max(0, {m1_param.name} * datum.x + {b1_param.name})',
    relu2=f'max(0, {m2_param.name} * datum.x + {b2_param.name})',
    y='datum.relu1 + datum.relu2'
).add_params(
    m1_param, b1_param, m2_param, b2_param
).properties(
    width=700,
    height=400,
    title='Combining Two ReLUs Creates Piecewise Linear Functions'
)

chart

```

With just two ReLUs, you can create a function with a "bend"—more complex than either ReLU alone. Play with the sliders and you'll see how adjusting slopes and offsets creates different shapes. Each additional ReLU adds another potential change in slope—another degree of freedom in the approximation.

Here's what's fascinating: as you add more ReLUs—5, 6, 7—you create increasingly complex piecewise-linear functions. It's like having a discrete approximation that gets finer and finer. As you scale up the number of ReLUs, it's like taking a limit—the number of these line segments goes to infinity, and you can approximate smooth curves with arbitrary precision.

#### Approximating a Complex Function

Let's make this concrete. We'll define a challenging target function with multiple frequency components:

$$f(x) = \sin(x) + \frac{1}{2}\sin(3x)$$

This is not a simple pattern—it has different frequencies mixed together. But watch how ReLUs can approximate it:


:::{note}
**Interactive Demo**
We have moved the interactive demonstration to a dedicated app for a better experience.
:::

```{raw} html
<iframe src="https://cybersec520-universal-approx.streamlit.app/Class_04_Approximation/?embed=true" width="100%" height="600px" style="border:none;"></iframe>
```

```{code-cell} python
:tags: [hide-input]

import altair as alt
#alt.data_transformers.enable("vegafusion")
import pandas as pd
import numpy as np

alt.data_transformers.disable_max_rows()

def relu(x):
    return np.maximum(0, x)

def target_function(x):
    """Complex target function with multiple frequencies"""
    return np.sin(x) + 0.5 * np.sin(3 * x)

def compute_approximation(n_segments, x):
    """Compute ReLU approximation for given number of segments"""
    y_true = target_function(x)
    
    # Create knot points
    x_knots = np.linspace(-np.pi, np.pi, n_segments + 1)
    y_knots = target_function(x_knots)
    
    # Calculate slopes between knot points
    slopes = np.diff(y_knots) / np.diff(x_knots)
    intercepts = y_knots[:-1] - slopes * x_knots[:-1]
    
    # Build piecewise linear approximation
    y_approx = slopes[0] * x + intercepts[0]
    for i in range(1, len(slopes)):
        delta_slope = slopes[i] - slopes[i-1]
        y_approx += delta_slope * relu(x - x_knots[i])
    
    error = np.abs(y_true - y_approx)
    mse = np.mean((y_true - y_approx)**2)
    
    return y_true, y_approx, error, x_knots, y_knots, mse

# Generate base data for all segment counts
x = np.linspace(-np.pi, np.pi, 1000)
segment_counts = list(range(2, 51))

# Pre-compute all approximations
all_data = []
mse_data = []

for n_seg in segment_counts:
    y_true, y_approx, error, x_knots, y_knots, mse = compute_approximation(n_seg, x)
    
    # Store MSE for loss curve
    mse_data.append({
        'n_segments': n_seg,
        'mse': mse
    })
    
    for i in range(len(x)):
        all_data.append({
            'x': x[i],
            'y_true': y_true[i],
            'y_approx': y_approx[i],
            'error': error[i],
            'n_segments': n_seg
        })

df = pd.DataFrame(all_data)
mse_df = pd.DataFrame(mse_data)

# Create knot points data
knot_data = []
for n_seg in segment_counts:
    x_knots = np.linspace(-np.pi, np.pi, n_seg + 1)
    y_knots = target_function(x_knots)
    for i in range(len(x_knots)):
        knot_data.append({
            'x': x_knots[i],
            'y': y_knots[i],
            'n_segments': n_seg
        })

knot_df = pd.DataFrame(knot_data)

# Create parameter slider for number of segments
segment_slider = alt.binding_range(min=2, max=50, step=1, name='Number of ReLU Segments (Neurons): ')
segment_param = alt.param(name='n_segments', value=5, bind=segment_slider)

# Main approximation chart
base = alt.Chart(df).transform_filter(
    alt.datum.n_segments == segment_param
)

# Target function line
target_line = base.mark_line(color='blue', strokeWidth=2.5, opacity=0.8).encode(
    x=alt.X('x:Q', title='Input (x)', scale=alt.Scale(domain=[-np.pi, np.pi])),
    y=alt.Y('y_true:Q', title='Output (y)')
)

# Approximation line
approx_line = base.mark_line(color='red', strokeWidth=2.5, strokeDash=[5, 5]).encode(
    x='x:Q',
    y=alt.Y('y_approx:Q')
)

# Knot points
knots = alt.Chart(knot_df).mark_point(
    color='green',
    size=80,
    opacity=0.7
).encode(
    x='x:Q',
    y='y:Q'
).transform_filter(
    alt.datum.n_segments == segment_param
)

# Combine approximation plot
approx_chart = (target_line + approx_line + knots).properties(
    width=400,
    height=300,
    title='ReLU Approximation of Target Function'
).add_params(segment_param)

# MSE Loss curve
loss_line = alt.Chart(mse_df).mark_line(color='#2E86AB', strokeWidth=2.5).encode(
    x=alt.X('n_segments:Q', title='Number of Segments (Neurons)', scale=alt.Scale(domain=[2, 50])),
    y=alt.Y('mse:Q', title='Mean Squared Error (MSE)')
)

# Moving marker on loss curve
loss_marker = alt.Chart(mse_df).mark_point(
    color='red',
    size=150,
    filled=True
).encode(
    x='n_segments:Q',
    y='mse:Q'
).transform_filter(
    alt.datum.n_segments == segment_param
)

# Combine loss curve with marker
loss_chart = (loss_line + loss_marker).properties(
    width=400,
    height=300,
    title='Training Loss: MSE vs Number of Segments'
)

# Combine charts side by side
chart = alt.hconcat(approx_chart, loss_chart).properties(
    title={
        "text": "Universal Approximation Theorem Demo",
        "subtitle": "How neural networks approximate complex functions using ReLU units",
        "fontSize": 16,
        "fontWeight": "bold"
    }
)

chart

```

Start with just 2-3 segments—the approximation is pretty rough. You can see the general shape, but there's significant error. Now gradually increase to 10 segments—much better. At 20 segments, you're getting quite close. At 50 segments, the approximation is nearly indistinguishable from the target function. The error (shown in the right plot) shrinks dramatically as you add more ReLU functions.

This is the Universal Approximation Theorem in action. With enough piecewise linear segments (enough ReLU neurons), you can approximate any continuous function to arbitrary precision.

:::{note}
**Important Clarification**: This demonstration simplifies how real neural networks work. Networks don't manually place ReLUs at specific knot points like we're doing here. During training, gradient descent automatically learns where to position these piecewise linear functions—what slopes and offsets to use for each ReLU. But the principle is identical: combine enough ReLUs appropriately, and you can approximate any continuous function.
:::

This mathematical foundation explains why neural networks can learn sophisticated attack patterns that traditional rule-based systems miss. Any malware behavior pattern, any user activity baseline, any attack sequence—if it can be described as a function of input features, a neural network can theoretically learn it.

Now, the theorem says this is *possible*. It doesn't tell us *how* to find those optimal weights. That's where training comes in.

---

## Training Neural Networks: Gradient Descent

We talked about loss functions before in the context of traditional ML, and neural networks use the same concept. You're still going to use a loss function to measure how wrong your predictions are. But you're going to use something called **gradient descent** to find the ideal parameters—the weights and biases that minimize that loss.

Let me show you a simplified example first, then we'll build up to the full complexity.

### A Simple Example with Quadratic Functions

Last time we did mean squared error. This time let's use mean absolute error—that's the L1 loss we mentioned earlier:

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |z_i - y_i|$$

where $z_i$ is your predicted value and $y_i$ is the true value.

It's minimizing the distance between the predicted points and true points, then dividing by the number of points you have. This is more sensitive to outliers than mean squared error. L1 is good for reducing average error; L2 (mean squared error) is more influenced by outliers because it squares the differences.

Here's what we're doing: we'll fit a simple quadratic function $ax^2 + bx + c$ to some noisy data points. We can calculate the loss for any choice of parameters $a$, $b$, $c$. As you play around with different values using sliders, you can see the loss changing. But you need to automate this, right? You can't just manually adjust parameters forever.

### Understanding the Cost Function

The cost function measures error across your entire dataset. For neural networks with millions of parameters, you can think of it as a high-dimensional surface:

$$C(w_1, w_2, w_3, ..., w_n)$$

where each $w$ is one of the weights and biases in the model, and $C$ is the total loss.

For our simple example with just three parameters ($a$, $b$, $c$), you could visualize this as a 3D surface—a bowl shape if you're lucky. For real neural networks with thousands or millions of parameters, you can't visualize it, but the math is the same.

The question is: how do you change these parameters to minimize that error? When it's just $y = mx + b$, you can solve it analytically. But we have multiple non-linear functions composed together, so we need a different approach.

### The Gradient

This is where the gradient comes in. The gradient is the direction of steepest *increase*—it points uphill on the loss surface. So if we want to minimize loss (go downhill), we move in the opposite direction—hence "gradient descent."

$$\nabla C = \left[\frac{\partial C}{\partial w_1}, \frac{\partial C}{\partial w_2}, ..., \frac{\partial C}{\partial w_n}\right]$$

Each component tells you how changing that specific parameter affects the total loss. Large positive value? Decreasing that parameter will reduce loss. Large negative value? Increasing that parameter will reduce loss. Near zero? That parameter doesn't matter much (at the current position).

Imagine you're on a mountainside in thick fog. You can't see the valley below, but you want to reach it. What do you do? Feel the ground around you, determine which direction slopes downward most steeply, take a step in that direction. Repeat until you reach flat ground (hopefully a valley, not a plateau or saddle point).

That's gradient descent. At each step, we calculate the gradient and move in the opposite direction.

### The Update Rule

The math is straightforward:

$$w_{new} = w_{old} - \epsilon \nabla C$$

where:
- $w$ represents all our parameters (weights and biases)
- $\epsilon$ is the **learning rate** (step size)
- $\nabla C$ is the gradient (vector of partial derivatives)

The learning rate controls how big a step you take. This turns out to be one of the most important hyperparameters you'll tune.

### Calculating Gradients in Practice

Let me show you how this works in code. We'll use PyTorch, which handles gradient calculation automatically. We start with some initial parameter guesses:

```{code-cell} python
import torch

# Initial parameters for quadratic function
params = torch.tensor([2.2, 2.2, 2.2], requires_grad=True)

# Calculate loss (mean absolute error for our quadratic function fit)
loss = calculate_mae(params, x_data, y_data)
print(f'Initial loss: {loss:.2f}')  # Shows how bad our initial guess is
```

Now here's the magic. To calculate gradients, we call `backward()`:

```{code-cell} python
loss.backward()  # This calculates all the gradients automatically
```

This is called the **backward pass**. When you move left to right through the network—inputs to outputs—that's the **forward pass**. When you go backwards calculating gradients, that's the **backward pass**.

You're finding out: what is the derivative of this cost function with respect to each parameter? How do I nudge each weight to lower the loss?

```python
print(params.grad)
# tensor([-1.4637, 0.0316, -0.9000])
```

These gradients tell us the direction to move. Now we update the parameters:

$$\text{new_param} = \text{old_param} - \epsilon \times \text{gradient}$$

```python
learning_rate = 0.01

with torch.no_grad():  # Don't track gradients during the update
    params -= learning_rate * params.grad
    loss = calculate_mae(params, x_data, y_data)

print(f'New loss: {loss:.2f}')  # Should be lower
```

The loss decreased! We went from 4.00 to 3.97. Small improvement, but we're moving in the right direction. Each parameter got nudged slightly based on its gradient.

### Taking Multiple Steps

Now let's take many steps—say, 15 iterations:

```python
params = torch.tensor([2.2, 2.2, 2.2], requires_grad=True)

for step in range(15):
    # Forward pass: calculate loss
    loss = calculate_mae(params, x_data, y_data)
    
    # Backward pass: calculate gradients
    loss.backward()
    
    # Update parameters
    with torch.no_grad():
        params -= learning_rate * params.grad
        params.grad.zero_()  # Reset gradients for next iteration
    
    print(f'Step {step}: loss = {loss:.2f}')
```

You'll see the loss steadily decreasing: 4.00 → 3.97 → 3.94 → 3.91... It's slowly finding the parameter values that minimize loss.

If we ran this for 40 steps with a learning rate of 0.01, you'd start to see something interesting: the loss begins to oscillate. It goes down, then up, then down, then up. What's happening? We're overshooting the minimum—taking steps that are too large. We bounce around the optimal point rather than settling into it.

### The Crucial Importance of Learning Rate

This is where the learning rate becomes absolutely critical:

**Too Large** ($\epsilon = 0.1$ or $\epsilon = 1.0$): You overshoot the minimum. The loss bounces wildly or even diverges (increases instead of decreases). Training becomes unstable—it may never converge at all. You're taking such big steps that you keep jumping over the valley.

**Too Small** ($\epsilon = 0.0001$): Painfully slow training. It will eventually reach the minimum, but it might take weeks for a real model. Each step is so tiny that you make glacial progress. On complex problems with local minima, you're more likely to get stuck.

**Just Right** ($\epsilon = 0.001$ to $0.01$): Steady, reliable progress toward the minimum. Loss decreases smoothly, training converges in reasonable time. This is what you're aiming for.

```python
# Try with a smaller learning rate
learning_rate = 0.001  # Much smaller

for step in range(50):
    loss = calculate_mae(params, x_data, y_data)
    loss.backward()
    
    with torch.no_grad():
        params -= learning_rate * params.grad
        params.grad.zero_()
    
    if step % 10 == 0:
        print(f'Step {step}: loss = {loss:.2f}')
```

With the smaller learning rate, training is more stable. The loss decreases steadily without oscillation. But it takes more steps to reach the same error level.

:::{tip}
**Learning Rate Strategies in Practice:**

- **Start with standard defaults**: 0.001 or 0.01 for most problems
- **Use learning rate schedules**: Start high for rapid initial progress, decrease over time for fine-tuning. Common pattern: multiply by 0.1 every few epochs.
- **Use adaptive optimizers**: Adam, AdamW automatically adjust learning rates per parameter based on gradient history. Still need to set an initial learning rate, but it's more forgiving.
- **Monitor your training curves**: If loss bounces wildly → decrease learning rate. If loss decreases very slowly → carefully try increasing.
:::

The solution in practice is graduated learning rates. Your learning rate starts relatively high until you start approaching convergence, then it drops to actually find that optimal minimum. Modern optimizers like Adam handle much of this automatically.

## The Cost Function for Classification

So far I've been talking about mean absolute error and mean squared error. Those work fine for regression problems (predicting continuous values). But for classification—which is what we care about in cybersecurity—we typically use **cross-entropy loss**.

For **binary classification** (malware detection: malicious vs. benign):

$$C = -\frac{1}{n}\sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

where:
- $y_i \in \{0, 1\}$ is the true label (0 = benign, 1 = malicious)
- $\hat{y}_i \in [0, 1]$ is the predicted probability

This loss function heavily penalizes confident wrong predictions. If the true label is 1 (malicious) but you predict 0.01 (very confident it's benign), the loss is huge. If you're uncertain (predict 0.5), the loss is moderate. This encourages the network to be calibrated—confident when it's correct, uncertain when it's unsure.

For **multi-class classification** (identifying specific threat types):

$$C = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{k} y_{ij} \log(\hat{y}_{ij})$$

where $k$ is the number of classes (ransomware, botnet, phishing, etc.), and $y_{ij}$ is 1 if sample $i$ belongs to class $j$, 0 otherwise.

Cross-entropy has nice mathematical properties for optimization and naturally handles probability outputs from softmax activation at the output layer.

## Batch Processing Strategies

When training neural networks, you have choices about how many examples to process before updating weights:

**Batch Gradient Descent**: Process the entire dataset, compute average loss across all samples, update weights once per epoch. Very efficient use of data—you get a stable estimate of the true gradient. But it's slow, memory-intensive, and can get stuck in local minima because the gradient is always pointing in the same direction.

**Stochastic Gradient Descent (SGD)**: Process one example at a time, update weights after each sample. Much faster updates, and the noise helps escape local minima. But very noisy training—the loss curve bounces around a lot because each individual sample gives you a different gradient direction.

**Mini-Batch Gradient Descent**: The Goldilocks solution. Process small batches (typically 32-256 examples), update weights after each batch. Good balance of stability and speed. The gradient estimate is reasonably stable (averaged over 32-256 samples), but you still get multiple updates per epoch. And the mini-batch noise helps escape poor local minima. This is what everyone uses in practice.

Typical batch sizes: 32, 64, 128, or 256. Powers of 2 work well with GPU architectures. Larger batches (512-2048) are sometimes used for very large datasets.

For real-time threat detection systems, mini-batch processing is essential. You want frequent updates as new threat data comes in, but enough stability that you're not overreacting to individual anomalies.

## Backpropagation: The Chain Rule in Action

Alright, so we understand gradient descent conceptually—calculate gradients, update weights. But here's the challenge: how do you actually calculate those gradients for millions of parameters efficiently?

Naive approach: For each weight, perturb it slightly, measure how the loss changes, compute the gradient. This requires one forward pass through the network per weight. With a million weights, that's a million forward passes per training step. Completely infeasible.

**Backpropagation** solves this brilliantly using the chain rule from calculus. It computes all gradients in a single forward pass + single backward pass through the network.

Here's the key insight: the chain rule tells you how to compute derivatives of composed functions. If $y = f(g(x))$, then:

$$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

Neural networks are exactly this—composed functions. You have layer 1 → layer 2 → layer 3 → output → loss. Each layer is a function of the previous layer's output.

### The Forward Pass

First, we compute all activations going forward through the network:

1. **Layer 1**: $z^{(1)} = W^{(1)} x + b^{(1)}$, then $a^{(1)} = \sigma(z^{(1)})$
2. **Layer 2**: $z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}$, then $a^{(2)} = \sigma(z^{(2)})$
3. Continue through all layers...
4. **Output**: Final prediction and loss $C$

We save all these intermediate values ($z$'s and $a$'s) because we'll need them for the backward pass.

### The Backward Pass

Now we work backwards from the loss, computing gradients layer by layer using the chain rule:

$$\frac{\partial C}{\partial w_{ij}^{(l)}} = \frac{\partial C}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial w_{ij}^{(l)}}$$

Let me break this down intuitively. Say you're predicting "cat" vs. "dog" and your network outputs 50% cat, 50% dog, but the true answer is "cat." You have some loss associated with being wrong.

The backward pass asks: "How much did each weight contribute to this error?"

**At the output layer**: How much did the final layer's weights contribute? We can compute this directly from the loss and the final activation.

**At the second-to-last layer**: How much did these weights contribute? They affected the final layer's inputs, which affected the output, which affected the loss. We multiply: $(\text{their effect on final layer}) \times (\text{final layer's effect on loss})$. That's the chain rule.

**At earlier layers**: Same process. Each layer's gradient depends on all subsequent layers' gradients, computed by multiplying through the chain of derivatives.

Think of it like analyzing a supply chain. If customer satisfaction drops (high loss), you need to trace back: How much did the retail stores contribute? How much did the distribution centers contribute? How much did the manufacturing plants contribute? You work backwards through the chain, assigning responsibility at each level.

The beauty is that we can reuse calculations. When computing gradients for layer 2, we already computed the gradients for layer 3. We just multiply by the chain rule term. This is why backpropagation is so efficient—$O(n)$ time where $n$ is the number of parameters, rather than $O(n^2)$ or worse for naive approaches.

:::{note}
**Why Backpropagation is Efficient**

Instead of computing gradients independently for each weight (requiring millions of forward passes), backpropagation computes them all in one backward sweep through the network, reusing calculations as it goes. This is what makes deep learning computationally feasible.
:::

You don't need to implement backpropagation yourself—PyTorch, TensorFlow, and JAX all handle this automatically via automatic differentiation. You call `.backward()` and the framework computes all the gradients. But understanding the concept helps you debug training issues and make informed architectural decisions.

## Complex Loss Landscapes

In the real world, your loss surfaces are very, very complex. You're not dealing with simple convex bowls. You have:

- **Local minima**: Valleys that aren't the deepest point globally, but look like minima locally. Gradient descent can get stuck here.
- **Saddle points**: Points where the gradient is zero, but you're not at a minimum—some directions go up, some go down. These can slow training significantly.
- **Plateaus**: Long flat regions where gradients are tiny. Training slows to a crawl.
- **Sharp valleys**: Narrow optimal regions that are hard to find and easy to overshoot.

```{iframe} https://losslandscape.com/explorer
:width: 100%
:height: 600px
:class: centered-iframe
Loss Landscape Explorer. [Link to source](https://losslandscape.com/explorer) for interactive gradient descent visualization.
```

This visualization shows how complex real loss landscapes can be. Play around with it to build intuition about why optimization is challenging.

Several techniques help navigate these challenges:

### Momentum

Don't just follow the current gradient—maintain "velocity" from previous steps:

$$v_t = \beta v_{t-1} + \nabla C$$
$$w_t = w_{t-1} - \epsilon v_t$$

Think of a ball rolling down a hill. It builds momentum and can roll over small bumps (escaping shallow local minima). The $\beta$ parameter (typically 0.9) controls how much previous velocity matters.

If you have a loss landscape with a small valley, momentum helps you not get stuck there. Just like a roller coaster—if you're going fast enough, you roll right over small dips without stopping.

### The Adam Optimizer

One of the most popular optimizers is **Adam** (Adaptive Moment Estimation). It combines momentum with adaptive learning rates per parameter. Without diving into all the math, the key idea is:

- It tracks both the first moment (mean) and second moment (variance) of gradients for each parameter
- Parameters with consistently large gradients get smaller learning rates (to avoid overshooting)
- Parameters with small gradients get larger learning rates (to make progress)
- It includes momentum to smooth out noisy gradients

This is what most people use in practice:

```python
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()  # Adam handles the complex update logic
```

AdamW is a variant of Adam with better weight decay (regularization) handling. It's become the default optimizer for most deep learning applications.

### Learning Rate Schedules

Start with a larger learning rate for quick initial progress, gradually decrease for fine-tuning:

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    train_one_epoch()
    scheduler.step()  # Decreases learning rate according to cosine schedule
```

Common schedules: step decay (decrease by factor of 10 every N epochs), exponential decay, cosine annealing. The idea is to explore quickly at first, then fine-tune as you approach the optimum.

:::{important}
**Dealing with Complex Loss Landscapes:**
- Use **mini-batch training** for balance between efficiency and randomness
- Employ **adaptive optimizers** like Adam or AdamW
- Use **momentum** to escape local minima and smooth noisy gradients
- Implement **learning rate schedules** that decrease over time
- Try **multiple random initializations** if results are unstable
:::

---

## Regularization: Preventing Overfitting

Now that we understand how neural networks train, let's talk about a critical problem: **overfitting**.

Imagine training a malware detector that achieves 99% accuracy on your training set. Fantastic! You deploy it to production... and it performs terribly, missing new malware variants constantly. What happened?

The network **memorized** the training data instead of learning general patterns. It's like a student who memorizes specific practice problems but can't solve new ones. In cybersecurity, this is disastrous—attackers constantly evolve tactics, so memorization is worthless.

### How Overfitting Happens

Without constraints, neural networks have enough capacity to memorize every training example. Consider a network with 10,000 parameters trained on 5,000 examples—there's more than enough capacity to perfectly fit every quirk and noise artifact in the training data.

Here's a real scenario: Your malware dataset includes a benign installer from 2018 that happens to use port 8443 (slightly unusual but legitimate). Through random chance, several malware samples in your training set also used port 8443. Without regularization, the network might learn an enormous weight for the "port=8443" feature, treating it as a strong malware indicator.

Result: The network perfectly classifies training data (including those specific malware samples that used port 8443), but fails on new malware (which doesn't happen to use port 8443) and produces false positives on legitimate software (which might use port 8443 for valid reasons like alternative HTTPS ports).

### L2 Regularization (Ridge Regression)

L2 regularization adds a penalty based on the **square of weights**:

$$C_{total} = C_{loss} + \lambda \sum_{i} w_i^2$$

where:
- $C_{loss}$ is the original loss (cross-entropy, MSE, etc.)
- $\lambda$ is the regularization strength (hyperparameter you tune)
- The sum is over all weights in the network

**What this does**: Penalizes large weights, encouraging the network to distribute importance across many features rather than relying heavily on a few. The network can't create those extreme specialized detectors (like our port 8443 example). It's forced to learn smoother, more general patterns.

**When to use L2**:
- You believe most input features provide some value
- You have correlated features (L2 keeps all of them with balanced weights)
- You want smooth, stable models less sensitive to input noise
- Most common choice—use this as your default

Typical $\lambda$ values: 0.0001 to 0.1, tuned via validation set performance. In PyTorch/TensorFlow, this is often called "weight decay."

### L1 Regularization (Lasso Regression)

L1 regularization penalizes the **absolute value of weights**:

$$C_{total} = C_{loss} + \lambda \sum_{i} |w_i|$$

**What this does**: Drives many weights to exactly zero, performing automatic feature selection. The network uses fewer features, ignoring irrelevant inputs. This creates sparse models.

**When to use L1**:
- You have many input features but suspect most are irrelevant
- You want an interpretable model (zero weights = ignored features)
- You want automatic feature selection
- You have high-dimensional sparse data (many features, relatively few samples)

**Cybersecurity example**: Network intrusion detection with 100 traffic features. L1 regularization might zero out 70 weights, revealing that only 30 features actually matter for threat detection. This:
1. Makes the model faster (fewer computations)
2. Improves interpretability (you can explain which features drive decisions)
3. Reduces overfitting (simpler model, fewer parameters to overfit)

### Comparing L1 vs L2

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|-----------|-----------|
| **Weight distribution** | Sparse (many exactly zero) | Dense (all small but nonzero) |
| **Feature selection** | Yes (automatic) | No (keeps all features) |
| **Model size** | Smaller (fewer active weights) | Larger (all weights active) |
| **Interpretability** | High (see which features matter) | Lower (all features contribute) |
| **Handles correlated features** | Picks one arbitrarily | Keeps all, distributes weights |
| **Best for** | High-dimensional sparse data | Dense data with many relevant features |

### Elastic Net: Combining L1 and L2

You can combine both:

$$C_{total} = C_{loss} + \lambda_1 \sum_{i} |w_i| + \lambda_2 \sum_{i} w_i^2$$

This gives you both sparsity (from L1) and stability with correlated features (from L2). Tune the ratio $\lambda_1 : \lambda_2$ based on your problem.

### Other Regularization Techniques

**Dropout** (not covered in detail today, but important): During training, randomly "drop" (set to zero) a percentage of neurons each iteration. This prevents the network from relying too heavily on any specific neurons, forcing distributed representations. At inference time, use all neurons. Extremely effective and widely used. Typical dropout rates: 0.2-0.5.

**Early Stopping**: Monitor validation loss during training. When it stops improving (or starts increasing), stop training even if training loss is still decreasing. This prevents the network from overfitting to training data. Simple but effective.

**Data Augmentation**: For images or sequences, create modified versions of training examples (rotate images, add noise, etc.). This artificially increases dataset size and helps the model generalize. Critical for computer vision, less applicable to tabular cybersecurity data.

:::{tip}
**Regularization in Practice:**

For most cybersecurity applications:
1. Start with **L2 regularization** (weight_decay=0.01 in Adam)
2. Add **dropout** (0.3-0.5) between layers if overfitting persists
3. Use **early stopping** (stop when validation loss stops improving)
4. Only use **L1** if you specifically need feature selection or have very high-dimensional sparse data

Monitor training vs validation loss curves. If training loss keeps decreasing but validation loss increases, you're overfitting—add more regularization.
:::

---

## Video Explanation

The video [What is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) is a 20-minute visualization of what we've covered in class—by 3Blue1Brown, which has excellent educational content. It goes through: what is a neural network, gradient descent, and shows the whole process with high qualiutyvisualizations 


 It's part of a whole series by 3Blue1Brown on neural networks:
- [What is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk) (19 minutes)
- [Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) (21 minutes)
- [Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U) (14 minutes)

These are genuinely some of the best educational videos on the internet. They provide exceptional visual intuition for the concepts we covered. I strongly encourage you to watch them—they'll make these concepts click in ways that equations alone cannot.

---

## Hands-On Lab: TensorFlow Playground

The objective of this lab is to build real intuition by experimenting interactively. 

### What is TensorFlow Playground?

TensorFlow Playground ([playground.tensorflow.org](https://playground.tensorflow.org)) is an interactive visualization tool that lets you see exactly what's happening inside a neural network as it trains. You can:

- **Watch learning happen in real-time**: See decision boundaries evolve as the network trains
- **Inspect individual neurons**: Observe what patterns each neuron learns to detect
- **Test architectural choices**: Compare 1 vs 2 vs 3 hidden layers instantly
- **Experiment with hyperparameters**: Change learning rates, activation functions, regularization and see immediate effects
- **Understand failure modes**: See what divergence, underfitting, and overfitting actually look like

This builds intuition that lectures alone cannot provide. It's magical to watch a network figure out how to separate complex patterns.

### Pre-Exploration: Getting Familiar

Everyone should start here to get comfortable with the interface:

1. Navigate to [playground.tensorflow.org](https://playground.tensorflow.org)
2. **Dataset**: Select "Circle" (default)
3. **Features**: Keep X₁ and X₂ selected (default)
4. **Hidden layers**: 1 layer, 3 neurons
5. **Learning rate**: 0.03 (default)
6. **Activation**: ReLU (default)
7. Click the play button (▶) to start training

Watch for about 30 seconds. You should see the decision boundary—the line separating orange from blue—evolving as the network learns. The background colors show what the network would predict for every point in the space. You're watching gradient descent in action, adjusting weights to minimize classification error.

### Lab Exploration Tasks

You'll complete four main exploration areas. There's a detailed lab document on Canvas with tables to fill out. Here's an overview:

**Exploration 1: Activation Function Comparison**
- Test ReLU, Tanh, Sigmoid, and Linear on the Circle dataset
- Compare training speed, final test loss, convergence behavior
- Fill out a comparison table with your observations
- **Key discovery**: Linear activation will completely fail—you'll see why non-linearity matters

**Exploration 2: Universal Approximation Testing**
- Try increasingly challenging patterns: Circle → XOR → Spiral
- For each pattern, find the minimal architecture that works
- Test: can 1 layer solve it? What about 2 layers? 3 layers?
- **Key discovery**: The XOR problem requires at least 2 layers (single layer can't solve it), demonstrating fundamental limitations of shallow networks

**Exploration 3: Feature Engineering Impact**
- Start with just X₁, X₂ (raw features)
- Add engineered features: X₁², X₂², X₁X₂, sin(X₁), sin(X₂)
- Compare performance: do good features let simple networks succeed?
- **Key insight**: Trade-off between feature engineering and network complexity

**Exploration 4: Learning Rate Effects**
- Test learning rates from 0.001 to 1.0 on the Spiral dataset
- Observe: slow convergence, good progress, oscillation, divergence
- **Key insight**: Build intuition for this critical hyperparameter

### Key Things to Observe

As you work through the lab, pay attention to:

**Decision Boundaries**: How does the shape relate to activation function? ReLU creates piecewise linear boundaries (angular, discontinuous). Tanh/Sigmoid create smooth curves. The shape directly reflects the activation function's properties.

**Layer Visualizations**: TensorFlow Playground shows what each hidden layer learns. Early layers detect simple patterns (regions of input space). Later layers combine these into complex decision surfaces. This is hierarchical feature learning visualized.

**Weight Magnitudes**: Connection thickness represents weight magnitude. Strong (thick) connections = important pathways. Weak (thin) connections = less relevant. Try adding L2 regularization and watch all weights shrink.

**Training Dynamics**: Some configurations converge smoothly (loss steadily decreases). Others oscillate wildly (learning rate too high). Some diverge completely (loss increases). Some get stuck (local minimum or dying ReLUs). Building visual intuition for these behaviors helps you debug real training runs.

**Overfitting in Real-Time**: Train for a very long time on one dataset. Watch the test loss. At some point, training loss keeps decreasing but test loss starts increasing. That's overfitting happening live—the network memorizing training data rather than learning general patterns.

### What I Want You to Notice

Let me show you a few specific things while you're exploring:

**L2 Regularization Effect**: If you crank L2 regularization way up (rate over 10), look at the weight vectors—they become essentially zero (10^-13). The network is prioritizing minimizing weights over actually learning. If you set regularization too high, you're neutering the network. It can't learn properly because it's too constrained.

As you increase regularization moderately, you'll see weights decrease across the board. The network is forced to distribute learning more evenly rather than relying on a few strong connections.

**Decision Boundary Evolution**: Look at the visualization showing individual neuron outputs. From just X and Y values, you get these initial decision boundaries. As you combine different neurons' outputs, you get more complex boundaries, then even more complex ones, until you reach the final classification boundary. You can see the progression from simple features to complex features as you move through layers.

**Activation Function Impact**: As you change activation functions, watch how decision boundaries change shape. It's based directly on the activation function shape. ReLU's discontinuous nature creates discontinuous boundaries. As you add more neurons, those boundaries become less obviously discontinuous—finer approximation.

**Learning Rate Dynamics**: You can actually adjust learning rate while training. Start at 0.03. If it's oscillating and not converging, drop it to 0.01. Once it plateaus, drop to 0.001 for fine-tuning. You'll see training stabilize and continue improving. This demonstrates graduated learning rates in action.

**Adding Complexity**: If you make the network deeper (add layers) or wider (add neurons per layer), you can separate more complex patterns. For the Spiral dataset, you'll probably need deeper networks and might benefit from sinusoidal features (sin(X₁), sin(X₂)) because the data has that periodic structure.

:::{tip}
**Lab Tips:**
- Don't just click randomly—form hypotheses about what will happen, then test them
- Pay attention to how decision boundaries evolve during training
- Watch the training loss curve—does it converge smoothly or oscillate?
- Try to find the minimal architecture needed to solve each problem
- Notice when regularization helps versus when it hurts
- Take notes on surprising behaviors—those are the best learning moments
:::

### Common Issues You Might Encounter

- **The XOR pattern is actually quite easy** for a 2-layer network. Even with just a few neurons, it separates cleanly. But a single layer absolutely cannot solve it—this is a fundamental limitation.
- **The Spiral pattern is genuinely hard**. You'll probably need sinusoidal features (sin(X), sin(Y)) to get good separation, because the data has that circular/periodic structure.
- **If training seems stuck**, try adjusting the learning rate. Too high → oscillation. Too low → painfully slow progress.
- **Don't worry if some patterns don't reach perfect separation**. That's part of the learning. Sometimes the architecture just isn't capable, or you need better features.

### Success Metrics

Someone asked: "What defines success?" I'd say if you get test loss below 0.05 (above 95% accuracy), that's successful. But define your own threshold.

Just note: in a real cybersecurity context, 85% accuracy probably never sees production. Can you imagine a 15% false positive rate? If you have 10,000 events a day, that's 1,500 false alerts. So it really depends on context. For spam filtering, higher error rates are acceptable. For critical security decisions, you need much higher accuracy.

### Deliverable

At the end of the lab session (or finish at home), submit a brief reflection (1-2 paragraphs) on Canvas describing:
- The most interesting thing you discovered
- One pattern that surprised you
- How this hands-on exploration changed or reinforced your understanding of neural networks

The detailed tables and observations should be filled out in the lab document and submitted as well.

---

## Wrapping Up

### Assignments

**Due This Week:**
- **TensorFlow Playground exploration lab** (due Tuesday, February 10th)
  - Complete all four exploration tasks
  - Fill out observation tables
  - Write brief reflection
- **Lab 2** (Extended to Sunday, February 8th)

**Next Week:**
- We'll cover **sequential models** (RNNs, LSTMs, transformers)
- We'll build a **phishing detection model** using natural language processing
- Walk through it in a Colab notebook in class
- You'll finish it as homework assignment


### Looking Ahead

Next week we're diving into natural language processing for cybersecurity. We'll build that phishing detector together, and we'll finally get to sequential models—RNNs, LSTMs, and the transformer architecture that powers modern LLMs. Those are the foundations for everything we'll do with text and time-series data this semester.

The foundations we've built today—understanding neurons, layers, activation functions, forward pass, backward pass, gradient descent, regularization—these support everything else we do with neural networks. Make sure you understand these core concepts. If anything is unclear, now's the time to ask or come to office hours.

:::{important}
**Key Takeaways from Today:**
- Neural networks are brain-inspired computing systems with interconnected layers that learn hierarchical representations
- **Non-linear activation functions** (especially ReLU) are what make deep learning powerful—without them, networks collapse to linear models
- **Universal Approximation Theorem** shows NNs can theoretically learn any continuous function given sufficient capacity
- **Gradient descent with backpropagation** is how we train these networks—efficiently computing gradients for millions of parameters
- **Regularization** (L1/L2, dropout, early stopping) prevents overfitting by constraining model complexity
- **Hyperparameters** like learning rate, architecture, activation functions, and regularization strength dramatically affect performance
:::

Any questions before we wrap up? Alright, thanks everyone. See you next week!

---

## Additional Resources

**Videos:**
- [3Blue1Brown Neural Network Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - Best visual introduction available
- Andrej Karpathy's YouTube lectures on neural networks - Practical, implementation-focused

**Books:**
- [*Deep Learning* by Goodfellow, Bengio, Courville](https://www.deeplearningbook.org/) - Comprehensive textbook, available free online
- Excellent intuitive explanations without excessive math in early chapters

**Interactive Tools:**
- [TensorFlow Playground](https://playground.tensorflow.org) - Essential for building intuition
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) - We'll use this when we cover CNNs
- [Loss Landscape Explorer](https://losslandscape.com/explorer) - Visualize optimization challenges

**Frameworks & Documentation:**
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - We'll use PyTorch for implementations
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials) - Alternative framework
- [Fast.ai](https://www.fast.ai/) - High-level library, great for rapid prototyping. Jeremy Howard's lectures are excellent and practical.

**Research Papers** *(for deeper understanding)*:
- Cybenko (1989): "Approximation by superpositions of a sigmoidal function" - Original universal approximation theorem
- Hornik et al. (1989): "Multilayer feedforward networks are universal approximators"
- He et al. (2015): "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" - Why ReLU works, proper initialization

---

*All diagrams, code examples, and interactive demonstrations referenced in these notes are available in the Class 4 Colab notebook and slides on Canvas. The TensorFlow Playground link with pre-configured settings is available on Canvas.*

*Image credits: IBM, Stanford University, Paperspace, University of Maryland*
