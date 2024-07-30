# EN705651-Summer24-LLM-Project

Authors: Alec Shamula, Oluwatobi Ajide, Akhil Gupta, Solomon Gruse   
Summary: 
In contemporary state-of-the-art large language models (LLMs), training typically involves a SoftMax activation layer to predict the most probable token for sequence completion. We propose to assess the viability of employing a Sigmoid activation layer, which is commonly utilized in traditional multi-label classification tasks, to train an LLM instead. Additionally, we will explore the use of Bayesian layers. Bayesian layers learn a distribution over the weights, allowing for multiple samples from this distribution to obtain the next token prediction. The probability distributions in Bayesian Neural Networks (BNNs) enable them to learn from the data and understand the confidence in what they have learned. This makes BNNs more robust and flexible, particularly when there is a limited volume of data or when the data contains noise.



## Project MVP Update: 07/30/2024

### 1.0 General Updates

### 2.0 Training Data

#### 2.1 Dataset


#### 2.2 Augmentation

### 3.0 GPT-2 Training/Testing


#### 3.1 Baseline
We will use a baseline model as a comparison metric for the proposed variations to the model architecture. The baseline model used is GPT2-2 with approximatley 124M parameters.

#### 3.2 Sigmoid
The proposed architecture will replace the Softmax layer with a Sigmoid layer. This is a common approach when dealing with multilabel classification problems. This highlights an important idea that in language, there is often more than one feasible next token. Additionally, we have replaced the Cross-Entropy Loss function with the Binary Cross-Entropy loss function. This is more suitable when dealing with a Sidmoid layer and allows us to define an individual probability distribution over each token. 

#### 3.3 Bayesian
An alternative architecture that we will explore replaces the last linear layer with a Bayesian layer. As a result, we learn a distribution that we can sample from for the weights leading into the activation function and sample from this distribution multiple times. This architecture will use the Softmax activation function as well as the Cross-Entropy Loss function.

### 4.0 Next Steps
