# EN705651-Summer24-LLM-Project

Authors: Alec Shamula, Oluwatobi Ajide, Akhil Gupta, Solomon Gruse   
Summary: 
In contemporary state-of-the-art large language models (LLMs), training typically involves a SoftMax activation layer to predict the most probable token for sequence completion. We propose to assess the viability of employing a Sigmoid activation layer, which is commonly utilized in traditional multi-label classification tasks, to train an LLM instead. Additionally, we will explore the use of Bayesian layers. Bayesian layers learn a distribution over the weights, allowing for multiple samples from this distribution to obtain the next token prediction. The probability distributions in Bayesian Neural Networks (BNNs) enable them to learn from the data and understand the confidence in what they have learned. This makes BNNs more robust and flexible, particularly when there is a limited volume of data or when the data contains noise.



## Project MVP Update: 07/30/2024

### 1.0 General Updates
In order to train the models, we will be using LightningAI as well as the APL Abyss HPC. 

Milestones:
- Able to run Baseline model using LightningAI
- Able to locally run Sigmoid model (not including data augmenter)
- Able to locally run Bayesian model
- Able to generate substitute candidate tokens and convert to vector encoding

### 2.0 Training Data

#### 2.1 Dataset
We are using Wikitext which contains approximatley 103M tokens for our training set.
https://huggingface.co/datasets/Salesforce/wikitext

#### 2.2 Augmentation

Training examples are adjusted such that there are multiple correct token generations. Consider the following example:

> "The boy went to the..."

The original training set may follow this sentence with the word "park", however, one could reasonably come up with many other *valid* possible next tokens. For example, the following sentences are all semantically correct:

> "The boy went to the *park*"  
> "The boy went to the *store*"  
> "The boy went to the *entrance*"

In the standard LLM training procedure, the target vector looks something like this:

`[0, 0, 0, 1, 0, ..., 0, 0, 0]`

In other words, there is a single correct token (in the whole vocabulary) that follows a previous context. Our method adapts this format to allow for multiple correct tokens:

`[0, 0, 0, 1, 0, ..., 1, 0, 0]`

Alternate valid tokens are generated using a pretrained GPT-2 model (small). For each example in a batch, a random index is generated. This is the index of the token to be "augmented" (generate additional tokens for). $k$ candidate tokens are generated using the top-k sampling strategy. By default, we set $k = 50$. The validity of each new token substitute is determined based on a perplexity threshold. By default, this value is set to $100$. Both $k$ and the perplexity threshold are tunable hyperparameters that can be adjusted for each run.

The implementation can be found in the `augmentation` directory. The `DataAugmenter` class within `data_augmentation.py` contains code that loads a pretrained GPT-2 model from Huggingface's `transformers` library. In addition, there is an `augment()` function that is responsible for generating candidate token substitutes, filtering these substitutes (based on perplexity), and generating a boolean output tensor. The following code snippet demonstrates basic functionality:

```
from data_augmentation import DataAugmenter

augmenter = DataAugmenter()
sentences = ["The boy went to the park.", "She loves to read books."]
inputs = augmenter.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
input_ids_batch = inputs['input_ids'].to('cuda')
output, target_indices = augmenter.augment(input_ids_batch, target_indices=[5,4], do_filter=True)
```

```
[['The boy went to the area.',
  'The boy went to the bench.',
  'The boy went to the entrance.'],
 ['She loves to read too.']]
```

### 3.0 GPT-2 Training/Testing

#### 3.1 Baseline
We will use a baseline model as a comparison metric for the proposed variations to the model architecture. The baseline model used is GPT-2 with approximatley 124M parameters.

#### 3.2 Sigmoid
The proposed architecture will replace the Softmax layer with a Sigmoid layer. This is a common approach when dealing with multilabel classification problems. This highlights an important idea that in language, there is often more than one feasible next token. Additionally, we have replaced the Cross-Entropy Loss function with the Binary Cross-Entropy loss function. This is more suitable when dealing with a Sigmoid layer and allows us to define an individual probability distribution over each token. The updated model can be found in the model_sigmoid.py file. Changes were required in the forward function as well as the generate function.

#### 3.3 Bayesian
An alternative architecture that we will explore replaces the last linear layer with a Bayesian layer. As a result, we learn a distribution that we can sample from for the weights leading into the activation function and sample from this distribution multiple times. This architecture will use the SoftMax activation function as well as the Cross-Entropy Loss function. This updated model can be found in the model_BN.py file. Using the Pyro library, a new class for a Bayesian layer is defined. Additionaly, updates were made to the forward function as well as the generate function.

### 4.0 Next Steps
- Integrate data augmentor into batch generation function to train model
- Train full size models using LightningAI/APL Abyss HPC
- Compare alternate frameworks using metrics such as Perplexity
