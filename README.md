# EN705651-Summer24-LLM-Project

## Authors: Alec Shamula, Oluwatobi Ajide, Akhil Gupta, Solomon Gruse   

The development of Transformer-based models like GPT has significantly advanced the field of natural language processing, particularly in the domain of text generation. Building on this foundation, we introduce a series of novel GPT-based models designed to enhance text generation by incorporating advanced techniques for uncertainty quantification, prediction flexibility, and overconfidence mitigation. 

We present Smooth-GPT, a model that utilizes label smoothing to temper overconfident predictions, promoting more generalized and balanced outputs. Bayes-GPT extends the traditional GPT architecture by integrating a Bayesian linear layer, enabling the model to capture and express uncertainty in its predictions. Sig-GPT replaces the conventional softmax activation with a sigmoid layer, allowing for multi-label predictions and greater flexibility in generating diverse outputs. Finally, Bayes-Sig-GPT combines the strengths of both Bayesian inference and sigmoid activation, offering a novel approach to balancing uncertainty and prediction diversity.

These innovations represent a significant step forward in the development of more robust and adaptable language models, capable of generating high-quality, contextually appropriate text across a variety of applications. This repository and attached paper detail the design and implementation of these models, demonstrating their potential to push the boundaries of current text generation capabilities.


## Prerequisites

* Please ensure you have at least Python 3.10 installed.   
* Install all libraries in `requirements.txt`
* CUDA-Enabled GPU (optional but very recommended)

  ```
      conda create --name myenv
      conda activate myenv
      cd /path/to/your/project
      pip/conda install -r requirements.txt
  ```


## Navigating Repository
The bulk of this codebase belongs inside `nanoGPT/`: an altered version of Andrej Karpathy's [GPT2 Tutorial Repo](https://github.com/karpathy/nanoGPT). The original README is copied into this subdirectory which provides guidelines for running the baseline model. 
We designed our test models to mimic the existing framework, and thus training our test models merely involves a file name swap usually. Within `nanoGPT`, you will also find `prepare.py` scripts for available datasets. Our results hone in on wikitext for this repository. Finally within `training_logs/`, you will find printed log data of our training experiments.

`augmentation/` is an extension developed to support the training data needs for our theoretical sigmoid models.

A detailed paper outlining our approach and results is additionally included at this level. 

## Run Instructions
Unfortunately our weights are too large in size too include in this repository, so running will entail training a model from scratch. If you wish to run a model, adhere to the following steps:

1) Choose a dataset to work with and navigate to `nanoGPT/data/`. Run the respective `prepare.py` file within the dataset folder you want to train on. This will generate significantly size .bin files that should be kept locally.
2) Within `nanoGPT/config/`, find the appropriate .py file and edit training/model configuration parameters to your satisfaction. You will also need to indicate the proper dataset directory here.
3) Once you have configuration parameters set and a model of interest, run the baseline train python command with edits on your naming. For example, if I wanted to run the sigmoid model...
```
cd nanoGPT
torchrun train_sigmoid.py config/train_sigmoid_gpt2_wikitext_100m.py
```

*Note "torchrun" should only be used if `cuda` is set to **True**. If you are training a model that is greater than 20M parameters, I highly recommend using CUDA-enabled GPUs. If using CPU, use "python" instead.

## WandB Training
![image](https://github.com/user-attachments/assets/4ea9f55c-9096-43c2-b672-d8312f445c07)


![image](https://github.com/user-attachments/assets/f45390d5-2978-4b6b-8e48-92c7e13d4b3e)





## Recommended Resources
* See our paper to dive deeper into the theory behind model implementations, results, and analysis.
* GPU compute is necessary to train significant models in a reasonable amount of time. We recommend [Lightning AI](https://lightning.ai/) for cloud based development and GPUs.
* [Karpathy GPT2 YouTube Tutorial](https://www.youtube.com/watch?v=l8pRSuU81PU)
* [Original Transformers Paper](https://arxiv.org/abs/1706.03762)
* [OpenAI GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [OpenAI GPT-3 Paper](https://arxiv.org/abs/2005.14165)
  



