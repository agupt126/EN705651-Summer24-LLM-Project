# Data Augmentation

## Overview

Training examples are adjusted such that there are multiple correct token generations. Consider the following example:

> "The boy went to the..."

The original training set may follow this sentence with the word "park", however, one could reasonably come up with many other *valid* possible next tokens. For example, the following sentences are all semantically correct:

> "The boy went to the *store*"  
> "The boy went to the *entrance*"

## Method

In the standard LLM training procedure, the target vector looks something like this:

`[0, 0, 0, 1, 0, ..., 0, 0, 0]`

In other words, there is a single correct token (in the whole vocabulary) that follows a previous context. Our method adapts this format to allow for multiple correct tokens:

`[0, 0, 0, 1, 0, ..., 1, 0, 0]`

Alternate valid tokens are generated using a pretrained GPT-2 model (small). For each example in a batch, a random index is generated. This is the index of the token to be "augmented" (generate additional tokens for). The validity of each new token substitute is determined based on a perplexity threshold. This is a tunable hyperparameter that can be adjusted for each run.