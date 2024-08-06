import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class DataAugmenter:
  def __init__(self, k=50, perplexity_threshold=100, device='cuda'):
    print('Initializing DataAugmenter!')

    # Load model and put it on device
    model_name = 'gpt2'
    self.device = device
    self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
    print('Loaded GPT-2 model!')

    # Load tokenizer and save relevant information
    self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.vocabulary_size = len(self.tokenizer)
    print('Loaded GPT-2 tokenizer!')

    # Save hyperparameters
    self.k = k
    self.perplexity_threshold = perplexity_threshold

  def _generate_substitutes(self, input_ids_batch, target_indices):
    with torch.no_grad():
        outputs = self.model(input_ids_batch)
        logits = outputs.logits

    # Extract logits for the target tokens
    batch_size = input_ids_batch.size(0)
    target_logits = logits[torch.arange(batch_size), target_indices, :]

    # Get top-k substitutes
    top_k_values, top_k_indices = torch.topk(target_logits, self.k, dim=-1)
    return top_k_indices

  def _calculate_perplexity(self, input_ids_batch):
    with torch.no_grad():
        outputs = self.model(input_ids_batch, labels=input_ids_batch)
        logits = outputs.logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids_batch[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_logits.size(0), shift_logits.size(1))
        loss = loss.mean(dim=1)

    return torch.exp(loss)

  def _filter_valid_substitutes(self, input_ids_batch, substitutes_batch, target_indices):
    batch_size, seq_len = input_ids_batch.size()
    valid_mask = torch.zeros_like(substitutes_batch, dtype=torch.bool)

    for i in range(substitutes_batch.size(1)):
        new_input_ids_batch = input_ids_batch.clone()
        new_input_ids_batch[torch.arange(batch_size), target_indices] = substitutes_batch[:, i]
        perplexity = self._calculate_perplexity(new_input_ids_batch)
        valid_mask[:, i] = perplexity < self.perplexity_threshold

    return valid_mask

  def _update_boolean_vector_batch(self, substitutes_batch, valid_mask):
    boolean_vector_batch = torch.zeros((substitutes_batch.size(0), self.vocabulary_size), dtype=torch.float16)

    valid_indices = substitutes_batch[valid_mask]
    batch_indices = torch.arange(substitutes_batch.size(0)).to(self.device).repeat_interleave(valid_mask.sum(dim=-1))

    boolean_vector_batch[batch_indices, valid_indices] = 1
    return boolean_vector_batch
  
  def _create_boolean_vector_without_filtering(self, X, ratio=0.5, k=50):
    with torch.no_grad():
      outputs = self.model(X)
      logits = outputs.logits
    batch_size = logits.shape[0]
    block_size = logits.shape[1]
    vocab_size = logits.shape[2]
    n = round(block_size * ratio)
    swap_indices = torch.stack([torch.randperm(block_size)[:n] for _ in range(batch_size)])  # batch_size x n_swaps (n)
    target_logits = logits[torch.arange(batch_size).repeat_interleave(n), swap_indices.flatten(), :].view(batch_size, n, vocab_size)
    _, top_k_indices = torch.topk(target_logits, k, dim=-1)
    boolean_vector = torch.zeros(batch_size, block_size, vocab_size)
    a = torch.arange(batch_size).repeat_interleave(n * k)
    b = swap_indices.repeat(1, k).flatten()
    c = top_k_indices.flatten()
    boolean_vector[a, b, c] = 1
    return boolean_vector

  def augment(self, X, target_indices=None, do_filter=True):
    # batch_size = X.shape[0]
    # block_size = X.shape[1]
    # boolean_vector = torch.zeros((batch_size, block_size, self.vocabulary_size), dtype=torch.float16)
    # i = torch.arange(batch_size).repeat_interleave(block_size)
    # j = torch.arange(block_size).repeat(batch_size)
    # k = X[:, :, None].flatten()
    # boolean_vector[i, j, k] = 1

    # Use user-supplied target_indices if available, else generate random indices
    # if target_indices is None:
    #   target_indices = torch.randint(low=0, high=X.shape[1], size=(X.shape[0],))

    # substitutes = self._generate_substitutes(X, target_indices)

    # if do_filter:
    #     valid_mask = self._filter_valid_substitutes(X, substitutes, target_indices)
    # else:
    #     valid_mask = torch.ones((X.size(0), self.k), dtype=bool).to(self.device)

    # i = torch.arange(batch_size)
    # boolean_vector[i, target_indices, :] = self._update_boolean_vector_batch(substitutes, valid_mask)

    boolean_vector = self._create_boolean_vector_without_filtering(X, ratio=0.5, k=50)
    target_indices = None  # Not currently returning this in create_boolean_vector_without_filtering()

    # del substitutes
    # del valid_mask
    torch.cuda.empty_cache()

    return boolean_vector, target_indices

  def generate_new_sentences(self, input_ids_batch, target_indices, boolean_vector_batch):
    vocabulary = list(self.tokenizer.get_vocab().keys())
    new_sentences_batch = []

    sentences = self.tokenizer.batch_decode(input_ids_batch, skip_special_tokens=True)
    for i, sentence in enumerate(sentences):
      new_sentences = []
      for idx, is_valid in enumerate(boolean_vector_batch[i][target_indices[i]]):
        if is_valid:
          new_input_ids = input_ids_batch[i].clone()
          new_input_ids[target_indices[i]] = idx
          new_sentence = self.tokenizer.decode(new_input_ids, skip_special_tokens=True)
          new_sentences.append(new_sentence)
      new_sentences_batch.append(new_sentences)
    
    return new_sentences_batch