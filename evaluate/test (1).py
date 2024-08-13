import os
import torch
import tiktoken
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from model import GPTConfig, GPT
from diversity_evaluator import TextQualityEvaluator  # Assuming your evaluator is in this file

batch_size = 12
block_size = 1024
dataset = 'wikitext'
data_dir = os.path.join('data', dataset)
# -----------------------------------------------------------------------------
init_from = 'resume'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'  # ignored if init_from is not 'resume'
num_samples = 10  # number of samples to draw per batch
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# Model loading
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'sigmoid_8k_iter_ckpt (1).pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# Load GPT-2 encodings
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={""})
decode = lambda l: enc.decode(l)

# Preload data once to avoid repeated I/O operations
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Create an evaluator instance
evaluator = TextQualityEvaluator(gram=4)
loops = 100

# Accumulators for average scores
accumulated_scores = {'self_bleu': 0, 'distinct_n': 0}

# Evaluate model on train and val data
for i in range(loops):
    input_ids, _ = get_batch('train')
    generated_texts = []
    print("Loop:", i)
    with torch.no_grad(), torch.cuda.amp.autocast(device_type == 'cuda'):
        for j in range(num_samples):
            y = model.generate(input_ids[j:j+1], max_new_tokens, temperature=temperature, top_k=top_k)
            generated_text = decode(y[0].tolist())
            generated_texts.append(generated_text)

    # Evaluate the generated texts using TextQualityEvaluator
    with ThreadPoolExecutor() as executor:
        evaluation_scores_list = list(executor.map(evaluator.evaluate, [generated_texts]))

    # Update accumulators
    accumulated_scores['self_bleu'] += evaluation_scores_list[0].get('self_bleu', 0)
    accumulated_scores['distinct_n'] += evaluation_scores_list[0].get('distinct_n', 0)

# Calculate average scores
average_scores = {metric: score / loops for metric, score in accumulated_scores.items()}

print("Average Evaluation Scores:")
print(average_scores)
