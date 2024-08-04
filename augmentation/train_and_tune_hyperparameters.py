import os
import time
import math
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model_sigmoid import GPTConfig, GPT
from augmentation.data_augmentation import DataAugmenter
import sys
sys.path.append('..')

def train_and_tune_hyperparameters():
    # REQUIRED TO RUN ON WINDOWS
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    # -----------------------------------------------------------------------------
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    out_dir = 'out'
    eval_interval = 2000
    log_interval = 1
    eval_iters = 200
    eval_only = False # if True, script exits right after the first eval
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
    # wandb logging
    wandb_log = False # disabled by default
    wandb_project = 'owt'
    wandb_run_name = 'gpt2' # 'run' + str(time.time())
    # data
    dataset = 'shakespeare'
    gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
    batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 1024
    # model
    n_layer = 6
    n_head = 6
    n_embd = 216
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate = 6e-4 # max learning rate
    max_iters = 600000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
    min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = False # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Initialize data augmentation
    # Define the range of hyperparameters for tuning
    perplexity_threshold_values = [100, 120, 130]
    top_k_values = [50, 60, 70]

    # Placeholder for best hyperparameters and best validation loss
    best_hyperparams = None
    best_val_loss = float('inf')

    # poor man's data loader
    data_dir = os.path.join('data', dataset)
    def get_batch(split, augmenter):
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        y, _ = augmenter.augment(x)
        y = y.to(device)
        return x, y

    # Function to evaluate a specific set of hyperparameters
    def evaluate_hyperparams(perplexity_threshold, top_k):
        augmenter = DataAugmenter(device=device_type, perplexity_threshold=perplexity_threshold, k=top_k)

        meta_path = os.path.join(data_dir, 'meta.pkl')
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        # model init
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                        bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
        if init_from == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if meta_vocab_size is None:
                print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
            model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50257
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
        else:
            # init a model from pre-trained weights
            print(f"Initializing from {init_from}")
            model = GPT.from_pretrained(init_from, override_args=model_args)
        model.to(device)
        optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
        if dtype == 'float16':
            scaler = torch.cuda.amp.GradScaler(enabled=True)

        iter_num = 0
        best_val_loss = float('inf')

        # Training loop (simplified for hyperparameter tuning)
        for _ in range(eval_iters):
            model.train()
            X, Y = get_batch('train', augmenter)
            with ctx:
                logits, loss = model(X, Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            val_loss = loss.item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        return best_val_loss

    # Hyperparameter tuning loop
    for perplexity_threshold in perplexity_threshold_values:
        for top_k in top_k_values:
            print(f"Evaluating perplexity_threshold={perplexity_threshold}, top_k={top_k}")
            val_loss = evaluate_hyperparams(perplexity_threshold, top_k)
            print(f"Validation loss: {val_loss}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_hyperparams = (perplexity_threshold, top_k)

    print(f"Best hyperparameters: {best_hyperparams} with validation loss: {best_val_loss}")

    # After tuning, proceed with the best hyperparameters
    best_perplexity_threshold, best_top_k = best_hyperparams
    augmenter = DataAugmenter(device=device_type, do_filter=True, perplexity_threshold=best_perplexity_threshold, top_k=best_top_k)

    return augmenter

