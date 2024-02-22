# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-enwik8-char-bloom-3'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'enwik8-char-bloom-3'
wandb_run_name = 'mini-gpt'

dataset = 'enwik8_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 512 # context of up to 512 previous characters

n_layer: int = 256
n_head: int = 2
n_embd: int = 512
attention_proj_size: int = 32
causal_self_attn_size: int = 1
mlp_intermediate_size: int = 4 * 512
dropout: float = 0.01

bias = False

learning_rate = 6e-4 # with baby networks can afford to go a bit higher
max_iters = 10_000
lr_decay_iters = 10_000 # make equal to max_iters usually
min_lr = 6e-5 # learning_rate / 10 usually
weight_decay = 1e-1

warmup_iters = 1000