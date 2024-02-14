# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-enwik8-char-al-rfou'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like

dataset = 'enwik8_char'
gradient_accumulation_steps = 2
batch_size = 32
block_size = 512 # context of up to 512 previous characters

n_layer = 64
n_head = 4
n_embd = 512
dropout = 0.0

learning_rate = 6e-3 # with baby networks can afford to go a bit higher
max_iters = 20_000
lr_decay_iters = 20_000 # make equal to max_iters usually
min_lr = 6e-5 # learning_rate / 10 usually
weight_decay = 1e-1

warmup_iters = 2500
