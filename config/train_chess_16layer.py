out_dir = "out-chess-16layer"
eval_interval = 2000
eval_iters = 100
log_interval = 50

always_save_checkpoint = True

wandb_log = True
wandb_project = "chess-gpt-16layer"
wandb_run_name = "16layer_lichess_50M"
wandb_run_id = "chess16layer-v1"

dataset = "lichess_hf_dataset"
gradient_accumulation_steps = 1
batch_size = 100
block_size = 1023

n_layer = 16
n_head = 8
n_embd = 512
dropout = 0.0
bias = True

learning_rate = 3e-4
max_iters = 600000
lr_decay_iters = 600000
min_lr = 3e-5
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
warmup_iters = 2000

compile = True
dtype = "bfloat16"
