class PTConfig:
    batch_size = 4
    max_length = 512
    n_samples = 10  # for train/test
    model = "sshleifer/tiny-gpt2"  # sshleifer/tiny-gpt2, gpt2, EleutherAI/gpt-neo-125M (not working)
    num_epochs = 3
    weight_decay = 0.01
    learning_rate = 0.005
    lr_scheduler_type = "linear"
    num_warmup_steps = 0
    max_train_steps = num_epochs
    n_prompt_tokens = 20
    init_from_vocab = True
    random_range = 0.5


class IDPTConfig:
    batch_size = 4
    max_length = 512
    n_samples = 10  # for train/test
    model = "sshleifer/tiny-gpt2"  # sshleifer/tiny-gpt2, gpt2, EleutherAI/gpt-neo-125M (not working)
    num_epochs = 3
    weight_decay = 0.01
    learning_rate = 0.005
    lr_scheduler_type = "linear"
    num_warmup_steps = 0
    max_train_steps = num_epochs
    n_prompt_tokens = 20
    init_from_vocab = True
    random_range = 0.5
