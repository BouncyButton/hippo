"""Native 90-update learning-rate schedule for the HFH A100 workers."""
SCHEDULE = dict(original_prefix_updates=20,plateau_lr=.07,decay_start=60,horizon=90,power=.9)
