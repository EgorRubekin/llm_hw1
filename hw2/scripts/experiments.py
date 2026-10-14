FA2 = "flash_attention_2"
BS64 = dict(per_device_train_batch_size=64, gradient_accumulation_steps=1)

EXPERIMENTS = {
    "baseline": dict(seconds=300),

    "bs16_acc4": dict(seconds=90, per_device_train_batch_size=16, gradient_accumulation_steps=4),

    "checkpointing_bs64": dict(seconds=90, gradient_checkpointing=True, **BS64),

    "offloading": dict(seconds=90, activation_offloading=True),

    "liger": dict(seconds=90, use_liger_kernel=True),
    "liger_bs64": dict(seconds=90, use_liger_kernel=True, **BS64),
    "compile": dict(seconds=90, torch_compile=True),
    "fa2": dict(seconds=90, attention=FA2),

    "fa2_padding_free": dict(seconds=90, attention=FA2, padding_free=True),
    "fa2_packing": dict(seconds=90, attention=FA2, packing=True),
    "fa2_packing_bs6": dict(seconds=90, attention=FA2, packing=True, per_device_train_batch_size=6),

    "fa2_padding_free_liger": dict(seconds=90, attention=FA2, padding_free=True, use_liger_kernel=True),
    "fa2_padding_free_liger_bs64": dict(seconds=90, attention=FA2, padding_free=True, use_liger_kernel=True, **BS64),
    "fa2_packing_liger_compile": dict(seconds=90, attention=FA2, packing=True, use_liger_kernel=True, torch_compile=True),
}
