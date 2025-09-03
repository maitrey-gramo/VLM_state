import os

import torch
from ipdb import set_trace
from vima.policy import VIMAPolicy


def create_policy_from_ckpt(ckpt_path, device):
    assert os.path.exists(ckpt_path), "Checkpoint path does not exist"
    ckpt = torch.load(ckpt_path, map_location=device)
    set_trace()
    policy_instance = VIMAPolicy(**ckpt["cfg"])
    policy_instance.load_state_dict(
        {k.replace("policy.", ""): v for k, v in ckpt["state_dict"].items()},
        strict=True,
    )
    policy_instance.eval()
    return policy_instance


# dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'hparams_name', 'hyper_parameters'])

# vima 200M
# ipdb> ckpt.keys()
# odict_keys(['state_dict', 'cfg'])
# ipdb> ckpt['cfg'].keys()
# dict_keys(['embed_dim', 'xf_n_layers', 'sattn_n_heads', 'xattn_n_heads'])
# ipdb> ckpt['cfg']
# {'embed_dim': 768, 'xf_n_layers': 11, 'sattn_n_heads': 24, 'xattn_n_heads': 24}
# ipdb> ckpt['cfg'] 2M
# {'embed_dim': 256, 'xf_n_layers': 1, 'sattn_n_heads': 8, 'xattn_n_heads': 8}