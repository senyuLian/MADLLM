import os
import sys
import tempfile
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plm_special.models.rl_policy import OfflineRLPolicy


def test_pre_rs_can_be_disabled_without_changing_dataset():
    class DummyEncoder(torch.nn.Module):
        def forward(self, x):
            return [x[:, :, 0:1], x[:, :, 1:2], x[:, :, 2:3], x[:, :, 3:4], x[:, :, 4:5], x[:, :, 5:6], x[:, :, 6:7], x[:, :, 7:8], x[:, :, 8:9]]

    class DummyPLM(torch.nn.Module):
        def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=False, stop_layer_idx=None):
            batch, seq_len, _ = inputs_embeds.shape
            last_hidden_state = torch.zeros(batch, seq_len, inputs_embeds.shape[-1])
            return {'last_hidden_state': last_hidden_state}

    model = OfflineRLPolicy(
        state_feature_dim=1,
        bitrate_levels=2,
        state_encoder=DummyEncoder(),
        plm=DummyPLM(),
        plm_embed_size=4,
        max_length=4,
        max_ep_len=3,
        device='cpu',
        use_pre_r=False,
    )

    agent_ids = torch.zeros(1, 1, 1)
    pre_rs = torch.ones(1, 1, 1)
    states = torch.zeros(1, 1, 9)
    actions = torch.zeros(1, 1, 1)
    returns = torch.zeros(1, 1, 1)
    timesteps = torch.zeros(1, 1, dtype=torch.int64)

    out = model(agent_ids, pre_rs, states, actions, returns, timesteps)
    assert out.shape == (1, 1, 2)
