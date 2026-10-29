"""Real ViT GPU backward/update checks for Figure 2 replacement paths."""
import argparse

import torch

from world_critic.config import load_config
from world_critic.model import WorldCriticModel
from world_critic.training import configure_training_stage, create_optimizer, autocast_context


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    args = p.parse_args()
    torch.manual_seed(42)
    cfg = load_config(args.config)
    cfg.model.action_dim = 7
    model = WorldCriticModel(cfg.model).cuda().train()
    configure_training_stage(model, cfg)
    model.spacetime_encoder.set_gate(1.)
    optimizer = create_optimizer(model, cfg)
    t = cfg.model.spacetime_frame_count
    history = torch.randn(1, t, 2, 3, 224, 224, device='cuda', requires_grad=True)
    captured = {}
    def capture(_m, _i, output):
        captured['tokens'] = tuple(output[0].shape)
    handle = model.spacetime_encoder.register_forward_hook(capture)
    with autocast_context(torch.device('cuda'), cfg.precision):
        out = model(images=torch.randn(1,2,2,3,224,224,device='cuda'),
                    actions=torch.randn(1,1,7,device='cuda'),
                    instruction_input_ids=torch.zeros(1,8,dtype=torch.long,device='cuda'),
                    instruction_attention_mask=torch.ones(1,8,dtype=torch.long,device='cuda'),
                    valid_mask=torch.ones(1,1,dtype=torch.bool,device='cuda'), history_images=history)
        loss = out.value.square().mean()+out.next_state_pred.square().mean()+out.q_value.square().mean()+out.risk_logits.square().mean()
    loss.backward()
    assert out.value.shape == (1,1,1) and torch.isfinite(loss)
    assert history.grad[:, -1].norm() > 0
    encoder = model.spacetime_encoder
    if cfg.model.spacetime_original_encoding:
        assert len(encoder.temporal_layers) == 0
        assert model.vision_encoder.camera_embedding.grad.norm() > 0
    else:
        assert encoder.temporal_layers[0].self_attn.in_proj_weight.grad.norm() > 0
    if cfg.model.spacetime_original_readout:
        assert encoder.perceiver_queries is None and len(encoder.perceiver_layers) == 0
        assert model.view_attention.in_proj_weight.grad.norm() > 0
        assert captured['tokens'] == (1,8,384)
    else:
        assert encoder.perceiver_queries.grad.norm() > 0
        assert captured['tokens'] == (1,cfg.model.perceiver_queries,384)
    optimizer.step()
    handle.remove()
    print({'config': args.config, 'loss': float(loss.detach()), **captured,
           'status': 'forward/backward/oldest-frame/active-module gradients/update passed'}, flush=True)


if __name__ == '__main__':
    main()
