"""Small domain-method adaptations; existing SpaceTime is evaluation-only.

TD-MPC2: upstream SimNorm/NormedLinear/MLP + latent consistency/reward loss.
VIP: value-implicit representation loss + train-only supervised return readout.
IQL: twin-Q TD learning and expectile V regression, no actor.
All use frozen public ResNet18/CLIP features, not trained SpaceTime features.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import importlib.util
import json
import math
import random
import subprocess
import hashlib
import urllib.request
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


REPOS = {'tdmpc2': 'https://github.com/nicklashansen/tdmpc2.git',
         'vip': 'https://github.com/facebookresearch/vip.git',
         'iql': 'https://github.com/ikostrikov/implicit_q_learning.git'}
METHODS = ['tdmpc2', 'vip', 'iql']


def fetch(args):
    vendor = Path(args.output) / 'vendor_domain'
    vendor.mkdir(parents=True, exist_ok=True)
    sources = {}
    paths = {'tdmpc2': ('main', ['tdmpc2/common/layers.py', 'tdmpc2/tdmpc2.py', 'LICENSE']),
             'vip': ('main', ['vip/trainer.py', 'vip/models/model_vip.py', 'LICENSE']),
             'iql': ('master', ['learner.py', 'value_net.py', 'LICENSE'])}
    for name, url in REPOS.items():
        path = vendor / name
        branch, files = paths[name]
        sources[name] = {'repo': url, 'branch_at_download': branch, 'files': {}}
        repo = url.removeprefix('https://github.com/').removesuffix('.git')
        for file in files:
            address = f'https://raw.githubusercontent.com/{repo}/{branch}/{file}'
            with urllib.request.urlopen(address, timeout=30) as response:
                content = response.read()
            destination = path / file
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            sources[name]['files'][file] = {'url': address, 'sha256': hashlib.sha256(content).hexdigest()}
        print(f'[source] {name}: official source + LICENSE archived with SHA256', flush=True)
    # Import only these upstream building blocks, without unrelated planning dependencies.
    tree = ast.parse((vendor / 'tdmpc2/tdmpc2/common/layers.py').read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in {'SimNorm', 'NormedLinear', 'mlp'}]
    code = 'import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\n'
    code += '\n\n'.join(ast.unparse(node) for node in selected)
    (vendor / 'tdmpc2_layers.py').write_text(code + '\n')
    (vendor / 'sources.json').write_text(json.dumps(sources, indent=2))


def cache(args):
    import cv2
    import pyarrow.parquet as pq
    from torchvision.models import resnet18, ResNet18_Weights
    from transformers import AutoTokenizer, CLIPTextModel

    output = Path(args.output) / 'cache_domain'
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    split = json.loads(Path(args.split_manifest).read_text())
    weights = ResNet18_Weights.IMAGENET1K_V1
    visual = resnet18(weights=weights)
    visual.fc = nn.Identity()
    visual.eval().requires_grad_(False).to(args.device)
    language_name = 'openai/clip-vit-base-patch32'
    tokenizer = AutoTokenizer.from_pretrained(language_name)
    language = CLIPTextModel.from_pretrained(language_name).eval().requires_grad_(False).to(args.device)
    mean = torch.tensor([.485, .456, .406], device=args.device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=args.device)[None, :, None, None]
    keys = config['data']['image_keys']
    offsets = config['data']['history_offsets']
    norm_states, norm_actions = [], []
    for name, root_value in [('train', args.train_root), ('test', args.test_root)]:
        root = Path(root_value)
        dest = output / name
        dest.mkdir(exist_ok=True)
        info = json.loads((root / 'meta/info.json').read_text())
        tasks = [json.loads(l) for l in (root / 'meta/tasks.jsonl').read_text().splitlines() if l.strip()]
        with torch.inference_mode():
            inputs = tokenizer([t['task'] for t in tasks], padding=True, truncation=True, return_tensors='pt').to(args.device)
            text = language(**inputs).pooler_output.cpu()
        text_by_id = {t['task_index']: text[i] for i, t in enumerate(tasks)}
        episodes = []
        started = time.perf_counter()
        total = 0
        for file in sorted(root.glob('data/chunk-*/episode_*.parquet')):
            table = pq.read_table(file)
            ep = int(table['episode_index'][0].as_py())
            n = table.num_rows
            cams = []
            for key in keys:
                video = root / info['video_path'].format(chunk_index=ep // info['chunks_size'],
                                                         episode_chunk=ep // info['chunks_size'],
                                                         video_key=key, episode_index=ep)
                cap = cv2.VideoCapture(str(video))
                result = []
                for start in range(0, n, 128):
                    batch = []
                    for _ in range(min(128, n - start)):
                        _, frame = cap.read()
                        h, w = frame.shape[:2]
                        resized = cv2.resize(frame, (round(w * 256 / min(h, w)), round(h * 256 / min(h, w))))
                        y, x = (resized.shape[0] - 224) // 2, (resized.shape[1] - 224) // 2
                        batch.append(cv2.cvtColor(resized[y:y+224, x:x+224], cv2.COLOR_BGR2RGB))
                    pixels = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(args.device).float() / 255.
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
                        result.append(visual((pixels - mean) / std).half().cpu())
                cap.release()
                cams.append(torch.cat(result))
            state = torch.tensor(np.asarray(table[config['data']['state_key']].to_pylist()), dtype=torch.float32)
            action = torch.tensor(np.asarray(table[config['data']['action_key']].to_pylist()), dtype=torch.float32)
            returns = torch.tensor(table[config['data']['return_key']].to_pylist(), dtype=torch.float32).flatten()
            success = bool(table[config['data']['success_key']][0].as_py())
            torch.save({'features': torch.stack(cams, dim=1), 'state': state, 'action': action,
                        'target': returns, 'success': success,
                        'frame_index': torch.tensor(table['frame_index'].to_pylist()),
                        'language': text_by_id[int(table['task_index'][0].as_py())]}, dest / f'{ep:06d}.pt')
            episodes.append({'episode_id': ep, 'frames': n, 'success': success})
            if name == 'train' and ep in split['train']:
                norm_states.append(state)
                norm_actions.append(action)
            total += n * len(keys)
            elapsed = time.perf_counter() - started
            print(f'[cache {name}] episode={ep} frames={n} images={total} seconds={elapsed:.1f} images/s={total/elapsed:.1f}', flush=True)
        manifest = {'root': str(root.resolve()), 'episodes': episodes, 'image_keys': keys, 'offsets': offsets,
                    'vision': 'torchvision/ResNet18/IMAGENET1K_V1', 'vision_dim': 512,
                    'language': language_name, 'language_commit': language.config._commit_hash,
                    'seconds': time.perf_counter() - started, 'split': split,
                    'preprocessing': 'RGB, resize shorter side 256, center crop 224, ImageNet normalize'}
        (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    normalization = {}
    for name, values in [('state', norm_states), ('action', norm_actions)]:
        data = torch.cat(values)
        normalization[name] = {'mean': data.mean(0).tolist(), 'std': data.std(0, unbiased=False).clamp_min(1e-6).tolist()}
    (output / 'normalization.json').write_text(json.dumps(normalization))
    (output / 'episode_split.json').write_text(json.dumps(split, indent=2))


def mlp(inputs, outputs, hidden=256):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.Mish(),
                         nn.Linear(hidden, hidden), nn.Mish(), nn.Linear(hidden, outputs))


class DomainCritic(nn.Module):
    def __init__(self, method, observation_dim, action_dim, vendor):
        super().__init__()
        self.method = method
        if method == 'tdmpc2':
            spec = importlib.util.spec_from_file_location('tdmpc2_layers', Path(vendor) / 'tdmpc2_layers.py')
            layer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(layer)
            simnorm = lambda: layer.SimNorm(SimpleNamespace(simnorm_dim=8))
            self.encoder = layer.mlp(observation_dim, [256, 256], 128, act=simnorm())
            self.dynamics = layer.mlp(128 + action_dim, [256, 256], 128, act=simnorm())
            self.reward = layer.mlp(128 + action_dim, [256, 256], 1)
            self.value = layer.mlp(128, [256, 256], 1)
        elif method == 'vip':
            self.encoder = mlp(observation_dim, 128)
            self.value = mlp(128, 1)
        else:
            self.encoder = mlp(observation_dim, 128)
            self.value = mlp(128, 1)
            self.q1, self.q2 = mlp(128 + action_dim, 1), mlp(128 + action_dim, 1)

    def forward(self, obs):
        return self.value(self.encoder(obs)).squeeze(-1)


def prepare_data(args):
    root = Path(args.output) / 'cache_domain'
    stats = json.loads((root / 'normalization.json').read_text())
    split = json.loads((root / 'episode_split.json').read_text())
    means = {key: torch.tensor(value['mean']) for key, value in stats.items()}
    stds = {key: torch.tensor(value['std']) for key, value in stats.items()}
    datasets = {}
    for name in ['train', 'val', 'test']:
        folder = root / ('test' if name == 'test' else 'train')
        manifest = json.loads((folder / 'manifest.json').read_text())
        selected = {e['episode_id'] for e in manifest['episodes']} if name == 'test' else set(split[name])
        rows = {k: [] for k in ['obs', 'next_obs', 'action', 'reward', 'done', 'target', 'episode', 'frame', 'start', 'goal', 'success']}
        for episode in manifest['episodes']:
            ep = episode['episode_id']
            if ep not in selected:
                continue
            data = torch.load(folder / f'{ep:06d}.pt', weights_only=False, map_location='cpu')
            # All methods receive the same 8 sampled-frame features, language and current state.
            offsets = torch.tensor(sorted(manifest['offsets'], reverse=True))
            n = len(data['target'])
            frame = torch.arange(n)
            history = (frame[:, None] - offsets).clamp_min(0)
            obs = torch.cat([data['features'][history].float().flatten(1),
                             data['language'].expand(n, -1),
                             (data['state'] - means['state']) / stds['state']], dim=-1)
            idx = torch.arange(0, n, 4 if name == 'train' else 1)
            if name == 'train':
                idx = torch.cat([idx, torch.tensor([n - 1])]).unique(sorted=True)
            next_idx = (idx + 1).clamp_max(n - 1)
            done = idx == n - 1
            reward = data['target'][idx] - torch.where(done, 0., data['target'][next_idx])
            payload = {'obs': obs[idx], 'next_obs': obs[next_idx],
                       'action': (data['action'][idx] - means['action']) / stds['action'],
                       'reward': reward, 'done': done.float(), 'target': data['target'][idx],
                       'episode': torch.full((len(idx),), ep), 'frame': data['frame_index'][idx],
                       'start': obs[0].expand(len(idx), -1), 'goal': obs[-1].expand(len(idx), -1),
                       'success': torch.full((len(idx),), data['success'])}
            for key, value in payload.items():
                rows[key].append(value)
        datasets[name] = {key: torch.cat(value).to(args.device) for key, value in rows.items()}
    return datasets


def evaluate(model, data, batch_size=512):
    model.eval()
    with torch.inference_mode():
        values = torch.cat([model(x) for x in data['obs'].split(batch_size)])
    error = values - data['target']
    return float(error.square().mean()), values.cpu()


def loss(model, target_model, data, idx):
    b = {k: v[idx] for k, v in data.items()}
    z = model.encoder(b['obs'])
    value = model.value(z).squeeze(-1)
    if model.method == 'tdmpc2':
        za = torch.cat([z, b['action']], -1)
        with torch.no_grad():
            next_z = target_model.encoder(b['next_obs'])
        consistency = F.mse_loss(model.dynamics(za), next_z)
        reward_loss = F.mse_loss(model.reward(za).squeeze(-1), b['reward'])
        value_loss = F.mse_loss(value, b['target'])
        total = value_loss + 20. * consistency + .1 * reward_loss
        return total, {'value': float(value_loss.detach()), 'consistency': float(consistency.detach())}
    if model.method == 'vip':
        # Successful terminal observations are training goals only; no goal is used at inference.
        good = b['success']
        eg = model.encoder(b['goal'][good])
        e0 = model.encoder(b['start'][good])
        e1 = model.encoder(b['next_obs'][good])
        sim = lambda a, g: -torch.linalg.vector_norm(a - g, dim=-1)
        vip_reward = torch.where(b['done'][good] > 0, 0., -1.)
        residual = vip_reward + .98 * sim(e1, eg) - sim(z[good], eg)
        vip = .02 * (-sim(e0, eg)).mean() + torch.logsumexp(-residual, 0) - math.log(len(residual))
        # Detach readout so scalar supervision does not alter the VIP representation objective.
        readout = model.value(z.detach()).squeeze(-1)
        regression = F.mse_loss(readout, b['target'])
        total = vip + regression
        return total, {'readout': float(regression.detach()), 'vip': float(vip.detach())}
    with torch.no_grad():
        target_z = target_model.encoder(b['obs'])
        target_za = torch.cat([target_z, b['action']], -1)
        q = torch.minimum(target_model.q1(target_za), target_model.q2(target_za)).squeeze(-1)
        next_value = model(b['next_obs'])
        td_target = b['reward'] + (1. - b['done']) * next_value
    diff = q - value
    v_loss = (torch.where(diff > 0, .7, .3) * diff.square()).mean()
    za = torch.cat([z, b['action']], -1)
    q_loss = F.mse_loss(model.q1(za).squeeze(-1), td_target) + F.mse_loss(model.q2(za).squeeze(-1), td_target)
    return v_loss + q_loss, {'expectile': float(v_loss.detach()), 'td': float(q_loss.detach())}


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    datasets = prepare_data(args)
    data = datasets['train']
    out = Path(args.output) / f'{args.model}_s{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    model = DomainCritic(args.model, data['obs'].shape[-1], data['action'].shape[-1],
                         Path(args.output) / 'vendor_domain').to(args.device)
    target_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best = float('inf')
    started = time.perf_counter()
    success_ids = torch.where(data['success'])[0]
    failure_ids = torch.where(~data['success'])[0]
    with (out / 'metrics.jsonl').open('w') as log:
        for step in range(1, args.steps + 1):
            # Balanced draws ensure VIP has real successful goals in every minibatch.
            half = args.batch_size // 2
            idx = torch.cat([success_ids[torch.randint(len(success_ids), (half,), device=args.device)],
                             failure_ids[torch.randint(len(failure_ids), (args.batch_size-half,), device=args.device)]])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total, parts = loss(model, target_model, data, idx)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(target_model.parameters(), model.parameters()):
                    target.lerp_(source, .01)
            if step % 100 == 0 or step == args.steps:
                val_mse, _ = evaluate(model, datasets['val'])
                elapsed = time.perf_counter() - started
                row = {'step': step, 'loss': float(total.detach()), 'parts': parts,
                       'val_mse': val_mse, 'seconds': elapsed}
                log.write(json.dumps(row) + '\n')
                log.flush()
                print(f'[{args.model} s{args.seed}] step={step}/{args.steps} val_mse={val_mse:.6f} elapsed={elapsed:.1f}s', flush=True)
                if val_mse < best:
                    best = val_mse
                    torch.save({'model': model.state_dict(), 'args': vars(args), 'best_step': step,
                                'method': __doc__, 'observation_dim': data['obs'].shape[-1],
                                'sources': json.loads((Path(args.output)/'vendor_domain/sources.json').read_text())}, out / 'best.pt')
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(out / 'best.pt', weights_only=False, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    test_mse, values = evaluate(model, datasets['test'])
    test = datasets['test']
    with (out / 'episode_curves.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['episode_id', 'frame_index', 'value', 'return'])
        writer.writerows(zip(test['episode'].cpu().tolist(), test['frame'].cpu().tolist(),
                             values.tolist(), test['target'].cpu().tolist()))
    summary = {'model': args.model, 'seed': args.seed, 'steps': args.steps,
               'best_step': checkpoint['best_step'], 'val_mse': best, 'test_mse': test_mse,
               'parameters': sum(p.numel() for p in model.parameters()),
               'train_seconds': elapsed, 'gpu': torch.cuda.get_device_name(),
               'feature_encoder': 'frozen ResNet18 ImageNet + frozen CLIP text',
               'train_samples': len(data['obs']), 'val_samples': len(datasets['val']['obs']),
               'test_samples': len(test['obs'])}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[done] {args.model} test_mse={test_mse:.6f} train_seconds={elapsed:.1f}', flush=True)


def smoke(args):
    torch.manual_seed(42)
    for method in METHODS:
        model = DomainCritic(method, 64, 7, Path(args.output) / 'vendor_domain').to(args.device)
        target = copy.deepcopy(model).requires_grad_(False)
        obs = torch.randn(16, 64, device=args.device)
        batch = {k: obs.clone() for k in ['obs', 'next_obs', 'start', 'goal']}
        batch.update(action=torch.randn(16, 7, device=args.device), reward=-torch.ones(16, device=args.device)*.01,
                     done=torch.zeros(16, device=args.device), target=-torch.rand(16, device=args.device),
                     success=torch.tensor([True]*8 + [False]*8, device=args.device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        before = model.value[-1].weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad()
            objective, _ = loss(model, target, batch, torch.arange(16, device=args.device))
            objective.backward()
            optimizer.step()
        assert torch.isfinite(objective) and not torch.equal(before, model.value[-1].weight)
        print(f'[smoke] {method}: finite loss, backward, parameter updates passed', flush=True)


def summarize(args):
    from scripts.compare_experiments import curve_path, load, compare_rows
    out = Path(args.output)
    ours_path = curve_path(args.ours_eval)
    ours = load(ours_path)
    paths = [out / f'{method}_s{seed}' for seed in args.seeds for method in METHODS]
    predictions = {path: load(path/'episode_curves.csv') for path in paths}
    common = set(ours)
    for rows in predictions.values():
        common &= set(rows)
    manifest = out / 'endpoint_manifest.json'
    manifest.write_text(json.dumps({'endpoints': [{'episode_id': ep, 'frame_index': frame} for ep, frame in sorted(common)]}))
    results = []
    comparisons = out / 'comparisons'
    comparisons.mkdir(exist_ok=True)
    for path in paths:
        summary = json.loads((path/'summary.json').read_text())
        result = compare_rows(predictions[path], ours, baseline_path=path, candidate_path=ours_path,
                              bootstrap_samples=20000, seed=summary['seed'], align='common',
                              endpoint_keys=common, endpoint_manifest=manifest)
        (comparisons/f'ours_vs_{path.name}.json').write_text(json.dumps(result, indent=2))
        ci = result['mse_paired_episode_bootstrap_ci95']
        results.append({'method': path.name, 'MSE': result['baseline_mse'], 'Pearson': result['baseline_pearson'],
                        'centered_MSE': result['baseline_centered_mse'], 'bias': result['baseline_mean_bias'],
                        'trainable_params': summary['parameters'], 'train_seconds': summary['train_seconds'],
                        'SpaceTime_MSE_reduction_pct': -result['relative_mse_change']*100,
                        'SpaceTime_minus_baseline_CI95': f'[{ci[0]:.6f}, {ci[1]:.6f}]'})
    results.append({'method': 'SpaceTime existing s42', 'MSE': result['candidate_mse'],
                    'Pearson': result['candidate_pearson'], 'centered_MSE': result['candidate_centered_mse'],
                    'bias': result['candidate_mean_bias'], 'trainable_params': 'existing', 'train_seconds': 0,
                    'SpaceTime_MSE_reduction_pct': '', 'SpaceTime_minus_baseline_CI95': ''})
    with (out/'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    headers = list(results[0])
    lines = ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
    for row in results:
        lines.append('| '+' | '.join(f'{row[k]:.6f}' if isinstance(row[k], float) else str(row[k]) for k in headers)+' |')
    lines += ['', f'Common endpoints: {len(common)}; paired episode bootstrap: 20000.',
              f'SpaceTime: {args.ours_eval}; predictions reused, no training.',
              'External methods are lightweight domain-method adaptations, not full algorithm reproductions.',
              'All external models use identical frozen public ResNet18/CLIP features and 8-frame H120 histories.',
              'TD-MPC2-style: upstream SimNorm/MLP + 1-step latent consistency and reward + MC V readout; no actor, planner, Q ensemble or two-hot targets.',
              'VIP-style: official value-implicit loss on successful training goals + detached supervised V readout; frozen ResNet18, not full end-to-end VIP.',
              'IQL-critic: twin-Q TD and expectile=0.7, gamma=1 to match undiscounted returns; no actor.',
              'Reward is the difference of consecutive provided returns, with terminal bootstrap zero, preserving the dataset return scale including clipping.',
              'No action, trajectory length, success, future frame or goal is input to the inference V function.',
              'Success/failure labels are used only to balance training batches and select valid VIP training goals.',
              'Best checkpoint chosen by episode-disjoint validation MSE every 100 updates. Test never selects weights.']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['fetch', 'cache', 'smoke', 'train', 'summarize'])
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--config')
    p.add_argument('--split-manifest')
    p.add_argument('--train-root')
    p.add_argument('--test-root')
    p.add_argument('--model', choices=METHODS)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--ours-eval')
    args = p.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
