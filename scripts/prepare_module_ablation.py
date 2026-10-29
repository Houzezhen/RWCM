"""Generate matched H120 two-stage configs for currently supported ablations."""
import argparse
import copy
from pathlib import Path

import yaml

from world_critic.config import load_config, validate_train_config

VARIANTS = ['no_temporal', 'queries128', 'four_frames', 'primary_only']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', default='/media/Data/user/WCM_outputs/module_ablation_20261026')
    p.add_argument('--config-dir', default='configs/module_ablation')
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    args = p.parse_args()
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        split = f'/media/Data/user/WCM_outputs/wcm_spacetime_vit_t8_direct_full_t120_s{seed}/episode_split.json'
        for variant in VARIANTS:
            for stage in ['joint', 'full']:
                source = 'configs/wcm_spacetime_direct.yaml' if stage == 'joint' else 'configs/wcm_spacetime_direct_full.yaml'
                config = copy.deepcopy(yaml.safe_load(Path(source).read_text()))
                output = Path(args.output_root) / f'{variant}_s{seed}'
                config.update(seed=seed, output_dir=str(output / stage), per_device_batch_size=4,
                              eval_batch_size=4, num_workers=4, save_every_epochs=10 if stage == 'joint' else 3,
                              checkpoint_last_only=True, deploy_from_best=False)
                config['data'].update(history_offsets=[0, 17, 34, 51, 69, 86, 103, 120],
                                      split_seed=3072, split_manifest=split)
                if stage == 'joint':
                    config['init_from'] = f'/media/Data/user/WCM_outputs/wcm_sparse4_image_b4_exp_s{seed}/deploy.pt'
                    config['partial_init'] = True
                else:
                    config['init_from'] = str(output / 'joint/deploy.pt')
                    config['partial_init'] = False
                if variant == 'no_temporal':
                    config['model']['spacetime_temporal_enabled'] = False
                elif variant == 'queries128':
                    config['model']['perceiver_queries'] = 128
                elif variant == 'four_frames':
                    config['data']['history_offsets'] = [0, 40, 80, 120]
                    config['model']['spacetime_frame_count'] = 4
                elif variant == 'primary_only':
                    config['data']['image_keys'] = ['observation.images.primary']
                path = config_dir / f'{variant}_s{seed}_{stage}.yaml'
                path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
                validate_train_config(load_config(path))
                print(path)


if __name__ == '__main__':
    main()
