"""Generate Figure 2 replacements: seed42, joint10, no full fine-tuning."""
import copy
from pathlib import Path

import yaml

from world_critic.config import load_config, validate_train_config


def main():
    dest = Path('configs/module_replacement')
    dest.mkdir(parents=True, exist_ok=True)
    source = yaml.safe_load(Path('configs/wcm_spacetime_direct.yaml').read_text())
    for variant in ['replace_a_sampling', 'replace_b_encoding', 'replace_c_readout']:
        config = copy.deepcopy(source)
        config.update(seed=42, epochs=10, per_device_batch_size=4, eval_batch_size=4,
                      num_workers=4, save_every_epochs=10, checkpoint_last_only=True,
                      deploy_from_best=False,
                      init_from='/media/Data/user/WCM_outputs/wcm_sparse4_image_b4_exp_s42/deploy.pt',
                      partial_init=True,
                      output_dir=f'/media/Data/user/WCM_outputs/module_replacement_20261027/{variant}_s42')
        config['data'].update(history_offsets=[0,17,34,51,69,86,103,120], split_seed=3072,
                              split_manifest='/media/Data/user/WCM_outputs/wcm_spacetime_vit_t8_direct_t120_s42/episode_split.json')
        config['model'].update(spacetime_frame_count=8, spacetime_original_encoding=False,
                               spacetime_original_readout=False)
        if variant == 'replace_a_sampling':
            config['data']['history_offsets'] = [0,1,2]
            config['model']['spacetime_frame_count'] = 3
        elif variant == 'replace_b_encoding':
            config['model']['spacetime_original_encoding'] = True
        else:
            config['model']['spacetime_original_readout'] = True
        path = dest / f'{variant}_s42.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        validate_train_config(load_config(path))
        print(path)


if __name__ == '__main__':
    main()
