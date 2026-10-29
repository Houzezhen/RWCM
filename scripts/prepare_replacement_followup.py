"""Continue abc replacements for full3; add a 16-query joint10+full3 model."""
import copy
from pathlib import Path

import yaml

from world_critic.config import load_config, validate_train_config


def main():
    dest = Path('configs/replacement_followup')
    dest.mkdir(parents=True, exist_ok=True)
    base = '/media/Data/user/WCM_outputs/replacement_followup_20261027'
    full_template = yaml.safe_load(Path('configs/wcm_spacetime_direct_full.yaml').read_text())
    for name in ['replace_a_sampling','replace_b_encoding','replace_c_readout','queries16']:
        if name == 'queries16':
            joint = yaml.safe_load(Path('configs/module_replacement/replace_b_encoding_s42.yaml').read_text())
            joint['model'].update(spacetime_original_encoding=False,spacetime_original_readout=False,perceiver_queries=16)
            joint['output_dir'] = f'{base}/{name}_s42/joint'
            save(joint, dest/f'{name}_s42_joint.yaml')
        else:
            joint = yaml.safe_load(Path(f'configs/module_replacement/{name}_s42.yaml').read_text())
        full = copy.deepcopy(joint)
        for key in ['epochs','training_stage','optim','unfreeze_vision_top_layers']:
            if key in full_template:
                full[key] = copy.deepcopy(full_template[key])
        full.update(epochs=3,training_stage='spacetime_full',partial_init=False,
                    checkpoint_last_only=True,deploy_from_best=False,save_every_epochs=3,
                    output_dir=f'{base}/{name}_s42/full',
                    init_from=f'{base}/{name}_s42/joint/deploy.pt' if name=='queries16' else
                    f'/media/Data/user/WCM_outputs/module_replacement_20261027/{name}_s42/deploy.pt')
        save(full,dest/f'{name}_s42_full.yaml')


def save(config,path):
    path.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    validate_train_config(load_config(path))
    print(path)


if __name__=='__main__':
    main()
