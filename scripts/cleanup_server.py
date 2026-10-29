"""One-time archive of historical launchers/manuscripts; preserve data and results."""
import json
import shutil
from pathlib import Path

root = Path('/home/user/code_1/WCM').resolve()
archive = root / '_archive/cleanup_20261026'
archive.mkdir(parents=True, exist_ok=True)
items = list(root.glob('*.sh'))
items += [root / p for p in ['run_scripts/_archive', 'scripts/_archive'] if (root / p).exists()]
items += [p for p in (root / 'paper').iterdir() if p.name not in {'stmemory', 'README.md'}]
docs = ['寄存器变体实验报告.md', '寄存器实验重构路线.md', '空间寄存器消融实验报告.md',
        '跨帧WCM离线强化学习方案.md', 'pipeline_handover.md', 'pistar06_vs_wcm.md',
        'rollout_wcm_adv_pi05.md', 'SpaceTime三方对照实验计划.md', 'Sparse4消融实验记录.md', 'WCM 价值函数流水线.md']
items += [root / name for name in docs if (root / name).exists()]
manifest = []
for source in items:
    relative = source.relative_to(root)
    dest = archive / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(dest))
    manifest.append({'source': str(relative), 'destination': str(dest)})
(archive / 'manifest_server.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f'Archived {len(manifest)} entries into {archive}')
