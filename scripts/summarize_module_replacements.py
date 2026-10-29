"""Compare Figure 2 a/b/c replacements to the paper's existing final H120 model."""
import argparse
import csv
import json
from pathlib import Path

from scripts.compare_experiments import compare_rows, curve_path, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    names = ['replace_a_sampling', 'replace_b_encoding', 'replace_c_readout']
    labels = ['a → 原始连续三帧', 'b → 原始逐帧CLS编码', 'c → 原始CLS/视角池化/context']
    paths = {'full': curve_path(str(root/'reference_paper_s42'))}
    paths.update({n: curve_path(str(root/f'{n}_s42/eval_ood')) for n in names})
    rows = {n: load(p) for n, p in paths.items()}
    keys = set.intersection(*(set(r) for r in rows.values()))
    manifest = root/'endpoint_manifest.json'
    manifest.write_text(json.dumps({'endpoints': [{'episode_id': e, 'frame_index': f} for e, f in sorted(keys)]}))
    table = []
    for name, label in zip(names, labels):
        result = compare_rows(rows['full'], rows[name], baseline_path=paths['full'], candidate_path=paths[name],
                              bootstrap_samples=20000, seed=42, align='common', endpoint_keys=keys,
                              endpoint_manifest=manifest)
        (root/f'{name}_vs_full.json').write_text(json.dumps(result, indent=2))
        if not table:
            table.append({'method': '论文 STMemory-120（历史最终模型）', 'MSE': result['baseline_mse'],
                          'Pearson': result['baseline_pearson'], 'centered_MSE': result['baseline_centered_mse'],
                          'bias': result['baseline_mean_bias'], 'delta_MSE': 0., 'delta_MSE_CI95': ''})
        table.append({'method': label, 'MSE': result['candidate_mse'], 'Pearson': result['candidate_pearson'],
                      'centered_MSE': result['candidate_centered_mse'], 'bias': result['candidate_mean_bias'],
                      'delta_MSE': result['mse_difference_candidate_minus_baseline'],
                      'delta_MSE_CI95': str(result['mse_paired_episode_bootstrap_ci95'])})
    with (root/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    headers = list(table[0])
    lines = ['# Figure 2 模块替换消融', '', f'125条 OOD 轨迹，共同 {len(keys)} 帧；新替换版seed42，joint10，无full阶段。', '',
             '| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
    for row in table:
        lines.append('| '+' | '.join(f'{row[k]:.6f}' if isinstance(row[k],float) else str(row[k]) for k in headers)+' |')
    lines += ['', 'ΔMSE 为替换版减完整模型；区间基于20000次配对episode bootstrap。',
              '完整模型复用论文H120的joint10+full3验证最优权重；新消融joint10、使用最后一轮，训练量和选模不同。',
              'a恢复原始帧数和跨度，b恢复CLS粒度并移除新增时序层，c恢复视角池化和顺序context接口。',
              '这是模块替换系统比较；不以单算子或等算力因果消融解释，不预设任何变体的胜负。']
    (root/'模块替换消融报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
