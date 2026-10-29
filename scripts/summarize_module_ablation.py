"""Summarize three seed-42 ablations against the historical full H120 model."""
import argparse
import csv
import json
from pathlib import Path

from scripts.compare_experiments import compare_rows, curve_path, load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='outputs/module_ablation_20261026')
    args = p.parse_args()
    root = Path(args.root)
    paths = {'full': curve_path(str(root/'reference_full_s42'))}
    for variant in ['no_temporal', 'queries128', 'four_frames']:
        paths[variant] = curve_path(str(root/f'{variant}_s42/eval_ood'))
    predictions = {name: load(path) for name, path in paths.items()}
    keys = set.intersection(*(set(rows) for rows in predictions.values()))
    manifest = root/'endpoint_manifest.json'
    manifest.write_text(json.dumps({'endpoints': [{'episode_id': ep, 'frame_index': f}
                                                  for ep, f in sorted(keys)]}, indent=2))
    labels = {'full': '完整 STMemory-120（历史）', 'no_temporal': '去时序注意力',
              'queries128': '128 queries', 'four_frames': '四帧输入'}
    results, table = {}, []
    for variant in ['no_temporal', 'queries128', 'four_frames']:
        result = compare_rows(predictions['full'], predictions[variant],
                              baseline_path=paths['full'], candidate_path=paths[variant],
                              bootstrap_samples=20000, seed=42, align='common',
                              endpoint_keys=keys, endpoint_manifest=manifest)
        results[variant] = result
        (root/f'{variant}_vs_full_s42.json').write_text(json.dumps(result, indent=2))
    r = results['no_temporal']
    table.append({'variant': labels['full'], 'MSE': r['baseline_mse'], 'Pearson': r['baseline_pearson'],
                  'centered_MSE': r['baseline_centered_mse'], 'bias': r['baseline_mean_bias'],
                  'delta_MSE': 0., 'MSE_change_pct': 0., 'delta_MSE_CI95': '', 'delta_Pearson_CI95': ''})
    for variant, r in results.items():
        mc, pc = r['mse_paired_episode_bootstrap_ci95'], r['pearson_paired_episode_bootstrap_ci95']
        table.append({'variant': labels[variant], 'MSE': r['candidate_mse'], 'Pearson': r['candidate_pearson'],
                      'centered_MSE': r['candidate_centered_mse'], 'bias': r['candidate_mean_bias'],
                      'delta_MSE': r['mse_difference_candidate_minus_baseline'],
                      'MSE_change_pct': r['relative_mse_change']*100,
                      'delta_MSE_CI95': f'[{mc[0]:.6f}, {mc[1]:.6f}]',
                      'delta_Pearson_CI95': f'[{pc[0]:.6f}, {pc[1]:.6f}]'})
    with (root/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    headers = list(table[0])
    lines = ['# STMemory 模块消融结果（seed 42）', '',
             f'三项消融均已完成两阶段训练和 OOD 评估；单主相机已按用户要求取消。共同测试 {r["episodes"]} 条轨迹、{len(keys):,} 帧。', '',
             '| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
    for row in table:
        lines.append('| '+' | '.join(f'{row[k]:.6f}' if isinstance(row[k], float) else str(row[k]) for k in headers)+' |')
    lines += ['', 'Δ 均为消融版本减历史完整模型；正 ΔMSE 表示误差增加。95% CI 使用 20000 次配对 episode bootstrap。', '',
              '## 结果解释', '',
              '三项变体的 ΔMSE 95% CI 均完全小于 0：本轮三项变体均比历史完整模型的 OOD MSE 更低。',
              '去时序注意力与四帧输入的 Pearson 差值区间包含 0，未检出显著相关性变化；128 queries 的 Pearson 差值区间为正，但下界接近 0。',
              '去时序层、四帧和 128 queries 的 centered MSE 也均低于完整模型，差异不完全由 mean bias 解释；这里未单独检验 centered MSE 的显著性。',
              '本轮不能支持“时序注意力或八帧输入提高精度”的论断。四帧输入是当前较低误差的轻量候选，128 queries 是较高 Pearson 的候选；不能仅凭本表宣称某变体在所有指标上最优。', '',
              '## 实验范围', '',
              '完整模型复用历史 seed42、验证集最优 checkpoint；新消融使用最后一轮 checkpoint。选模规则不同，因此结果作为单 seed 消融证据，不是完全匹配选模的因果实验。',
              '所有三项消融的 joint 阶段均从同 seed 公共上游 checkpoint 初始化，并训练 joint 10 + full 3 epoch。',
              '已完成变体最终权重保留 full/checkpoints/last.pt 和 full/deploy.pt，joint 权重已清理。',
              '均值池化与 16 queries 本轮未实现、未运行。结果仅覆盖本表三项变体。']
    (root/'消融结果报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
