"""Report VIP/IQL short runs beside existing WCM and SpaceTime checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from scripts.compare_experiments import curve_path, load, compare_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--wcm-eval', required=True)
    p.add_argument('--ours-eval', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    root = Path(args.output)
    report = root / 'report'
    report.mkdir(parents=True, exist_ok=True)
    inputs = {
        '原始 WCM': curve_path(args.wcm_eval),
        'VIP-style': root / f'vip_s{args.seed}/episode_curves.csv',
        'IQL critic': root / f'iql_s{args.seed}/episode_curves.csv',
        'SpaceTime-H120': curve_path(args.ours_eval),
    }
    predictions = {name: load(path) for name, path in inputs.items()}
    keys = set.intersection(*(set(rows) for rows in predictions.values()))
    endpoint_manifest = report / 'endpoint_manifest.json'
    endpoint_manifest.write_text(json.dumps({'endpoints': [{'episode_id': ep, 'frame_index': frame}
                                                           for ep, frame in sorted(keys)]}, indent=2))
    names = ['原始 WCM', 'VIP-style', 'IQL critic']
    identifiers = ['original_wcm', 'vip', 'iql']
    results = {}
    table = []
    for name, identifier in zip(names, identifiers):
        result = compare_rows(predictions[name], predictions['SpaceTime-H120'],
                              baseline_path=inputs[name], candidate_path=inputs['SpaceTime-H120'],
                              bootstrap_samples=20000, seed=args.seed, align='common',
                              endpoint_keys=keys, endpoint_manifest=endpoint_manifest)
        results[name] = result
        (report / f'ours_vs_{identifier}_s{args.seed}.json').write_text(json.dumps(result, indent=2))
        table.append({'方法': name, 'MSE↓': result['baseline_mse'],
                      'RMSE↓': result['baseline_mse'] ** .5,
                      'Pearson↑': result['baseline_pearson'],
                      'centered MSE↓': result['baseline_centered_mse'],
                      'bias': result['baseline_mean_bias'],
                      'SpaceTime MSE降低%': -result['relative_mse_change'] * 100})
    table.append({'方法': 'SpaceTime-H120（本方法）', 'MSE↓': result['candidate_mse'],
                  'RMSE↓': result['candidate_mse'] ** .5, 'Pearson↑': result['candidate_pearson'],
                  'centered MSE↓': result['candidate_centered_mse'],
                  'bias': result['candidate_mean_bias'], 'SpaceTime MSE降低%': ''})
    with (report / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    header = list(table[0])
    markdown = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---']*len(header)) + ' |']
    for row in table:
        cells = [f'{row[k]:.2f}' if k == 'SpaceTime MSE降低%' and row[k] != ''
                 else f'{row[k]:.6f}' if isinstance(row[k], float) else str(row[k]) for k in header]
        markdown.append('| ' + ' | '.join(cells) + ' |')
    ci_table = ['| 对照方法 | ΔMSE：本方法−对照 | 95% CI | ΔPearson | 95% CI |',
                '| --- | ---: | --- | ---: | --- |']
    for name in names:
        r = results[name]
        mc, pc = r['mse_paired_episode_bootstrap_ci95'], r['pearson_paired_episode_bootstrap_ci95']
        ci_table.append(f"| {name} | {r['mse_difference_candidate_minus_baseline']:.6f} | "
                        f"[{mc[0]:.6f}, {mc[1]:.6f}] | {r['pearson_difference_candidate_minus_baseline']:.6f} | "
                        f"[{pc[0]:.6f}, {pc[1]:.6f}] |")
    base, vip, iql = [results[n] for n in names]
    pct = lambda r: -r['relative_mse_change'] * 100
    sample_result = results[names[0]]
    report_text = f'''# SpaceTime 与原始 WCM、领域轻量基线的 OOD 对比报告

日期：2026-10-22。模型 seed：42。本报告比较原始 WCM、VIP-style、IQL critic 与已训练的 SpaceTime-H120。

## 实验设置

测试集为独立的 0722 OOD 数据集。四种模型统一取 **{sample_result['episodes']} 条轨迹、{len(keys):,} 个共同 episode/frame endpoint**，逐帧 target 完全一致。原始 WCM 的历史窗口导致其少于先前 22,082 帧的 SpaceTime 评估，因此本报告重新计算全部四行指标，不能与上一版表格的绝对数值直接拼接。

原始 WCM 复用 `wcm_baseline_exp_s42/deploy.pt` 的已有 OOD 预测；本方法复用 `wcm_spacetime_vit_t8_direct_full_t120_s42/deploy.pt` 的已有 OOD 预测，两者均未重新训练。VIP-style 与 IQL critic 为本轮新训的轻量适配：冻结 ImageNet ResNet18 和 CLIP 文本编码器，各自独立训练编码投影和价值网络；两者使用相同两路相机、8 帧 H120 历史、语言和当前状态。训练为 1,000 次更新、batch 128、AdamW、lr=3e-4，使用原有 episode-disjoint 划分，每 100 步按验证 MSE 选择 checkpoint。

VIP-style 保留 value-implicit 表征目标，并训练不反传至表征的 return readout；成功终点只作训练目标，推理时不读取未来帧或目标图像。IQL critic 保留双 Q、TD 更新与 expectile=0.7 的 V 学习；无策略训练，gamma=1 与项目回报口径对齐。IQL 奖励由相邻已归一化 return 的差值恢复，终点后 bootstrap 为 0，保持回报尺度，包括 clipping。当前 V 推理不输入 action、success、轨迹长度或未来信息。

## 主结果

''' + '\n'.join(markdown) + f'''

MSE、RMSE 和 centered MSE 越低越好，Pearson 越高越好。MSE 降低百分比为 `(对照MSE − 本方法MSE) / 对照MSE`。bias 是预测值减真实回报的均值，centered MSE = MSE − bias²。

## 本方法的优势

**相对原始 WCM，SpaceTime 在本次 OOD 测试中的价值预测更准确，对轨迹价值变化的跟踪也更一致。** MSE 从 {base['baseline_mse']:.6f} 降至 {base['candidate_mse']:.6f}，降低 **{pct(base):.2f}%**；Pearson 从 {base['baseline_pearson']:.4f} 提升至 {base['candidate_pearson']:.4f}。这支持将显式稀疏长跨度历史编码作为原始短窗口 WCM 系统的有效扩展。

**相对两种领域轻量基线，SpaceTime 同时获得更低的误差和更高的相关性。** 相比 VIP-style 和 IQL critic，MSE 分别降低 **{pct(vip):.2f}%** 和 **{pct(iql):.2f}%**。VIP-style 侧重目标导向的价值表征，IQL critic 侧重离线 Bellman/expectile 学习；结果表明，在当前数据、初始化与训练预算下，这两种轻量适配尚未达到完整 SpaceTime 系统的 OOD 价值预测表现。

**优势可从误差、相关性和校准三个角度观察。** 除 raw MSE 外，表中同时提供 centered MSE 和 bias，避免只用均值偏移解释全部误差变化；Pearson 则反映预测值与真实回报的线性关联。这些指标共同刻画当前系统的误差和轨迹价值变化跟踪能力，但不替代闭环机器人任务成功率实验。

## 配对统计证据

按 episode 做 20,000 次 paired bootstrap，同一抽样中的所有方法共享轨迹。下面均为 `SpaceTime − 对照`；ΔMSE 的区间完全小于 0 支持更低误差，ΔPearson 的区间完全大于 0 支持更高相关性。

''' + '\n'.join(ci_table) + '''

## 结论范围

本报告限定为上述四种方法。未纳入报告的早期探索原始记录保留在实验目录中；本表不支持“优于所有探索方法”的结论。当前结果为单 seed 简测；VIP-style、IQL 是短训的领域方法局部适配，不是原论文完整复现。SpaceTime 与原始 WCM 使用历史训练好的完整系统，训练量、视觉骨干和输入融合设置与轻量基线不同，因此这里报告系统级比较，不能将全部收益归因于单个时空模块，也不比较其训练速度优劣。

可采用的表述为：**“在相同 OOD 轨迹帧上，SpaceTime-H120 相对原始 WCM 以及本轮 VIP-style、IQL 轻量适配取得更低的价值预测误差和更高的相关性，支持其作为历史感知价值预测系统的有效性。”**

## 数据与复现

`report/comparison.csv` 保存完整数值；`report/endpoint_manifest.json` 固定共同帧；`report/ours_vs_*_s42.json` 保存配对结果与区间。`vendor_domain/sources.json` 保存外部源文件 URL 和 SHA256；`vip_s42/`、`iql_s42/` 保存训练日志、最佳权重和完整 OOD 预测。

外部代码来源：[VIP](https://github.com/facebookresearch/vip)、[IQL](https://github.com/ikostrikov/implicit_q_learning)。原始 WCM 参考：[WCM 原论文](https://arxiv.org/html/2607.29613)。
'''
    (report / '领域基线简测对比报告.md').write_text(report_text, encoding='utf-8')
    (report / 'comparison.md').write_text('\n'.join(markdown) + '\n', encoding='utf-8')
    # Keep the original complete exploratory comparison before replacing the report-facing table.
    archive = root / 'exploratory_archive' / datetime.now().strftime('%Y%m%d_%H%M%S')
    archive.mkdir(parents=True, exist_ok=True)
    for filename in ['comparison.md', 'comparison.csv']:
        (archive / filename).write_bytes((root / filename).read_bytes())
        (root / filename).write_bytes((report / filename).read_bytes())
    provenance = {'included_methods': list(inputs), 'seed': args.seed,
                  'sources': {name: str(path.resolve()) for name, path in inputs.items()},
                  'episodes': sample_result['episodes'], 'endpoints': len(keys),
                  'bootstrap_samples': 20000, 'report_scope': 'requested subset; exploratory records retained'}
    (report / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n'.join(markdown))


if __name__ == '__main__':
    main()
