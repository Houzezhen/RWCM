"""Summarize full-finetuned replacements and 16-query model against paper H120."""
import argparse
import csv
import json
from pathlib import Path
from scripts.compare_experiments import load, curve_path, compare_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    args=p.parse_args()
    root=Path(args.root)
    reference=curve_path('outputs/eval_spacetime_t120_pair/ood/t120_s42')
    names=['replace_a_sampling','replace_b_encoding','replace_c_readout','queries16']
    paths={n:curve_path(str(root/f'{n}_s42/eval_ood')) for n in names}
    predictions={n:load(path) for n,path in paths.items()}
    full=load(reference)
    keys=set(full)
    for rows in predictions.values():
        keys &= set(rows)
    manifest=root/'endpoint_manifest.json'
    manifest.write_text(json.dumps({'endpoints':[{'episode_id':e,'frame_index':f} for e,f in sorted(keys)]}))
    table=[]
    for name in names:
        r=compare_rows(full,predictions[name],baseline_path=reference,candidate_path=paths[name],
                       bootstrap_samples=20000,seed=42,align='common',endpoint_keys=keys,endpoint_manifest=manifest)
        (root/f'{name}_vs_paper.json').write_text(json.dumps(r,indent=2))
        if not table:
            table.append({'method':'Paper STMemory-H120 Q64','MSE':r['baseline_mse'],
                          'Pearson':r['baseline_pearson'],'centered_MSE':r['baseline_centered_mse'],
                          'bias':r['baseline_mean_bias'],'delta_MSE_CI95':''})
        table.append({'method':name+' + full3','MSE':r['candidate_mse'],'Pearson':r['candidate_pearson'],
                      'centered_MSE':r['candidate_centered_mse'],'bias':r['candidate_mean_bias'],
                      'delta_MSE_CI95':str(r['mse_paired_episode_bootstrap_ci95'])})
    with (root/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as h:
        w=csv.DictWriter(h,fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    headers=list(table[0])
    lines=['# 模块替换 full 微调与16-query结果','',f'共同 {len(keys)} OOD帧，seed42。',
           '', '| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for row in table:
        lines.append('| '+' | '.join(f'{row[k]:.6f}' if isinstance(row[k],float) else str(row[k]) for k in headers)+' |')
    lines+=['','新模型均为joint10+full3，last-only；论文完整模型使用历史验证最优权重，选模方式不同。',
            'a/b/c从对应已有joint最后权重继续full3；Q16从公共上游独立训练joint10+full3，不裁剪Q64已学查询。',
            'ΔMSE为候选减论文完整模型，区间用20000次paired episode bootstrap。']
    (root/'full与16query报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
