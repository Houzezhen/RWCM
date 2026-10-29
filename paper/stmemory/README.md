# STMemory 论文（当前唯一工作稿）

英文全文：`stmemory_world_models_en.tex` → `stmemory_world_models_en.pdf`（11 页）。
本目录由 2026-10-24 重构版（原 `revision_20261008_value/redesign_20261024`）整理而来，已不依赖任何旧稿目录。旧稿统一归档在项目 `_archive/cleanup_20261026/paper/`。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `stmemory_world_models_en.tex` / `.pdf` | 正文源码与编译结果 |
| `refs.bib` | 参考文献 |
| `figures/` | 正文 6 张图和附录 1 张图（PDF 矢量，PNG 600 dpi），均由脚本或 `figure_src/` 生成 |
| `figure_src/` | Fig. 1 分类图、Fig. 2 框图的 TikZ 源码 |
| `tables/` | 由 `tools/make_tables.py` 自动生成的表格内容，勿手改 |
| `data/results.json` | 正文全部数值，逐项记录来源文件（由 `tools/build_data.py` 生成） |
| `data/source/` | 不在 RWCM 实验目录里的输入：判别 margin、训练曲线 CSV，π*0.6 价值模型结果及其原始报告 |
| `tools/` | `build_data.py` → `make_tables.py` → `make_figures.py`；`extract_frames.py` 抽帧，`otf2ttf.py` 字体转换 |
| `assets/` | 相机帧、价值视频帧、Inter 字体 |
| `supplementary/` | 补充视频 S1、S2（由 `RWCM/scripts/make_spacetime_video_assets.py` 生成） |

其余实验数值直接读取 `RWCM/assets/results_wcm_to_spacetime/` 与 `RWCM/outputs/` 下的评估文件。

## 重建步骤

在本目录下运行（Python 用 conda base，即 `D:\Anaconda3\python.exe`）：

```
python -I tools/build_data.py
python -I tools/make_tables.py
python -I tools/make_figures.py
xelatex -output-directory=figure_src figure_src/fig01_taxonomy.tex      # 仅在修改框图时需要
xelatex -output-directory=figure_src figure_src/fig02_architecture.tex
copy figure_src\fig0*.pdf figures\
latexmk -pdf -outdir=build stmemory_world_models_en.tex
copy build\stmemory_world_models_en.pdf .
```

`build/` 与 `figure_src/` 下的 aux/log/pdf 为编译中间文件，可随时删除。

## π*0.6 价值模型对比（2026-10-25 加入）

- 数据：`data/source/pistar06_value_ood.json`，抄自 `data/source/evo_vs_wcm_spacetime_report_20261023.md`（服务器 `outputs/evo_eval_0824_ood/`）。`build_data.py` 会用已保存的 WCM、STMemory-60 指标校验报告中的配对差值。
- 只有一个训练模型，只在 WCM 的 21,832 个 OOD 共同帧上评估，表 1 中标 †、*；分布内测试集无结果。
- 按此前要求不出现拼接（mosaic / Image-B4）对比，报告中相关行未采用。
- 正文改动：4.1 Baselines、4.2 Main results 新增一段、表 1 和图 3 新增一行、贡献点、相关工作、结论各一句、附录 A 增加实现细节；`refs.bib` 用 arXiv:2511.14759 替换了原先未被引用且题名有误的 `pi06` 条目，新增 SigLIP、Gemma 3。


## 对批注的处理

- 标题改为 `STMemory: Fixed-Query Spatiotemporal Memory for World Models`。
- 引言：先抛出问题，再介绍现有三类做法（配分类图 Fig. 1），然后说明本文方法为何重要，最后列贡献点。Related Work 单独成节，分三段。
- 方法：只讲数学原理，不出现具体数字。小公式放在段落里，大公式单独成行并编号，共 8 个。偏移列表移到附录 B，结果判别公式移到方法 3.5 节。
- 实验：依次为 Experimental Setup（数据集、基线、指标、实现细节）、主结果、跨度消融、效率、结果判别、定性结果，不再单列 Results 章。
- 数字多的地方改用表格（共 4 张正文表、3 张附录表），正文只保留关键数字。超参数和硬件信息放进附录表格。
- 不再提具体种子编号和每个种子的 batch，统一写成“三个随机种子”。内部代号（0722、0814、5cut、0703、H60、T8 等）全部改为正式描述；模型命名为 STMemory-60 和 STMemory-120。
- 去掉所有兜底性和防御性措辞、斜杠、箭头、多余的冒号和 `\textsc` 字体。图注缩短到两三句。
- 新增与其他方法的对比：VIP 和 IQL critic，二者使用相同的 8 帧、跨度 120 的历史输入。
- 按你的要求，不纳入 TD-MPC2，也不再出现输入拼接（mosaic）版本，Sparse4 消融同样属于拼接方案，一并不用。

## 数值说明

- 所有数字均来自 RWCM 中已有的评估文件，没有重算模型输出。AUROC 由已保存的逐回合价值计算得到。
- 原始 WCM 只在 21,832 个共同帧上有定义，其余方法使用 22,082 帧。VIP、IQL 和 STMemory-120 在分布内测试集上都只有单种子结果，表中用 * 标出。
- 判别实验只使用价值输出。`risk_accuracy` 中 STMemory 在 f0 到 f075 各进度点的结果来自单独训练的 risk head，不属于“仅用价值”的证据，因此没有采用。

## 还缺的数据（补齐后可直接写入）

1. **模块消融**（批注“到底是哪个模块在起作用”）。tex 中已留好表格模板（搜索 `tab:modules`），建议在 OOD 集上各跑三个种子：
   - 去掉时序注意力层
   - 用均值池化代替查询压缩
   - 查询数 R = 16 和 128
   - 观测数 T = 4（跨度 120）
   - 只用主相机
2. **分布内测试集上 STMemory-120 的 seed 3072 和 seed 42 结果**（`5cut_h120_vs_*` 对比 JSON）。补上后可以去掉单种子标记。
3. **逐帧预测曲线** `episode_curves.json`（在服务器 `outputs/eval_spacetime_t120_pair/.../episode_curves/` 下）。有了它可以画“预测价值 vs 真实回报”曲线，替换 Fig. 6 中带视频叠加层的帧。
4. （可选）VIP 和 IQL 的另外两个种子，使对比表全部为三种子。
