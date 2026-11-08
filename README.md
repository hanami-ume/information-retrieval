# TDT3 命令行搜索引擎（信息检索大作业 2026）

从零实现的命令行搜索引擎，为 TDT3 语料的 37,526 篇新闻建立位置倒排索引，支持自由文本与双引号短语混合查询，相关性打分（BM25、TF-IDF lnc.ltc）均为手写实现。搜索引擎只依赖 Python 3 标准库。

- 技术报告：[`report/main.pdf`](report/main.pdf)（LaTeX 源文件 `report/main.tex`）
- 作业讲义：`信息检索大作业2026.pdf`

## 快速开始

```bash
unzip -q tdt3.zip -x "__MACOSX/*"      # 解压语料到 ./tdt3
python3 src/search.py --build          # 建索引（约 22 秒，写入 index/tdt3.idx）
```

交互模式（推荐，双引号短语可直接输入）：

```bash
python3 src/search.py
# search --hits=10 hurricane
# search "newt gingrich" down
# search --model=tfidf nba strike closed-door
# quit
```

一次性查询（shell 会剥去双引号，短语需外加单引号）：

```bash
./search --hits=27 new york san francisco
./search '"new york city"' apple cider
```

## 选项

| 选项 | 说明 |
|---|---|
| `--hits=N` | 返回前 N 条结果（默认 10） |
| `--model=bm25\|tfidf` | 打分函数（默认 bm25） |
| `--k1`, `--b` | BM25 参数（默认 1.5 / 0.75） |
| `--strict-phrase` | 引号短语改为硬匹配（默认软匹配） |
| `--hyphen=phrase\|phrase+parts\|parts` | 连字符词的处理方式（默认 phrase） |
| `--no-stem` | 与 `--build` 合用，建不做词干还原的索引 |
| `--color=auto\|always\|never` | 命中词高亮 |

## 目录结构

```
src/search.py            命令行入口（一次性查询 / 交互模式 / 建索引）
src/irsearch/porter.py   Porter 词干算法（依原论文实现）
src/irsearch/analyzer.py 分词与规范化（小写、数字、缩写、撇号、连字符）
src/irsearch/corpus.py   读取 SGML 文档
src/irsearch/index.py    位置倒排索引的构建、保存与加载
src/irsearch/query.py    查询解析（短语软匹配、连字符处理）
src/irsearch/scoring.py  BM25 与 TF-IDF（lnc.ltc）打分
src/irsearch/snippet.py  查询相关摘要与高亮
src/irsearch/engine.py   检索流程封装
experiments/             复现报告中全部数据、图表与截图的脚本
report/                  技术报告（main.tex、main.pdf、figures/）
```

## 复现实验

```bash
pip install matplotlib nltk            # 仅实验脚本需要；nltk 只用于校验 Porter 实现
python3 experiments/run_experiments.py # 统计数据 → experiments/results/
python3 experiments/make_figures.py    # 图表 → report/figures/
python3 experiments/render_screens.py  # 控制台截图（需要 Node.js + Playwright）
cd report && xelatex main.tex && xelatex main.tex
```
