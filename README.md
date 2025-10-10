# LLM Transformer 项目

本项目实现了基于 PyTorch 的 Transformer（LLM/Decoder-only 和 Encoder-Decoder）模型，支持端到端训练、推理和评测。

## 依赖安装

```bash
pip install -r requirements.txt
```

## 数据准备

1. 下载数据和分词器：
   ```bash
   python scripts/download_datasets.py
   python tokenization/build_gpt2_tokenizer.py
   ```
2. 数据预处理：
   ```bash
   python scripts/preprocess.py
   # 如需为翻译目标序列添加BOS/EOS：
   python scripts/add_bos_eos.py
   ```

## 训练

- 训练 LLM/Decoder-only（语言建模）：
  ```bash
  python scripts/train_decoder.py
  ```
- 训练 Encoder-Decoder（翻译）：
  ```bash
  python scripts/train_encoder_decoder.py
  ```

## 推理与评测

- 交互式生成：
  ```bash
  python scripts/generate.py
  ```
- 交互式翻译：
  ```bash
  python scripts/translate.py
  ```
- Notebook 端到端评测：
  运行 notebooks/TranslationDemo.ipynb 查看 BLEU 分数

## 目录结构
- data/           # 数据集及预处理结果
- models/         # 模型结构实现
- scripts/        # 训练、推理、预处理脚本
- tokenization/   # 分词器
- utils/          # 工具函数
- notebooks/      # 交互式实验与评测

## 备注
- 推荐在 Mac M1/CPU/GPU 环境下运行，支持 MPS 加速。
- 训练参数、模型结构等可在脚本内灵活调整。
# mini-llm-py
