"""
数据集下载脚本

支持下载：
- IWSLT2017: 英中翻译数据集
- WikiText-2: 文本生成数据集
"""

import os
import argparse
from datasets import load_dataset


def download_and_save_wikitext2(data_dir="data/wikitext2"):
    """
    下载 WikiText-2 文本生成数据集

    Args:
        data_dir: 数据保存目录
    """
    os.makedirs(data_dir, exist_ok=True)

    print("\n正在下载 WikiText-2 数据集...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    for split in ["train", "validation", "test"]:
        output_file = os.path.join(data_dir, f"{split}.txt")
        print(f"  正在保存 {split} 数据...")

        texts = dataset[split]["text"]
        with open(output_file, "w", encoding="utf-8") as f:
            for line in texts:
                f.write(line.strip() + "\n")

        print(f"    ✓ {split}.txt: {len(texts)} 行")

    print(f"\n✓ WikiText-2 数据集已保存到 {data_dir}\n")


def download_and_save_iwslt2017(data_dir="data/iwslt2017"):
    """
    下载 IWSLT2017 英中翻译数据集

    Args:
        data_dir: 数据保存目录
    """
    os.makedirs(data_dir, exist_ok=True)

    print("\n正在下载 IWSLT2017 英中翻译数据集...")
    dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")

    for split in ["train", "validation", "test"]:
        src_file = os.path.join(data_dir, f"{split}.en.txt")
        tgt_file = os.path.join(data_dir, f"{split}.zh.txt")

        print(f"  正在保存 {split} 数据...")
        with open(src_file, "w", encoding="utf-8") as f_src, \
             open(tgt_file, "w", encoding="utf-8") as f_tgt:
            for item in dataset[split]["translation"]:
                f_src.write(item["en"].strip() + "\n")
                f_tgt.write(item["zh"].strip() + "\n")

        print(f"    ✓ {split}.en.txt: {len(dataset[split])} 条")
        print(f"    ✓ {split}.zh.txt: {len(dataset[split])} 条")

    print(f"\n✓ IWSLT2017 数据集已保存到 {data_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description="下载并保存翻译/生成数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 下载翻译数据集（IWSLT2017）
  python -m scripts.download_datasets --translation

  # 下载生成数据集（WikiText-2）
  python -m scripts.download_datasets --generation

  # 下载所有数据集
  python -m scripts.download_datasets --all
        """
    )

    parser.add_argument(
        "--translation",
        action="store_true",
        help="下载翻译数据集（IWSLT2017 英中）"
    )
    parser.add_argument(
        "--generation",
        action="store_true",
        help="下载生成数据集（WikiText-2）"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载所有数据集"
    )

    args = parser.parse_args()

    # 如果没有指定任何参数，显示帮助
    if not (args.translation or args.generation or args.all):
        parser.print_help()
        return

    # 根据参数下载对应数据集
    if args.all:
        download_and_save_iwslt2017()
        download_and_save_wikitext2()
    else:
        if args.translation:
            download_and_save_iwslt2017()
        if args.generation:
            download_and_save_wikitext2()

    print("=" * 60)
    print("数据下载完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
