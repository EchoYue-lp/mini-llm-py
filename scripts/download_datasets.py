import os
from datasets import load_dataset

def download_and_save_wikitext2(data_dir="data/wikitext2"):
    os.makedirs(data_dir, exist_ok=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    for split in ["train", "validation", "test"]:
        texts = dataset[split]["text"]
        with open(os.path.join(data_dir, f"{split}.txt"), "w", encoding="utf-8") as f:
            for line in texts:
                f.write(line.strip() + "\n")
    print(f"WikiText-2 数据集已保存到 {data_dir}")

def download_and_save_iwslt2017(data_dir="data/iwslt2017"):
    os.makedirs(data_dir, exist_ok=True)
    dataset = load_dataset("opus100", "en-zh")
    for split in ["train", "validation", "test"]:
        src_file = os.path.join(data_dir, f"{split}.en.txt")
        tgt_file = os.path.join(data_dir, f"{split}.zh.txt")
        with open(src_file, "w", encoding="utf-8") as f_src, open(tgt_file, "w", encoding="utf-8") as f_tgt:
            for item in dataset[split]:
                f_src.write(item["translation"]["en"].strip() + "\n")
                f_tgt.write(item["translation"]["zh"].strip() + "\n")
    print(f"Opus100 EN-ZH 数据集已保存到 {data_dir}")

if __name__ == "__main__":
    download_and_save_wikitext2()
    download_and_save_iwslt2017()
