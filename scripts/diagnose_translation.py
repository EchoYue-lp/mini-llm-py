#!/usr/bin/env python3
"""
诊断翻译模型问题
"""
import sys

import torch
from transformers import GPT2TokenizerFast
from models.transformer_models import EncoderDecoderModel
from utils.translation_utils import greedy_translate, beam_search_translate

def diagnose_model(checkpoint_path):
    """诊断模型问题"""
    print(f"\n{'='*60}")
    print(f"诊断模型: {checkpoint_path}")
    print(f"{'='*60}\n")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 检查配置
    config = checkpoint.get('config', {})
    print("📋 模型配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 检查训练状态
    print(f"\n📊 训练状态:")
    print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'Unknown')}")

    # 加载模型
    device = 'cpu'
    model = EncoderDecoderModel(
        config['src_vocab_size'],
        config['tgt_vocab_size'],
        config['d_model'],
        config['num_layers'],
        config['num_heads'],
        config['d_ff'],
        config['max_len'],
        config.get('dropout', 0.1)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 加载 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("tokenization/gpt2")

    # 测试翻译
    test_sentences = [
        "Hello",
        "Good morning",
        "How are you?",
        "I love you",
        "Thank you very much"
    ]

    print(f"\n🧪 翻译测试:\n")

    for src in test_sentences:
        src_ids = tokenizer.encode(src, add_special_tokens=False)

        # Greedy
        greedy_result = greedy_translate(model, src_ids, tokenizer, max_len=50, device=device)

        # Beam search
        beam_result = beam_search_translate(model, src_ids, tokenizer, beam_width=5, max_len=50, device=device)

        print(f"  源: {src}")
        print(f"    Greedy: {greedy_result if greedy_result else '(空)'}")
        print(f"    Beam:   {beam_result if beam_result else '(空)'}")
        print()

    # 检查模型权重
    print(f"📈 模型权重统计:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            mean = param.data.mean().item()
            std = param.data.std().item()
            if abs(mean) > 10 or std > 10:
                print(f"  ⚠️  {name}: mean={mean:.4f}, std={std:.4f} (可能异常)")

    # 检查输出分布
    print(f"\n🔍 输出分布检查:")
    src = "Hello world"
    src_ids = tokenizer.encode(src, add_special_tokens=False)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    bos_id = tokenizer.eos_token_id  # GPT2 用 EOS 作为 BOS
    tgt_tensor = torch.tensor([bos_id], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(src_tensor, tgt_tensor)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top5_probs, top5_ids = probs.topk(5)

        print(f"  源: {src}")
        print(f"  Top-5 预测:")
        for prob, idx in zip(top5_probs, top5_ids):
            token = tokenizer.decode([idx.item()])
            print(f"    {token:20s} ({prob.item()*100:.2f}%)")

    # 诊断结论
    print(f"\n{'='*60}")
    print("🔬 诊断结论:")
    print(f"{'='*60}")

    val_loss = checkpoint.get('val_loss', float('inf'))
    epoch = checkpoint.get('epoch', 0)

    issues = []

    if val_loss > 5.0:
        issues.append(f"验证损失过高 ({val_loss:.2f})，模型可能未充分训练")

    if epoch < 5:
        issues.append(f"训练轮数太少 ({epoch})，建议至少训练 10-20 epochs")

    if config.get('max_len', 0) < 96:
        issues.append(f"max_len 太小 ({config.get('max_len')})")

    if issues:
        print("❌ 发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ 未发现明显问题，可能需要更多训练")

    print("\n💡 建议:")
    print("  1. 确保训练至少 20-50 epochs")
    print("  2. 检查训练损失是否持续下降")
    print("  3. 验证损失应该 < 2.0 才有较好翻译能力")
    print("  4. 使用 TensorBoard 查看训练曲线")
    print(f"  5. 当前 batch_size={config.get('batch_size', 'unknown')}, 可以尝试调整")

if __name__ == "__main__":
    import glob

    # 查找所有检查点
    checkpoints = glob.glob("encoder_decoder*.pt") + glob.glob("*.pt")
    checkpoints = [c for c in checkpoints if 'encoder' in c or 'decoder' in c]

    if not checkpoints:
        print("❌ 未找到模型检查点文件")
        print("\n请确保在项目根目录运行，或指定检查点路径:")
        print("  python -m scripts.diagnose_translation <checkpoint_path>")
        sys.exit(1)

    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        print("找到的检查点:")
        for i, cp in enumerate(checkpoints, 1):
            print(f"  {i}. {cp}")
        checkpoint_path = checkpoints[0]
        print(f"\n使用: {checkpoint_path}")

    diagnose_model(checkpoint_path)
