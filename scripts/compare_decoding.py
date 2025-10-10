"""
对比不同解码策略的效果
演示 Greedy vs Beam Search 的区别
"""

import torch
import torch.nn.functional as F


def simulate_translation_scores():
    """
    模拟翻译 "I like cats" 的概率分布
    演示为什么 beam search 能找到更好的翻译
    """

    print("=" * 60)
    print("场景: 翻译 'I like cats'")
    print("=" * 60)

    # 模拟概率分布（简化）
    vocab = ["我", "我们", "喜欢", "很", "爱", "猫", "的", "。"]

    # 第一步: 翻译 "I"
    step1_probs = {
        "我": 0.6,
        "我们": 0.4,
    }

    # 第二步: 给定前面的词，翻译 "like"
    step2_probs = {
        ("我",): {"喜欢": 0.7, "很": 0.2, "爱": 0.1},
        ("我们",): {"喜欢": 0.5, "很": 0.3, "爱": 0.2},
    }

    # 第三步: 翻译 "cats"
    step3_probs = {
        ("我", "喜欢"): {"猫": 0.9, "的": 0.1},
        ("我", "很"): {"喜欢": 0.8, "爱": 0.2},
        ("我", "爱"): {"猫": 0.7, "的": 0.3},
    }

    print("\n📊 概率分布:")
    print("-" * 60)
    print("Step 1 (翻译 'I'):")
    for word, prob in step1_probs.items():
        print(f"  {word}: {prob:.2f}")

    print("\n🎯 Greedy Decoding:")
    print("-" * 60)

    # Greedy: 每步选最高概率
    greedy_path = []
    greedy_prob = 1.0

    # Step 1
    best_word = max(step1_probs, key=step1_probs.get)
    greedy_path.append(best_word)
    greedy_prob *= step1_probs[best_word]
    print(f"Step 1: 选择 '{best_word}' (prob={step1_probs[best_word]:.2f})")

    # Step 2
    prev = tuple(greedy_path)
    best_word = max(step2_probs[prev], key=step2_probs[prev].get)
    prob = step2_probs[prev][best_word]
    greedy_path.append(best_word)
    greedy_prob *= prob
    print(f"Step 2: 选择 '{best_word}' (prob={prob:.2f}, 累积={greedy_prob:.3f})")

    # Step 3
    prev = tuple(greedy_path)
    if prev in step3_probs:
        best_word = max(step3_probs[prev], key=step3_probs[prev].get)
        prob = step3_probs[prev][best_word]
        greedy_path.append(best_word)
        greedy_prob *= prob
        print(f"Step 3: 选择 '{best_word}' (prob={prob:.2f}, 累积={greedy_prob:.3f})")

    print(f"\n结果: {''.join(greedy_path)}")
    print(f"总概率: {greedy_prob:.4f}")

    print("\n🔍 Beam Search (k=2):")
    print("-" * 60)

    # Beam Search: 保留 top-k 候选
    beam_width = 2

    # Step 1
    beams = [(word, prob, [word]) for word, prob in
             sorted(step1_probs.items(), key=lambda x: x[1], reverse=True)[:beam_width]]

    print("Step 1: 初始化 beam")
    for word, prob, path in beams:
        print(f"  Beam: {path} (prob={prob:.2f})")

    # Step 2
    candidates = []
    for word, prob, path in beams:
        prev = tuple(path)
        if prev in step2_probs:
            for next_word, next_prob in step2_probs[prev].items():
                new_path = path + [next_word]
                new_prob = prob * next_prob
                candidates.append((next_word, new_prob, new_path))

    # 保留 top-k
    beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    print("\nStep 2: 扩展并保留 top-2")
    for word, prob, path in beams:
        print(f"  Beam: {path} (prob={prob:.3f})")

    # Step 3
    candidates = []
    for word, prob, path in beams:
        prev = tuple(path[-2:])  # 最后两个词
        if prev in step3_probs:
            for next_word, next_prob in step3_probs[prev].items():
                new_path = path + [next_word]
                new_prob = prob * next_prob
                candidates.append((next_word, new_prob, new_path))

    # 找最优
    best = max(candidates, key=lambda x: x[1])
    print("\nStep 3: 最终候选")
    for word, prob, path in sorted(candidates, key=lambda x: x[1], reverse=True):
        marker = " ✓ 最优" if path == best[2] else ""
        print(f"  {path} (prob={prob:.4f}){marker}")

    print(f"\n结果: {''.join(best[2])}")
    print(f"总概率: {best[1]:.4f}")

    print("\n" + "=" * 60)
    print("📈 对比总结:")
    print("=" * 60)
    print(f"Greedy:  {''.join(greedy_path):20s} prob={greedy_prob:.4f}")
    print(f"Beam:    {''.join(best[2]):20s} prob={best[1]:.4f}")
    print(f"提升:    {((best[1] - greedy_prob) / greedy_prob * 100):>20.1f}%")

    if best[1] > greedy_prob:
        print("\n✅ Beam Search 找到了更优的翻译！")
    else:
        print("\n➡️  在这个例子中两者相同")


def explain_beam_search():
    """解释 beam search 的工作原理"""
    print("\n" + "=" * 60)
    print("💡 Beam Search 原理")
    print("=" * 60)

    print("""
Beam Search 维护 k 个最有希望的候选序列（beam）

每一步:
1. 对每个 beam，生成所有可能的下一个词
2. 计算新序列的累积概率
3. 保留概率最高的 k 个序列

关键优势:
✓ 避免局部最优（greedy 的问题）
✓ 考虑全局序列概率
✓ 可以"纠正"早期的次优选择

时间复杂度:
- Greedy: O(T × V)
- Beam:   O(T × V × k)

其中:
- T: 序列长度
- V: 词表大小
- k: beam width
    """)

    print("\n推荐设置:")
    print("  翻译任务: beam_width = 4-6")
    print("  摘要任务: beam_width = 3-5")
    print("  代码生成: beam_width = 5-10")


if __name__ == "__main__":
    simulate_translation_scores()
    explain_beam_search()
