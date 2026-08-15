import torch
from models.transformer_models import DecoderOnlyModel
from utils.mask_utils import create_causal_mask
from utils.checkpoint_utils import load_model_from_checkpoint
from utils.tokenizer_utils import load_gpt2_tokenizer

def greedy_generate(model, input_ids, tokenizer, max_len=50, device="cpu"):
    if input_ids is None or len(input_ids) == 0:
        raise ValueError("input_ids must contain at least one token")
    if max_len < 0:
        raise ValueError("max_len (generated token count) must be non-negative")
    if len(input_ids) + max_len > model.max_len:
        raise ValueError("prompt plus generated tokens exceeds model max_len")

    tokens = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(max_len):
                mask = create_causal_mask(tokens.size(1), device=device)
                logits, _ = model(tokens, mask=mask)
                next_token = logits[:, -1, :].argmax(-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
    finally:
        model.train(was_training)
    return tokens.squeeze(0).tolist()

def generate_text(prompt, model, tokenizer, device="cpu", max_len=50):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    output_ids = greedy_generate(model, input_ids, tokenizer, max_len=max_len, device=device)
    # 截断到EOS
    if tokenizer.eos_token_id in output_ids:
        output_ids = output_ids[:output_ids.index(tokenizer.eos_token_id)]
    return tokenizer.decode(output_ids, skip_special_tokens=True)

def main():
    model_path = "decoder_only_best.pt"  # 使用最佳模型
    tokenizer_dir = "tokenization/gpt2"
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_gpt2_tokenizer(tokenizer_dir)

    print(f"正在从 {model_path} 加载模型...")
    print(f"使用设备: {device}\n")

    # 使用 checkpoint_utils 自动加载模型配置
    model, checkpoint_info = load_model_from_checkpoint(
        model_path,
        DecoderOnlyModel,
        model_type='decoder',
        device=device
    )

    # 如果 checkpoint 中没有保存配置，使用默认值
    if not checkpoint_info:
        print("警告：使用默认配置，如果训练时修改了配置，生成结果可能不正确")

    print("\n请输入 prompt，回车生成，输入 exit 退出：")
    while True:
        prompt = input("Prompt> ").strip()
        if prompt.lower() == "exit":
            break
        gen = generate_text(prompt, model, tokenizer, device=device)
        print("Output>", gen)

if __name__ == "__main__":
    main()
