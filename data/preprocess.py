def preprocess_asm_function(text: str) -> str:
    # 示例：空格分隔操作码与操作数，语句分号连接
    lines = text.strip().split('\n')
    processed = [' '.join(line.strip().split()) for line in lines if line.strip()]
    return ' ; '.join(processed)
