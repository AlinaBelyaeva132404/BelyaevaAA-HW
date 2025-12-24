# Генерация текста
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("my_model")
tokenizer = GPT2Tokenizer.from_pretrained("my_model")
tokenizer.pad_token = tokenizer.eos_token

prompt = "Это лучший товар потому что"
inputs = tokenizer(prompt, return_tensors="pt")

out = model.generate(
    **inputs,
    max_length=60,
    do_sample=True,
    top_p=0.9,
    temperature=0.7
)

print(tokenizer.decode(out[0], skip_special_tokens=True))
