from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok = AutoTokenizer.from_pretrained('/data/nishant/Nishant/Prashant/Project/models/mistral-7b-instruct')
model = AutoModelForCausalLM.from_pretrained('/data/nishant/Nishant/Prashant/Project/models/mistral-7b-instruct', torch_dtype=torch.bfloat16, device_map='auto')
model.eval()

raw = 'The capital of France is Paris.\n\nQuestion: What is the capital of France?\nAnswer:'
messages = [{"role": "user", "content": raw}]
chat_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Chat string:", repr(chat_str))
inputs = tok(chat_str, return_tensors='pt').to('cuda')

with torch.inference_mode():
    out = model.generate(inputs.input_ids, max_new_tokens=30, do_sample=False)

new_toks = out[0, inputs.input_ids.size(1):]
print('Generated text:', repr(tok.decode(new_toks, skip_special_tokens=True)))
