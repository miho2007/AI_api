from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

MODEL_NAME = "distilgpt2"

# Load once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

class Prompt(BaseModel):
    prompt: str
    max_new_tokens: int = 48
    temperature: float = 0.8

@app.post("/generate")
def generate_text(data: Prompt):
    inputs = tokenizer(
        data.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=min(data.max_new_tokens, 64),
            temperature=data.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": text}
