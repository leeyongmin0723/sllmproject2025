from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")

@app.post("/generate")
async def generate_text(input_data: TextInput):
    inputs = tokenizer(input_data.text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": result}
