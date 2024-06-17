from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# Mount the static directory to serve CSS, JS, images, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ChatBot", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("card_chat.html", {"request": request})

@app.post("/get", response_class=HTMLResponse)
async def chat(request: Request, msg: str = Form(...)):
    response = get_chat_response(msg)
    return response

def get_chat_response(text: str):
    chat_history_ids = None
    for step in range(5):
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

