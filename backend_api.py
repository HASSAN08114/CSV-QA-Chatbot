from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from agents_handler.agents import get_dynamic_agent, get_rag_agent
import tempfile

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = None
rag_agent = None

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    global df, rag_agent
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp.write(await file.read())
    temp.close()
    df = pd.read_csv(temp.name)
    llm_type = "gemini"
    api_key = "YOUR_GEMINI_API_KEY"  # <-- Replace with your Gemini API key
    rag_agent = get_rag_agent(df, get_dynamic_agent(df, llm_type=llm_type, api_key=api_key))
    return {"status": "CSV uploaded and RAG agent ready", "columns": list(df.columns)}

@app.post("/chat")
async def chat(request: Request):
    global rag_agent
    data = await request.json()
    user_message = data.get("message", "")
    if rag_agent is None:
        return {"response": "Please upload a CSV first."}
    response = rag_agent(user_message)
    return {"response": response}