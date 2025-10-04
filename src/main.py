from orbitsAPI.src.rag.rag import RAG
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

rag = RAG()

# cria o app
app = FastAPI()

# libera acesso do frontend React (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção, troque "*" pelo domínio do seu site
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# modelo do que o frontend vai mandar
class Query(BaseModel):
    question: str

# rota principal da API
@app.post("/api/query")
def ask_model(query: Query):
    response = rag.prompt(query.question)
    # aqui futuramente colocaremos o RAG + modelo LLM
    return {"answer": response}

# rota simples de teste
@app.get("/")
def root():
    return {"message": "API da Space Biology está rodando 🚀"}
