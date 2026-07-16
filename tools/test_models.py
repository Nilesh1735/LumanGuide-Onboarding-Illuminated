import sys, time, json
sys.path.append(r'C:\Users\LOQ\OneDrive\Desktop\Adaptive-Rag-main')
from src.llms import openai as llm_conf
from langchain_google_genai import ChatGoogleGenerativeAI

candidates = [
    'gemini-1.5-flash',
    'gemini-1.5',
    'gemini-1.5-pro',
    'gemini-1.0',
    'text-bison@001',
    'text-bison',
    'chat-bison',
]

results = []
for model in candidates:
    try:
        client = ChatGoogleGenerativeAI(model=model, google_api_key=llm_conf.GOOGLE_API_KEY)
        ok = False
        detail = None
        start = time.time()
        try:
            resp = client.invoke([{"role":"user","content":"Hello"}])
            ok = True
            detail = 'invoked'
        except Exception as e:
            detail = str(e)
        elapsed = time.time() - start
        results.append({"model": model, "ok": ok, "detail": detail, "elapsed": elapsed})
    except Exception as e:
        results.append({"model": model, "ok": False, "detail": str(e)})

print(json.dumps(results, indent=2))
