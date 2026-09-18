# pip install fastapi uvicorn g4f smartg4f pydantic requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import time
from pathlib import Path
from datetime import datetime
from g4f.client import Client
from g4f.Provider import RetryProvider, BaseProvider

app = FastAPI(title="Adaptive Free LLM API")
RANKING_FILE = Path("model_rankings.json")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None

def get_dynamic_providers():
    """Safely get all working providers without hardcoding names like 'Bing'"""
    providers = []
    # Try smartg4f first (handle API changes)
    try:
        import smartg4f
        if hasattr(smartg4f, 'get_fastest_providers'):
            return smartg4f.get_fastest_providers()
        elif hasattr(smartg4f, 'get_fast_providers'):
            return smartg4f.get_fast_providers()
        elif hasattr(smartg4f, 'providers'):
            return smartg4f.providers
    except Exception:
        pass

    # Fallback: dynamically inspect g4f.Provider for any working provider
    from g4f import Provider
    for name in dir(Provider):
        if not name.startswith('_'):
            try:
                obj = getattr(Provider, name)
                if isinstance(obj, type) and issubclass(obj, BaseProvider):
                    if getattr(obj, 'working', False):
                        providers.append(obj)
            except Exception:
                continue
    return providers

def get_known_models():
    """Curated list of models known to work across various free providers"""
    return [
        "deepseek-chat", "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4.1-flash", "deepseek-v4",
        "qwen-3-max", "qwen-max", "qwen2.5-72b-instruct", "qwen-turbo",
        "kimi-k2", "moonshot-v1-128k",
        "glm-4.6", "glm-4-flash", "glm-4-plus",
        "llama-3.3-70b", "llama-3.1-70b", "llama-4-scout", "llama-4-maverick",
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
        "claude-3.5-sonnet", "claude-3-opus",
        "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"
    ]

class ModelStats:
    def __init__(self):
        self.success_count = 0
        self.failure_count = 0
        self.total_latency = 0.0
        self.last_success = None
        self.last_failure = None
    
    def to_dict(self):
        total = self.success_count + self.failure_count
        return {
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_latency": self.total_latency,
            "avg_latency": self.total_latency / self.success_count if self.success_count > 0 else 0,
            "success_rate": self.success_count / total if total > 0 else 0,
            "last_success": self.last_success,
            "last_failure": self.last_failure,
        }

class AdaptiveRanker:
    def __init__(self):
        self.rankings: Dict[str, ModelStats] = {}
        self.load()
    
    def load(self):
        if RANKING_FILE.exists():
            try:
                with open(RANKING_FILE) as f:
                    data = json.load(f)
                    for model_id, stats in data.items():
                        self.rankings[model_id] = ModelStats()
                        self.rankings[model_id].success_count = stats.get("success_count", 0)
                        self.rankings[model_id].failure_count = stats.get("failure_count", 0)
                        self.rankings[model_id].total_latency = stats.get("total_latency", 0)
                        self.rankings[model_id].last_success = stats.get("last_success")
                        self.rankings[model_id].last_failure = stats.get("last_failure")
            except Exception:
                pass
    
    def save(self):
        data = {k: v.to_dict() for k, v in self.rankings.items()}
        with open(RANKING_FILE, "w") as f:
            json.dump(data, f, indent=2)
    
    def record_success(self, model_id: str, latency: float):
        if model_id not in self.rankings: self.rankings[model_id] = ModelStats()
        self.rankings[model_id].success_count += 1
        self.rankings[model_id].total_latency += latency
        self.rankings[model_id].last_success = datetime.now().isoformat()
        self.save()
    
    def record_failure(self, model_id: str):
        if model_id not in self.rankings: self.rankings[model_id] = ModelStats()
        self.rankings[model_id].failure_count += 1
        self.rankings[model_id].last_failure = datetime.now().isoformat()
        self.save()
    
    def score_model(self, model_id: str) -> float:
        if model_id not in self.rankings:
            # Baseline scores for untested models to prioritize long-context/smart models
            if "deepseek-v4" in model_id: return 600.0
            if "qwen-3" in model_id: return 550.0
            if "kimi" in model_id: return 550.0
            if "glm-4" in model_id: return 550.0
            if "llama-3" in model_id: return 500.0
            return 400.0
        
        stats = self.rankings[model_id]
        total_requests = stats.success_count + stats.failure_count
        if total_requests == 0: return 400.0
        
        success_rate = stats.success_count / total_requests
        avg_latency = stats.total_latency / stats.success_count if stats.success_count > 0 else 10.0
        
        # Score formula: success_rate * 1000 + (1 / avg_latency) * 100
        return (success_rate * 1000) + (1.0 / max(avg_latency, 0.1)) * 100
    
    def get_best_models(self, available_models: List[str], top_n: int = 30) -> List[str]:
        scored = [(m, self.score_model(m)) for m in available_models]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_n]]

ranker = AdaptiveRanker()

@app.get("/v1/models")
def list_models():
    models = get_known_models()
    ranked = ranker.get_best_models(models, top_n=len(models))
    return {"data": [{"id": m, "score": ranker.score_model(m), "stats": ranker.rankings.get(m).to_dict() if ranker.rankings.get(m) else None} for m in ranked]}

@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    all_models = get_known_models()
    best_models = ranker.get_best_models(all_models, top_n=30)
    
    if request.model and request.model != "auto":
        best_models = [request.model] + [m for m in best_models if m != request.model]
    
    providers = get_dynamic_providers()
    if not providers:
        raise HTTPException(status_code=500, detail="No providers available")
        
    client = Client(provider=RetryProvider(providers, shuffle=False), timeout=120)
    
    for model_id in best_models:
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                stream=False
            )
            latency = time.time() - start_time
            ranker.record_success(model_id, latency)
            
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response.choices[0].message.content}, "finish_reason": "stop"}],
                "_meta": {"latency": latency, "model_used": model_id, "model_score": ranker.score_model(model_id)}
            }
        except Exception as e:
            ranker.record_failure(model_id)
            print(f"Failed {model_id}: {str(e)[:100]}")
            continue
    
    raise HTTPException(status_code=503, detail="All models failed")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Adaptive Free LLM API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "health": "/health"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
def root():
    return {"message": "Adaptive Free LLM API", "docs": "/docs", "models": "/v1/models"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
