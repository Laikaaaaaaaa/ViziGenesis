#!/usr/bin/env python3
"""
ViziGenesis — Financial Chat Inference Server
==============================================
Serves the fine-tuned financial LLM with RAG-enhanced responses.
Acts as a mini financial ChatGPT with real data grounding.

Supports two modes:
1. LOCAL: Runs the fine-tuned model locally (requires GPU)
2. DEMO: Uses template-based responses + RAG data (no GPU needed)

Run:
    python -m backend.fin_llm.chat                     # local model
    python -m backend.fin_llm.chat --mode demo          # demo mode
    python -m backend.fin_llm.chat --mode api           # as API server

API endpoints (when --mode api):
    POST /chat  {"message": "..."}  → {"response": "..."}
    GET  /health                    → {"status": "ok"}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.fin_llm.rag import FinancialRAG, get_rag


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are ViziGenesis, an expert AI financial analyst powered by comprehensive market data. You have deep knowledge of:

- Macroeconomics: GDP, CPI, PCE, employment, Fed policy, interest rates, money supply
- Stock Markets: US equities, international markets, Vietnam (VN-Index), ETFs
- Technical Analysis: RSI, MACD, Bollinger Bands, moving averages, volume analysis
- Corporate Fundamentals: P/E, revenue, margins, EPS, cash flow, debt ratios
- Cross-Asset Correlations: Bonds, commodities, FX, crypto, sector rotation
- Risk Management: Portfolio construction, hedging, position sizing

Rules:
1. Always ground your analysis in specific data when available
2. Provide actionable trading implications
3. Cite specific indicators, tickers, and price levels
4. Discuss both bull and bear scenarios
5. Include risk management considerations
6. Reference historical parallels when relevant
7. For major news, reason via transmission channels: policy/rates -> USD/yields -> credit/liquidity -> earnings -> positioning
8. Provide scenario probabilities (base/bull/bear) and clear invalidation signals"""


# ══════════════════════════════════════════════════════════════
#  LOCAL MODEL INFERENCE
# ══════════════════════════════════════════════════════════════

class LocalModelChat:
    """Chat with the locally fine-tuned model."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(ROOT / "models" / "fin_llm" / "final")
        self.model = None
        self.tokenizer = None
        self.rag = get_rag()

    def load(self) -> None:
        """Load the fine-tuned model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        model_dir = Path(self.model_path)

        # Check if this is a LoRA adapter or full model
        if (model_dir / "adapter_config.json").exists():
            # LoRA adapter — load base + adapter
            with open(model_dir / "adapter_config.json") as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path", "")

            print(f"Loading base model: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            print(f"Loading LoRA adapter: {model_dir}")
            self.model = PeftModel.from_pretrained(base_model, str(model_dir))
        elif (model_dir / "merged").exists():
            # Merged model
            print(f"Loading merged model: {model_dir / 'merged'}")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir / "merged"),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            print(f"Loading model from: {model_dir}")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        self.rag.load()
        print("Model and RAG loaded successfully!")

    def generate(self, message: str, max_new_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generate a response to a user message."""
        import torch

        if self.model is None:
            self.load()

        # RAG: retrieve relevant context
        context = self.rag.build_context_string(message, max_tokens=1000)

        # Build prompt with context
        system = SYSTEM_PROMPT
        if context:
            system += f"\n\nRelevant data for this query:\n{context}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ]

        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            input_text = f"### System:\n{system}\n\n### User:\n{message}\n\n### Response:\n"

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()


# ══════════════════════════════════════════════════════════════
#  DEMO MODE (no GPU required)
# ══════════════════════════════════════════════════════════════

class DemoChat:
    """
    Demo mode that uses RAG data + template responses.
    Works without a GPU — good for testing the pipeline.
    """

    def __init__(self):
        self.rag = get_rag()

    def load(self) -> None:
        self.rag.load()

    def generate(self, message: str, **kwargs) -> str:
        """Generate a response using RAG data + templates."""
        self.rag.load()

        # Retrieve context
        contexts = self.rag.retrieve(message)
        context_str = self.rag.build_context_string(message)

        if not contexts:
            return (
                "I don't have specific data relevant to your query in my database. "
                "Please try asking about specific stocks (e.g., AAPL, NVDA), "
                "macro indicators (CPI, GDP, unemployment), or market analysis topics."
            )

        # Build response from retrieved data
        parts = [f"Based on the data in my financial database:\n"]

        for ctx in contexts[:5]:
            data = ctx["data"]
            ctype = ctx["type"]

            if ctype == "stock":
                sym = data.get("symbol", "")
                parts.append(
                    f"**{sym}**: Trading at ${data.get('latest_price', 0):.2f} "
                    f"(daily: {data.get('daily_return_pct', 0):+.2f}%). "
                    f"52-week range: ${data.get('52w_low', 0):.2f}-${data.get('52w_high', 0):.2f}. "
                    f"Period returns: {data.get('returns', {})}."
                )

            elif ctype == "fundamentals":
                parts.append(data.get("text", ""))

            elif ctype == "macro":
                ind = data.get("indicator", "")
                parts.append(
                    f"**{ind.replace('_', ' ').title()}**: {data.get('latest_value', 0):.4g} "
                    f"({data.get('pct_change', 0):+.2f}% change, {data.get('data_points', 0)} observations "
                    f"from {data.get('history_start', 'N/A')})."
                )

            elif ctype == "narrative":
                parts.append(f"\n{data.get('text', '')}")

            elif ctype == "news":
                parts.append(f"📰 {data.get('title', '')} ({data.get('date', '')})")

        return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════
#  API SERVER MODE
# ══════════════════════════════════════════════════════════════

def create_api_app(chat_engine):
    """Create a FastAPI app for the chat endpoint."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(title="ViziGenesis Financial AI", version="3.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class ChatRequest(BaseModel):
        message: str
        max_tokens: int = 1024
        temperature: float = 0.7

    class ChatResponse(BaseModel):
        response: str
        context_used: int = 0

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "vizigenesis-fin-llm"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        response = chat_engine.generate(req.message, max_new_tokens=req.max_tokens, temperature=req.temperature)
        rag = get_rag()
        ctx_count = len(rag.retrieve(req.message))
        return ChatResponse(response=response, context_used=ctx_count)

    @app.get("/data/summary")
    def data_summary():
        rag = get_rag()
        rag.load()
        return {
            "macro_indicators": len(rag._macro_cache),
            "stocks": len(rag._stock_cache),
            "fundamentals": len(rag._fundamentals_cache),
            "news_articles": len(rag._news),
            "narratives": len(rag._narratives),
        }

    return app


# ══════════════════════════════════════════════════════════════
#  INTERACTIVE CLI
# ══════════════════════════════════════════════════════════════

def interactive_chat(chat_engine) -> None:
    """Run an interactive chat session in the terminal."""
    print("\n" + "="*60)
    print("  ViziGenesis Financial AI Chat")
    print("  Type 'quit' or 'exit' to stop")
    print("  Type 'data' to see available data summary")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "data":
            rag = get_rag()
            rag.load()
            print(f"\n📊 Data Summary:")
            print(f"  Macro indicators: {len(rag._macro_cache)}")
            print(f"  Stocks: {len(rag._stock_cache)}")
            print(f"  Fundamentals: {len(rag._fundamentals_cache)}")
            print(f"  News articles: {len(rag._news)}")
            print(f"  Analysis narratives: {len(rag._narratives)}")
            continue

        response = chat_engine.generate(user_input)
        print(f"\n🤖 ViziGenesis: {response}")


# ══════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ViziGenesis Financial AI Chat")
    parser.add_argument("--mode", choices=["local", "demo", "api"], default="demo",
                       help="local=GPU model, demo=template+RAG, api=HTTP server")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to fine-tuned model (for local mode)")
    parser.add_argument("--port", type=int, default=8100,
                       help="API server port (for api mode)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="API server host")

    args = parser.parse_args()

    if args.mode == "local":
        chat = LocalModelChat(args.model_path)
        chat.load()
    else:
        chat = DemoChat()
        chat.load()

    if args.mode == "api":
        import uvicorn
        app = create_api_app(chat)
        print(f"\nStarting API server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        interactive_chat(chat)


if __name__ == "__main__":
    main()
