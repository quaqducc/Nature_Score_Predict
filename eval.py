import argparse
import csv
import json
from typing import Dict, List, Optional

from product_similarity.pipeline import _load_fewshot_cases, build_prompt, retrieve_contexts
from product_similarity.model import ChatAPIWrapper, LLMWrapper
from product_similarity.agents import FactorAgent, FactorAgentConfig, evaluate_multiple_factors
from product_similarity.judge import LLMJudge, JudgeConfig
from product_similarity.spsc import retrieve_spsc_contexts


DEFAULT_ANALYZER_MODEL = None  # None => only build prompt; override with HF id or chat API via CLI


def run_analyzer(product_1: str, product_2: str, contexts: List[str], *,
                 model_name: Optional[str] = DEFAULT_ANALYZER_MODEL,
                 chat_api_base_url: Optional[str] = None,
                 chat_api_key: Optional[str] = None,
                 chat_api_model: Optional[str] = None,
                 device: int = -1,
                 max_new_tokens: int = 256,
                 temperature: float = 0.0,
                 top_p: float = 1.0) -> str:
    fewshot_cases = _load_fewshot_cases()
    prompt = build_prompt(fewshot_cases, product_1, product_2, contexts, max_fewshot=5)
    if chat_api_base_url and chat_api_key and chat_api_model:
        chat = ChatAPIWrapper(
            base_url=str(chat_api_base_url),
            api_key=str(chat_api_key),
            model=str(chat_api_model),
            max_tokens=max_new_tokens,
        )
        return chat.run(prompt, temperature=max(temperature, 0.0), top_p=top_p)
    if model_name:
        llm = LLMWrapper(model_name=model_name, device=device, max_new_tokens=max_new_tokens)
        return llm.run(prompt, temperature=temperature, top_p=top_p)
    return ""


def run_agents(product_1: str, product_2: str, contexts: List[str], *,
               use_chat_api: bool = False,
               chat_api_base_url: Optional[str] = None,
               chat_api_key: Optional[str] = None,
               chat_api_model: Optional[str] = None,
               default_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
               device: int = -1,
               max_new_tokens: int = 256) -> Dict[str, Dict[str, object]]:
    # Map optional per-factor context string if desired; here we pass the same joined contexts
    shared_ctx = "\n\n".join(contexts)
    per_factor_ctx = {
        "Nature": shared_ctx,
        "Intended Purpose": shared_ctx,
        "Channel of trade": shared_ctx,
    }
    agent = FactorAgent(
        default=FactorAgentConfig(model_name=default_model, device=device, max_new_tokens=max_new_tokens),
        per_factor=None,
        use_chat_api=use_chat_api,
        chat_api_base_url=chat_api_base_url,
        chat_api_key=chat_api_key,
        chat_api_model=chat_api_model,
    )
    factors = ["Nature", "Intended Purpose", "Channel of trade"]
    return evaluate_multiple_factors(agent, product_1, product_2, factors, per_factor_ctx)


def evaluate_dataset(csv_path: str, *,
                     model_name: Optional[str] = DEFAULT_ANALYZER_MODEL,
                     agent_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
                     chat_api_base_url: Optional[str] = None,
                     chat_api_key: Optional[str] = None,
                     chat_api_model: Optional[str] = None,
                     device: int = -1,
                     max_new_tokens: int = 256,
                     include_spsc: bool = True,
                     spsc_top_k: int = 2) -> Dict[str, object]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    judge = LLMJudge(JudgeConfig(weights={"Nature": 0.5, "Intended Purpose": 0.5, "Channel of trade": 0.0}))

    results = []
    correct = 0
    total = 0

    for r in rows:
        p1 = r.get("Item 1", "").strip()
        p2 = r.get("Item 2", "").strip()
        gold_overall = r.get("Level of similarity")
        try:
            gold = int(float(gold_overall)) if gold_overall not in (None, "", "-") else None
        except Exception:
            gold = None

        contexts = retrieve_contexts(p1, p2, top_k=3)
        if include_spsc:
            try:
                spsc_ctx = retrieve_spsc_contexts(p1, p2, top_k=spsc_top_k)
                if spsc_ctx:
                    contexts = contexts + spsc_ctx
            except Exception:
                pass
        analyzer_text = run_analyzer(
            p1,
            p2,
            contexts,
            model_name=model_name,
            chat_api_base_url=chat_api_base_url,
            chat_api_key=chat_api_key,
            chat_api_model=chat_api_model,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        factor_outputs = run_agents(
            p1,
            p2,
            contexts,
            use_chat_api=bool(chat_api_base_url and chat_api_key and chat_api_model),
            chat_api_base_url=chat_api_base_url,
            chat_api_key=chat_api_key,
            chat_api_model=chat_api_model,
            default_model=agent_model,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        judged = judge.combine_factor_scores(factor_outputs)

        pred = int(judged.get("overall_similarity", 0))
        if gold is not None:
            total += 1
            if pred == gold:
                correct += 1

        results.append({
            "product_1": p1,
            "product_2": p2,
            "contexts": contexts,
            "analyzer": analyzer_text,
            "factors": factor_outputs,
            "judge": judged,
            "gold_overall": gold,
            "pred_overall": pred,
        })

    metrics = {
        "total_labeled": total,
        "exact_match": (correct / total) if total > 0 else None,
    }
    return {"metrics": metrics, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-agent evaluation for product similarity")
    parser.add_argument("--csv", default="data/100_samples.csv", help="CSV dataset path")
    parser.add_argument("--analyzer-model", default=None, help="HF model id for analyzer (or empty to skip)")
    parser.add_argument("--agent-model", default="mistralai/Mistral-7B-Instruct-v0.2", help="HF model id for factor agents")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-spsc", action="store_true", help="Disable adding SPSC context")
    parser.add_argument("--spsc-top-k", type=int, default=2)
    parser.add_argument("--chat-api-base-url", default=None)
    parser.add_argument("--chat-api-key", default=None)
    parser.add_argument("--chat-api-model", default=None)
    args = parser.parse_args()

    out = evaluate_dataset(
        args.csv,
        model_name=args.analyzer_model or None,
        agent_model=args.agent_model,
        chat_api_base_url=args.chat_api_base_url,
        chat_api_key=args.chat_api_key,
        chat_api_model=args.chat_api_model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        include_spsc=(not args.no_spsc),
        spsc_top_k=args.spsc_top_k,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


