import argparse
import csv
import json
import re
from typing import Dict, List, Optional

from product_similarity.pipeline import run_similarity, parse_scores
from product_similarity.retriever import contexts_from_class_numbers
from product_similarity.spsc import retrieve_spsc_contexts


DEFAULT_ANALYZER_MODEL = None  # Not used in the nature-only evaluation


def evaluate_dataset(csv_path: str, *,
                     model_name: Optional[str] = None,
                     agent_model: Optional[str] = None,  # kept for backward compat, unused
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

    results = []
    correct = 0
    total = 0

    for r in rows:
        # Expect preprocessed columns: Nature, class_1, product_1, class_2, product_2
        p1 = (r.get("product_1") or r.get("Item 1") or "").strip()
        p2 = (r.get("product_2") or r.get("Item 2") or "").strip()
        gold_nature_raw = r.get("Nature")
        gold = None
        if gold_nature_raw not in (None, "", "-"):
            m = re.search(r"(\\d)", str(gold_nature_raw))
            if m:
                try:
                    gold = int(m.group(1))
                except Exception:
                    gold = None

        # Build contexts strictly from provided class numbers if available in CSV
        class_1 = r.get("class_1") or r.get("class1") or r.get("Class 1")
        class_2 = r.get("class_2") or r.get("class2") or r.get("Class 2")
        contexts = contexts_from_class_numbers([class_1, class_2])
        if include_spsc:
            try:
                spsc_ctx = retrieve_spsc_contexts(p1, p2, top_k=spsc_top_k)
                if spsc_ctx:
                    contexts = contexts + spsc_ctx
            except Exception:
                pass

        # Run the pipeline to obtain Nature score only
        sim = run_similarity(
            p1,
            p2,
            class_1=class_1,
            class_2=class_2,
            max_fewshot=5,
            include_spsc=include_spsc,
            spsc_top_k=spsc_top_k,
            chat_api_base_url=chat_api_base_url,
            chat_api_key=chat_api_key,
            chat_api_model=chat_api_model,
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        pred_scores = sim.get("scores") or {}
        pred_nature = pred_scores.get("nature")
        if pred_nature is None:
            # Fallback: try to parse again from raw output
            pred_nature = (parse_scores(sim.get("output_text", "")) or {}).get("nature")

        if gold is not None and pred_nature is not None:
            total += 1
            if int(pred_nature) == int(gold):
                correct += 1

        results.append({
            "product_1": p1,
            "product_2": p2,
            "class_1": class_1,
            "class_2": class_2,
            "contexts": contexts,
            "gold_nature": gold,
            "pred_nature": pred_nature,
            "output_text": sim.get("output_text", ""),
        })

    metrics = {
        "total_compared": total,
        "nature_exact_match": (correct / total) if total > 0 else None,
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


