import json
import os
import re
from typing import Dict, Optional

from .prompt import build_prompt
from .retriever import contexts_from_class_numbers, DATA_DIR
from .spsc import retrieve_spsc_contexts


FEWSHOT_PATH = os.path.join(DATA_DIR, "fewshot_cases.json")


def _load_fewshot_cases() -> list:
	if not os.path.exists(FEWSHOT_PATH):
		raise FileNotFoundError(f"Missing fewshot cases at: {FEWSHOT_PATH}")
	with open(FEWSHOT_PATH, "r", encoding="utf-8") as f:
		return json.load(f)


def parse_scores(output: str) -> Dict[str, Optional[int]]:
	"""
	Parse model output to extract integer scores for nature, purpose, and overall.
	If a score cannot be found, value is None.
	"""
	scores: Dict[str, Optional[int]] = {"nature": None, "purpose": None, "overall": None}
	if not output:
		return scores

	patterns = {
		"nature": r"Nature\s*Score:\s*(\d)",
		"purpose": r"Purpose\s*Score:\s*(\d)",
		"overall": r"Overall(?:\s+Similarity)?(?:\s+Score)?\s*[:\-]\s*(\d)",
	}

	for key, pat in patterns.items():
		m = re.search(pat, output, re.IGNORECASE)
		if m:
			scores[key] = int(m.group(1))
	return scores


def run_similarity(
    product_1: str,
    product_2: str,
    *,
    class_1: Optional[object] = None,
    class_2: Optional[object] = None,
    max_fewshot: int = 5,
    top_k: int = 0,
    include_spsc: bool = True,
    spsc_top_k: int = 2,
    model_name: Optional[str] = None,
    # Chat API (OpenAI-compatible) options:
    chat_api_base_url: Optional[str] = None,
    chat_api_key: Optional[str] = None,
    chat_api_model: Optional[str] = None,
    device: int = -1,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Dict[str, object]:
	"""
	Run the end-to-end similarity pipeline. If model_name is None, we skip
	local inference and only return the built prompt and empty output.
	"""
	fewshot_cases = _load_fewshot_cases()
	# Build contexts strictly from provided NICE class numbers.
	# If classes are not provided, contexts will be empty.
	contexts = contexts_from_class_numbers([class_1, class_2])

	# Always-on (by default) SPSC augmentation
	if include_spsc and spsc_top_k and spsc_top_k > 0:
		try:
			spsc_ctx = retrieve_spsc_contexts(product_1, product_2, top_k=spsc_top_k)
			if spsc_ctx:
				contexts = contexts + spsc_ctx
		except Exception:
			# Keep robust if SPSC retrieval fails
			pass

	prompt = build_prompt(fewshot_cases, product_1, product_2, contexts, max_fewshot=max_fewshot)

	output_text = ""
	if chat_api_base_url and chat_api_key and chat_api_model:
		try:
			from .model import ChatAPIWrapper
			chat = ChatAPIWrapper(
                base_url=str(chat_api_base_url),
                api_key=str(chat_api_key),
                model=str(chat_api_model),
                max_tokens=max_new_tokens,
            )
			output_text = chat.run(prompt, temperature=max(temperature, 0.0), top_p=top_p)
		except Exception:
			output_text = ""
	elif model_name:
		try:
			from .model import LLMWrapper
			llm = LLMWrapper(model_name=model_name, device=device, max_new_tokens=max_new_tokens)
			output_text = llm.run(prompt, temperature=temperature, top_p=top_p)
		except Exception:
			# Keep output_text empty on any inference error
			output_text = ""

	return {
		"product_1": product_1,
		"product_2": product_2,
		"class_1": class_1,
		"class_2": class_2,
		"contexts": contexts,
		"prompt": prompt,
		"output_text": output_text,
		"scores": parse_scores(output_text),
	}

