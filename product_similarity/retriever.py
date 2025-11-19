import json
import os
import re
from typing import List, Optional, Iterable


PACKAGE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NICE_PATH = os.path.join(DATA_DIR, "nice_chunks.json")


def _load_nice_chunks() -> list:
	if not os.path.exists(NICE_PATH):
		raise FileNotFoundError(f"Missing NICE data at: {NICE_PATH}")
	with open(NICE_PATH, "r", encoding="utf-8") as f:
		return json.load(f)


_NICE_CHUNKS_CACHE: Optional[list] = None


def _get_nice_chunks_cached() -> list:
	global _NICE_CHUNKS_CACHE
	if _NICE_CHUNKS_CACHE is None:
		_NICE_CHUNKS_CACHE = _load_nice_chunks()
	return _NICE_CHUNKS_CACHE


def retrieve_contexts(product_1: str, product_2: str, top_k: int = 3) -> List[str]:
	"""
	Keyword-based retriever over NICE data using local JSON.
	Returns top_k short context strings.
	"""
	terms = set(
		[t for t in re.findall(r"[A-Za-z]+", (product_1 + " " + product_2).lower()) if len(t) > 3]
	)

	scored: list[tuple[int, str]] = []
	for entry in _get_nice_chunks_cached():
		heading = entry.get("heading", "")
		note = entry.get("explanatory_note", "")
		items = entry.get("items", [])
		class_no = entry.get("class_number", "?")

		blob = (
			heading
			+ "\n"
			+ note
			+ "\n"
			+ "\n".join([it.get("Goods and Service", "") for it in items])
		).lower()
		score = sum(blob.count(term) for term in terms)

		if score > 0:
			matched_items = [
				it.get("Goods and Service", "")
				for it in items
				if any(term in it.get("Goods and Service", "").lower() for term in terms)
			]
			snippet_items = "; ".join(matched_items[:3])
			context = f"Class {class_no}: {heading}"
			if snippet_items:
				context += f"\nExamples: {snippet_items}"
			scored.append((score, context))

	scored.sort(key=lambda x: x[0], reverse=True)
	return [ctx for _, ctx in scored[:top_k]]



def _find_class_entry(class_number: str) -> Optional[dict]:
	"""
	Return NICE entry dict for a given class number (string match).
	"""
	for entry in _get_nice_chunks_cached():
		if str(entry.get("class_number", "")).strip() == str(class_number).strip():
			return entry
	return None


def contexts_from_class_numbers(class_numbers: Iterable[object], max_items_per_class: int = 3) -> List[str]:
	"""
	Build context strings directly from provided NICE class numbers, bypassing keyword retrieval.
	Only include the heading and explanatory_note for each class.
	"""
	seen = set()
	contexts: List[str] = []
	for cn in class_numbers:
		if cn is None:
			continue
		cn_str = str(cn).strip()
		if not cn_str or cn_str in seen:
			continue
		seen.add(cn_str)
		entry = _find_class_entry(cn_str)
		if not entry:
			continue
		heading = entry.get("heading", "")
		note = entry.get("explanatory_note", "")
		class_no = entry.get("class_number", cn_str)
		context = f"Class {class_no}: {heading}"
		if isinstance(note, str) and note.strip():
			context += f"\nExplanatory Note: {note.strip()}"
		contexts.append(context)
	return contexts

