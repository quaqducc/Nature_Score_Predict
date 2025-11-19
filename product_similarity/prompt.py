import json
from typing import List, Dict, Optional


def format_fewshot(example: Dict) -> str:
	"""
	Format one few-shot example into a structured text block.
	Compatible with new few-shot schema:
	{
		\"case_id\": str,
		\"input\": {
			\"product_1\": str,
			\"product_2\": str,
			\"class_info\": { ... }  # arbitrary mapping of class -> description
		},
		\"output\": { \"nature_score\": int }
		# optional: \"reasoning\": { ... }
	}
	"""
	case_id = example.get("case_id", "UNKNOWN")
	input_data = example.get("input", {})
	reasoning = example.get("reasoning", {})
	output = example.get("output", {})

	blocks = [
		f"### Example {case_id}",
		f"**Input:**",
		f"- Product 1: {input_data.get('product_1')}",
		f"- Product 2: {input_data.get('product_2')}",
		f"- Class Info: {json.dumps(input_data.get('class_info', {}), ensure_ascii=False, indent=2)}",
	]

	# Only include reasoning section if provided in the few-shot
	if isinstance(reasoning, dict) and len(reasoning) > 0:
		blocks.append("")
		blocks.append("**Reasoning:**")
		blocks.extend([f"- {k.capitalize()}: {v}" for k, v in reasoning.items()])

	blocks.append("")
	blocks.append("**Output:**")
	blocks.append(f"- Nature Score: {output.get('nature_score')}")

	return "\n".join(blocks)

def build_prompt(
    fewshot_examples: List[Dict],
    product_1: str,
    product_2: str,
    retrieved_contexts: List[str],
    max_fewshot: Optional[int] = None,
) -> str:
    """
	Build the full prompt for LLM evaluation focusing ONLY on the Nature factor.
	Outputs a Nature Score in [0–4] with concise reasoning.
    """

    # ==== 1. Task description ====
    instruction = (
		"You are an expert examiner in trademark classification (NICE system).\n"
		"Your task is to assess ONLY the similarity in NATURE between two goods or services.\n"
		"Nature refers to what the product essentially is (its type, composition, materials, technical category).\n\n"
		"=== EVALUATION PROCESS (Nature only) ===\n"
		"• Identify what each product essentially is (chemical, device, material, etc.).\n"
		"• Check if they belong to the same technical category or share similar materials/composition.\n"
		"• Ignore intended purpose and other factors unless strictly necessary to clarify nature.\n\n"
		"=== SCORING SCALE ===\n"
		"Use whole numbers only:\n"
		"0 = Not similar\n"
		"1 = Slightly similar\n"
		"2 = Moderately similar\n"
		"3 = Similar\n"
		"4 = Highly similar / identical\n\n"
		"Keep your reasoning concise and factual (max 3–5 sentences per case).\n"
		"Your final answer must include 'Reasoning' and 'Output' sections, matching the format of the examples.\n\n"
    )

    # ==== 2. Few-shot examples ====
    limited_examples = (
        fewshot_examples[:max_fewshot]
        if isinstance(max_fewshot, int) and max_fewshot > 0
        else fewshot_examples
    )
    fewshot_text = "\n\n".join([format_fewshot(ex) for ex in limited_examples])

    # ==== 3. NICE / guideline context ====
    context_text = ""
    if retrieved_contexts:
        context_text = (
            "### Reference Context (from NICE classification & Guidelines):\n"
            + "\n".join(retrieved_contexts)
            + "\n\n"
        )

    # ==== 4. Query ====
    query_text = (
        "### New Case\n"
        f"- Product 1: {product_1}\n"
        f"- Product 2: {product_2}\n\n"
		"Please reason step-by-step focusing ONLY on the Nature factor.\n"
        "Then provide your conclusion following this format:\n\n"
        "Reasoning:\n"
		"- Nature: (brief reasoning)\n\n"
        "Output:\n"
		"- Nature Score: [0–4]\n"
		"Your final line must clearly show the Nature Score.\n"
    )

    # ==== 5. Combine ====
    return instruction + fewshot_text + "\n\n" + context_text + query_text

