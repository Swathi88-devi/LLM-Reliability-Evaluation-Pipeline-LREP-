import json
import argparse
from pipeline import LLMEvaluator, EvalConfig
def load_json(path):
    """Load a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def save_json(obj, path):
    """Save a Python object to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False
def evaluate_from_json(conversation_path, context_path, output_path):
    """
    Evaluate a single conversation + context pair from JSON files.
    """
    conv = load_json(conversation_path)
    ctx = load_json(context_path)

    cfg = EvalConfig(
        model_name="all-MiniLM-L6-v2",
        hall_threshold=0.35,
        batch_size=32,
        top_k_keywords=8
    )
    evaluator = LLMEvaluator(cfg)
    result = evaluator.evaluate(conv, ctx)
    save_json(result, output_path)
    print(f"✓ Evaluation completed. Results saved to: {output_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM evaluation using JSON inputs.")
    parser.add_argument("--conversation", required=True, help="Path to conversation JSON")
    parser.add_argument("--context", required=True, help="Path to context JSON")
    parser.add_argument("--out", default="result.json", help="Output JSON path")
    args = parser.parse_args()
    evaluate_from_json(args.conversation, args.context, args.out)
