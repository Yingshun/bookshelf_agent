"""
RAG Evaluation Script for virttest bookshelf
Evaluates retrieval quality (Precision@K, Recall@K, MRR) and answer quality
(keyword hit rate, semantic similarity, faithfulness).
"""

import warnings
warnings.filterwarnings('ignore', message='.*Qdrant client version.*')
warnings.filterwarnings('ignore', message='.*Pydantic V1.*')

import argparse
import json
import re
import time
import yaml
from datetime import datetime
from statistics import mean

from langchain_core.messages import HumanMessage
from agentic_bookshelf import vectorstore, compiled_graph

_st_model = None


def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def load_test_cases(path, category=None, ids=None):
    with open(path) as f:
        data = yaml.safe_load(f)
    cases = data["test_cases"]
    if category:
        cases = [c for c in cases if c.get("category") == category]
    if ids:
        id_set = set(ids)
        cases = [c for c in cases if c["id"] in id_set]
    return cases


def evaluate_retrieval(retrieved_names, expected_sources):
    relevant = set(retrieved_names) & set(expected_sources)
    precision = len(relevant) / len(retrieved_names) if retrieved_names else 0.0
    recall = len(relevant) / len(expected_sources) if expected_sources else 0.0

    rr = 0.0
    for rank, name in enumerate(retrieved_names, 1):
        if name in set(expected_sources):
            rr = 1.0 / rank
            break

    hit = 1.0 if relevant else 0.0

    return {
        "retrieved_names": retrieved_names,
        "expected_sources": expected_sources,
        "relevant_retrieved": list(relevant),
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "reciprocal_rank": round(rr, 4),
        "hit_at_k": hit,
    }


def evaluate_answer(answer, expected_keywords, reference_answer=None):
    answer_lower = answer.lower()
    hits = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    misses = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    keyword_hit_rate = len(hits) / len(expected_keywords) if expected_keywords else 0.0

    semantic_sim = None
    if reference_answer:
        from sklearn.metrics.pairwise import cosine_similarity
        model = _get_st_model()
        emb_answer = model.encode([answer])
        emb_ref = model.encode([reference_answer])
        semantic_sim = round(float(cosine_similarity(emb_answer, emb_ref)[0][0]), 4)

    return {
        "keyword_hits": hits,
        "keyword_misses": misses,
        "keyword_hit_rate": round(keyword_hit_rate, 4),
        "semantic_similarity": semantic_sim,
        "answer_length": len(answer),
    }


def compute_faithfulness(answer, retrieved_docs_text):
    technical_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*(?:\([^)]*\))?'
    answer_terms = set(re.findall(technical_pattern, answer))
    stop_words = {
        "this", "that", "with", "from", "have", "will", "your", "which",
        "their", "about", "would", "there", "been", "some", "them", "than",
        "other", "into", "could", "more", "also", "each", "make", "like",
        "does", "when", "used", "True", "False", "None", "true", "false",
        "none", "function", "method", "class", "return", "parameter",
        "value", "type", "name", "following", "example",
    }
    answer_terms = {t for t in answer_terms if len(t) >= 4 and t not in stop_words}

    if not answer_terms:
        return 1.0

    docs_lower = retrieved_docs_text.lower()
    grounded = sum(1 for t in answer_terms if t.lower() in docs_lower)
    return round(grounded / len(answer_terms), 4)


def run_answer_eval(question, eval_user_id):
    state = {
        "messages": [HumanMessage(content=question)],
        "mem0_user_id": eval_user_id,
        "retry_count": 0,
    }
    final_state = compiled_graph.invoke(state)
    return final_state["messages"][-1].content


def run_evaluation(test_cases, k=5, skip_answer=False, eval_user_id="eval_user"):
    results = []
    ret = vectorstore.as_retriever(search_kwargs={"k": k})

    for i, tc in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {tc['id']}: {tc['question'][:60]}...")
        result = {
            "id": tc["id"],
            "question": tc["question"],
            "category": tc.get("category", ""),
        }

        docs = ret.invoke(tc["question"])
        retrieved_names = [doc.metadata.get("name", "") for doc in docs]
        result["retrieval"] = evaluate_retrieval(retrieved_names, tc["expected_sources"])

        print(f"  Retrieval: P@{k}={result['retrieval']['precision_at_k']:.2f}  "
              f"R@{k}={result['retrieval']['recall_at_k']:.2f}  "
              f"RR={result['retrieval']['reciprocal_rank']:.2f}  "
              f"Hit={result['retrieval']['hit_at_k']:.0f}")

        if not skip_answer:
            t0 = time.perf_counter()
            answer = run_answer_eval(tc["question"], eval_user_id)
            elapsed = time.perf_counter() - t0

            answer_metrics = evaluate_answer(
                answer, tc["expected_answer_keywords"], tc.get("reference_answer")
            )
            docs_text = "\n".join(doc.page_content for doc in docs)
            answer_metrics["faithfulness"] = compute_faithfulness(answer, docs_text)
            answer_metrics["generated_answer"] = answer
            answer_metrics["latency_s"] = round(elapsed, 2)
            result["answer"] = answer_metrics

            print(f"  Answer: KW={answer_metrics['keyword_hit_rate']:.2f}  "
                  f"Sim={answer_metrics['semantic_similarity'] or 'N/A'}  "
                  f"Faith={answer_metrics['faithfulness']:.2f}  "
                  f"({elapsed:.1f}s)")

        results.append(result)

    return results


def print_results(results, k):
    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)
    header = f"{'ID':<10} {'Category':<16} {'P@'+str(k):<6} {'R@'+str(k):<6} {'RR':<6} {'Hit':<4}"
    print(header)
    print("-" * len(header))
    for r in results:
        m = r["retrieval"]
        print(f"{r['id']:<10} {r['category']:<16} {m['precision_at_k']:<6.2f} "
              f"{m['recall_at_k']:<6.2f} {m['reciprocal_rank']:<6.2f} {m['hit_at_k']:<4.0f}")

    retrieval_metrics = [r["retrieval"] for r in results]
    print(f"\nAggregate: "
          f"Mean P@{k}={mean(m['precision_at_k'] for m in retrieval_metrics):.2f}  "
          f"Mean R@{k}={mean(m['recall_at_k'] for m in retrieval_metrics):.2f}  "
          f"MRR={mean(m['reciprocal_rank'] for m in retrieval_metrics):.2f}  "
          f"Hit Rate={mean(m['hit_at_k'] for m in retrieval_metrics):.2f}")

    if any("answer" in r for r in results):
        print("\n" + "=" * 70)
        print("ANSWER RESULTS")
        print("=" * 70)
        header = f"{'ID':<10} {'KW Hit':<8} {'Sem Sim':<8} {'Faith':<8} {'Latency':<8}"
        print(header)
        print("-" * len(header))
        for r in results:
            if "answer" not in r:
                continue
            a = r["answer"]
            sim = f"{a['semantic_similarity']:.2f}" if a["semantic_similarity"] is not None else "N/A"
            print(f"{r['id']:<10} {a['keyword_hit_rate']:<8.2f} {sim:<8} "
                  f"{a['faithfulness']:<8.2f} {a['latency_s']:<8.1f}s")

        answer_results = [r["answer"] for r in results if "answer" in r]
        sims = [a["semantic_similarity"] for a in answer_results if a["semantic_similarity"] is not None]
        print(f"\nAggregate: "
              f"Mean KW Hit={mean(a['keyword_hit_rate'] for a in answer_results):.2f}  "
              f"Mean Sim={mean(sims):.2f if sims else 'N/A'}  "
              f"Mean Faith={mean(a['faithfulness'] for a in answer_results):.2f}")


def save_report(results, k, path):
    retrieval_metrics = [r["retrieval"] for r in results]
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {"k": k, "num_test_cases": len(results)},
        "aggregate": {
            "retrieval": {
                "mean_precision_at_k": round(mean(m["precision_at_k"] for m in retrieval_metrics), 4),
                "mean_recall_at_k": round(mean(m["recall_at_k"] for m in retrieval_metrics), 4),
                "mrr": round(mean(m["reciprocal_rank"] for m in retrieval_metrics), 4),
                "hit_rate": round(mean(m["hit_at_k"] for m in retrieval_metrics), 4),
            },
        },
        "per_question": results,
    }

    if any("answer" in r for r in results):
        answer_results = [r["answer"] for r in results if "answer" in r]
        sims = [a["semantic_similarity"] for a in answer_results if a["semantic_similarity"] is not None]
        report["aggregate"]["answer"] = {
            "mean_keyword_hit_rate": round(mean(a["keyword_hit_rate"] for a in answer_results), 4),
            "mean_semantic_similarity": round(mean(sims), 4) if sims else None,
            "mean_faithfulness": round(mean(a["faithfulness"] for a in answer_results), 4),
        }

    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation for virttest bookshelf")
    parser.add_argument("--test-file", default="eval_test_cases.yaml")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--ids", type=str, nargs="*", default=None)
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--report", default="eval_report.json")
    parser.add_argument("--eval-user-id", default="eval_user")
    args = parser.parse_args()

    test_cases = load_test_cases(args.test_file, category=args.category, ids=args.ids)
    print(f"Loaded {len(test_cases)} test case(s), k={args.k}")

    results = run_evaluation(
        test_cases, k=args.k, skip_answer=args.skip_answer, eval_user_id=args.eval_user_id
    )
    print_results(results, args.k)
    save_report(results, args.k, args.report)
    print(f"\nReport saved to {args.report}")


if __name__ == "__main__":
    main()
