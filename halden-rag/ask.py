"""Interactive Q&A over the documents.

    python ask.py
    python ask.py --backend embed --top-k 6
"""
import argparse

from rag import add_pipeline_args, build_pipeline, extract_citations, generate, get_client, is_refusal


def main() -> None:
    parser = argparse.ArgumentParser()
    add_pipeline_args(parser)
    args = parser.parse_args()

    chunks, retriever = build_pipeline(args)
    client = get_client()
    print(f"Loaded {len(chunks)} chunks. Ask a question (blank line to quit).")

    while True:
        question = input("\nquestion> ").strip()
        if not question:
            break
        retrieved = retriever.search(question, args.top_k)
        answer, usage = generate(client, question, retrieved, args.dates)
        print(f"\n{answer}\n")
        if not is_refusal(answer):
            print("cited:    ", extract_citations(answer) or "none (suspicious!)")
        print("retrieved:", [f"{c.id} ({score:.2f})" for c, score in retrieved])
        print(f"tokens in={usage['input_tokens']} out={usage['output_tokens']} {usage['seconds']:.1f}s")


if __name__ == "__main__":
    main()
