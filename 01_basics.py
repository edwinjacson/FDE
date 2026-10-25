"""Exercise 1: API basics.

Covers: a single call, token usage, system prompts, streaming, multi-turn history.

Setup:
    pip install anthropic pydantic
    export ANTHROPIC_API_KEY=your-key
Run:
    python 01_basics.py
"""
import anthropic

MODEL = "claude-haiku-5-5"
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def text_of(message) -> str:
    """Join the text blocks (the response may also contain thinking blocks)."""
    return "".join(b.text for b in message.content if b.type == "text")


def single_call() -> None:
    response = client.messages.create(
        model=MODEL,
        max_tokens=20,
        messages=[{"role": "user", "content": "Explain Terraform state in 3 sentences."}],
    )
    print(text_of(response))
    print(f"\n[in={response.usage.input_tokens} out={response.usage.output_tokens} "
          f"stop={response.stop_reason}]")


def chat() -> None:
    """Multi-turn chat with a system prompt and streamed output.

    The API is stateless: YOU resend the whole history every turn.
    Watch input tokens grow as the conversation gets longer.
    """
    system = "You are a concise senior DevOps mentor. Answer in under 120 words."
    history: list[dict] = []

    while True:
        user = input("\nyou> ").strip()
        if user in {"", "quit", "exit"}:
            break
        history.append({"role": "user", "content": user})

        print("claude> ", end="")
        with client.messages.stream(
            model=MODEL, max_tokens=20, system=system, messages=history
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            final = stream.get_final_message()

        history.append({"role": "assistant", "content": text_of(final)})
        print(f"\n[in={final.usage.input_tokens} out={final.usage.output_tokens}]")


if __name__ == "__main__":
    single_call()
    # chat()

    # Things to try (note what changes):
    # 1. Run the same prompt 3 times and compare outputs. (Claude Haiku 5.5 does not
    #    accept non-default temperature/top_p/top_k, so steer style via the prompt instead.)
    # 2. Change the system prompt to "answer only in JSON" and see how reliable it is.
    # 3. Set max_tokens=20 and inspect stop_reason.
