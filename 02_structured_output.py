"""Exercise 2: structured output.

Extract validated JSON from messy text by giving the model a tool whose input
schema is your Pydantic model, then validate with Pydantic and feed errors
back to the model for a retry.

Note: Claude Sonnet 5.5 rejects forced tool choice, so this script avoids it. We ask for the tool call
in the prompt instead of using tool_choice.

Run:
    python 02_structured_output.py
"""
from datetime import date
from dotenv import load_dotenv

import anthropic
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

MODEL = "claude-haiku-5-5"
client = anthropic.Anthropic()


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float


class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    invoice_date: date
    currency: str = Field(description="ISO 4217 code, e.g. GBP")
    line_items: list[LineItem]
    total: float


def check_totals(inv: Invoice) -> None:
    """Business-rule check the schema alone can't express."""
    computed = sum(i.quantity * i.unit_price for i in inv.line_items)
    if abs(computed - inv.total) > 0.01:
        raise ValueError(f"Line items sum to {computed:.2f} but total is {inv.total:.2f}")


def extract_invoice(text: str, max_retries: int = 2) -> Invoice:
    tools = [{
        "name": "record_invoice",
        "description": "Record structured invoice data extracted from the text.",
        "input_schema": Invoice.model_json_schema(),
    }]
    messages = [{
        "role": "user",
        "content": ("Extract the invoice details from the text below and record them by "
                    "calling the record_invoice tool.\n\n"
                    f"<text>\n{text}\n</text>"),
    }]

    for attempt in range(max_retries + 1):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            tools=tools,
            messages=messages,
        )
        print(resp.content)
        block = next((b for b in resp.content if b.type == "tool_use"), None)
        if block is None:
            raise RuntimeError("Model did not call the tool; check your prompt")
        try:
            inv = Invoice.model_validate(block.input)
            check_totals(inv)
            return inv
        except (ValidationError, ValueError) as e:
            print(f"  attempt {attempt + 1} failed validation: {e}")
            # Show the model its own output plus the error, then let it fix it.
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "is_error": True,
                    "content": f"Validation failed: {e}. Correct the data and call the tool again.",
                }],
            })
    raise RuntimeError("Could not get valid output after retries")


SAMPLES = [
    """
    From: accounts@brightpipe.co.uk
    Hi, invoice BP-2291 dated 3rd March 2026. 12 hrs consulting @ £85/hr,
    and 1x server setup fee £250. Total due: £1,270. Thanks, Brightpipe Ltd
    """,
    """
    ACME CLOUD SERVICES  --  INV# 0044-B  --  14/11/2025
    Storage (500 GB) ........ 500 x 0.02 USD
    Support plan ............ 1 x 75.00 USD
    TOTAL: 85.00 USD
    """,
]

if __name__ == "__main__":
    for s in SAMPLES:
        print("Extracting...")
        print(extract_invoice(s).model_dump_json(indent=2), "\n")

    # Stretch goals:
    # 1. Add a sample with a missing field (no invoice number). Make the schema
    #    allow None and have the model say "unknown" instead of guessing.
    # 2. Build 10 samples with hand-written expected answers and print field-level accuracy.
    #    This is the seed of your eval harness for Project 1.
    # 3. Compare the tool-schema approach to just asking for JSON in the prompt. Measure failures.
