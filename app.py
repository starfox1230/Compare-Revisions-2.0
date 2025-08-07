# smoke_test.py
import os, json, openai
from openai import OpenAI

print("SDK version:", openai.__version__)
print("API key present:", bool(os.getenv("OPENAI_API_KEY")))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1) Minimal Responses call with a known-good model fallback
MODEL_ID = os.getenv("MODEL_ID", "gpt-5-mini")  # try gpt-4o-mini if this 404s
print("Using model:", MODEL_ID)

try:
    resp = client.responses.create(
        model=MODEL_ID,
        instructions="Output only JSON.",
        input="Return {\"ok\": true}",
        max_output_tokens=64,
        response_format={"type": "json_object"},
    )
    # output_text isn't always present on older SDKs; handle both shapes
    text = getattr(resp, "output_text", None)
    if text is None:
        # aggregate text the long way
        chunks = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text" and getattr(c, "text", None):
                    chunks.append(c.text)
        text = "".join(chunks) if chunks else json.dumps(resp.model_dump())
    print("Responses call OK. Text:", text)
except Exception as e:
    print("Responses call failed:", repr(e))

# 2) See if this key can see gpt-5-mini
try:
    models = client.models.list()
    names = [m.id for m in models.data]
    print("Model count:", len(names))
    print("gpt-5-mini visible?", "gpt-5-mini" in names)
except Exception as e:
    print("Listing models failed:", repr(e))
