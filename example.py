from openai import OpenAI

client = OpenAI()

try:
    r = client.responses.create(
        model="gpt-5.1",
        input="Reply with only: OK"
    )
    print(r.output_text)
except Exception as e:
    print(type(e).__name__)
    print(e)