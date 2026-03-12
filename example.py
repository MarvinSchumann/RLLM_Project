from openai import OpenAI

client = OpenAI()

try:
    r = client.responses.create(
        model="gpt-5.1",
        input="Anser the Question only with the answer word: There is a yellow triangle, below it is a red object. To the right of the red object is a blue circle. The blue circle is only next to a square. What shape is the red object?"
    )
    print(r.output_text)
except Exception as e:
    print(type(e).__name__)
    print(e)