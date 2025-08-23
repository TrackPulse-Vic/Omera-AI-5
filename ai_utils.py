import os

from openai import OpenAI


async def generateImage(prompt,model):
    KEY = os.getenv("API_KEY")
    client = OpenAI(base_url="", api_key=KEY)

    response = client.images.generate(
    model=model,
    prompt=prompt
    )

    return response.data[0].url, response.data[0].revised_prompt

# NOT USED ANYMORE
async def understantImage(image_url, prompt, model, user=None):
    KEY = os.getenv("API_KEY")
    client = OpenAI(
        api_key=KEY,
        base_url="",
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.01,
    )

    return(completion.choices[0].message.content)
    