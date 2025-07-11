import os

from openai import OpenAI


async def generateImage(prompt,model='grok-2-image-1212'):
    KEY = os.getenv("XAI_API_KEY")
    client = OpenAI(base_url="https://api.x.ai/v1", api_key=KEY)

    response = client.images.generate(
    model=model,
    prompt=prompt
    )

    return response.data[0].url, response.data[0].revised_prompt

# NOT USED ANYMORE
async def understantImage(image_url, prompt, model='grok-2-vision-latest', user=None):
    KEY = os.getenv("XAI_API_KEY")
    client = OpenAI(
        api_key=KEY,
        base_url="https://api.x.ai/v1",
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
    