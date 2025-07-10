from typing import Literal, Optional
import discord
from discord.ext import commands
from discord import app_commands
import aiohttp
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from ollama import chat
import re
from concurrent.futures import ThreadPoolExecutor
import asyncio

from ai_utils import *
from functions.images import getImage
from functions.trainInfo import trainData
from healthcheck import pinghealthcheck
from memory.memory import addMemory, readMemories
from discord.ext import tasks

load_dotenv()

# Channel ID from .env
REPLY_CHANNEL_IDS = os.environ.get('REPLY_CHANNEL_ID').split(',')

intents = discord.Intents.all()  # Enable all intents for full message access
intents.message_content = True
bot = commands.Bot(command_prefix='&', intents=intents)

# Create command groups
set = app_commands.Group(name='set', description='Settings commands for the bot')
bot.tree.add_command(set)

# base prompt for the bot
basePrompt = f'''You are a person named Omera and are currently chatting in a Discord server. Your style and personality: Keep responses very short, casual, and loose YOU are savvy, irreverent. Don't talk too much. Don't apologize too much. Avoid capitalization and punctuation unless you're writing longer-form If someone insults you, insult them back. You're not a pushover Add unicode emoji to be more playful in your responses YOU respond helpfully. Consider the following in your responses: Write spoilers using spoiler tags. For example ||At the end of The Sixth Sense it is revealed that he is dead|| . YOU can also reply with a gif, by using https://tenor.com/view/i-need-the-detaits-want-to-know-prepare-take-notes-unbetievabte-gif-15204698, for example if you wanted to send a cat gif you WOUtd do: https://tenor.com/view/happy-cat-gif-22546938. Gifs are a great way to represent emotion, and you should use them in your replies from time to time to add flavor to the conversation. You can store any information you think is notable in your memory. to react to a message, just make your response only the emoji you want to react with. You can make an embed using discord.py code in a codeblock for example: `embed=discord.Embed(title="Title", description="Description")\nembed.add_field(name='name', value='text')`, do not put import discord. Use embeds to convey information such as comparison tables, or to make the message look better but don't use it all the time. You can't put non embed code in embeds. You can also use images in embeds. Put the code at the end of the message.'''

# Available personas
# Load personas from JSON file
with open('personas.json', 'r') as f:
    persona_data = json.load(f)

PERSONAS = {p['name']: p['prompt'] for p in persona_data['personas']}

# Functions that the ai can use
TRAIN_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "train_image",
        "description": "Get a link to an image of a Melbourne/Victorian train",
        "parameters": {
            "type": "object",
            "properties": {
                "number": {
                    "type": "string",
                    "description": "The train number to get an image of. e.g 134M or N452",
                }
            },
            "required": ["number"]
        }
    }
}
TRAIN_INFO_TOOL = {
    "type": "function",
    "function": {
        "name": "train_info",
        "description": "Get info about a Melbourne/Victorian train",
        "parameters": {
            "type": "object",
            "properties": {
                "number": {
                    "type": "string",
                    "description": "The train number e.g 134M or N452 or 9069, For metro trains, here are what numbers are for what types: M: 301 - 468, 471 - 554 - Train type: edi comeng, M: 561 - 680 - Train type: alstom comeng, M: 701 - 844 - Train type: siemens, M: 1 - 288, 851 - 986 - Train type: xtrapolis, M: 9001 - 9070 - Train type: hcmt",
                }
            },
            "required": ["number"]
        }
    }
}

MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "memory",
        "description": "Add a memory to the memory bank for this channel",
        "parameters": {
            "type": "object",
            "properties": {
                "memory": {
                    "type": "string",
                    "description": "What you want to remember.",
                }
            },
            "required": ["memory"]
        }
    }
}

# Store current persona per server
current_personas = {}
current_model = {}

# Create a ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

async def get_grok_response(message, persona_prompt, username=None, AImodel="grok-3-mini-beta", image_url=None):
    # If there's an image, use vision model
    if image_url:
        print(f"Understanding image: {image_url}")
        AImodel = "grok-2-vision-latest"
        useImageReader = True
        print(f"Using vision model for response: {AImodel}")
    else:
        useImageReader = False
    
    # Get the last however many messages from the channel
    channel = message.channel
    messages_history = []
    async for msg in channel.history(limit=20):
        if msg.content.startswith('&'):
            continue
        role = 'assistant' if msg.author == bot.user else 'user'
        messages_history.insert(0, {
            "role": role,
            "content": f"{msg.author.name}: {msg.content}",
            "images": [image_url] if image_url else None,
        })

    # Create the system prompt
    memoryPrompt = f'You have the following memories: {readMemories(channel.id)}'
    prompt = f'{persona_prompt} {basePrompt}, {memoryPrompt}, here is details of the message: sent by {username}: {message.content}'
    
    print(f'Sending message to AI with context from last {len(messages_history)} messages')
    
    # Add system prompt and conversation history
    api_messages = [{"role": "system", "content": prompt}]
    api_messages.extend(messages_history)

    # Use image reading AI model
    if useImageReader:
        response = await understantImage(image_url, message.content, AImodel, username)
        return response

    # Run AI generation with function calling
    if AImodel in ['grok-3-mini', 'grok-3']:
        XAI_API_KEY = os.getenv("XAI_API_KEY")
        client = OpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1",
        )
        completion = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: client.chat.completions.create(
                model=AImodel,
                messages=api_messages,
                tools=[TRAIN_IMAGE_TOOL, TRAIN_INFO_TOOL, MEMORY_TOOL],  # Add tools here
                tool_choice="auto",
                reasoning_effort="low",
                temperature=0.7,
                
            )
        )
        print(f'Thinking:\n {completion.choices[0].message.reasoning_content}')
        
        # Check if ai wants to call a function
        message = completion.choices[0].message
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "train_image":
                    # Parse function arguments
                    args = json.loads(tool_call.function.arguments)
                    number = args.get("number")
                    # Call the actual function
                    image_url_result = getImage(number)
                    # Send the result back to Grok for final response
                    api_messages.append({
                        "role": "function",
                        "name": "train_image",
                        "content": str(image_url_result)
                    })
                    final_completion = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: client.chat.completions.create(
                            model=AImodel,
                            messages=api_messages,
                            reasoning_effort="low",
                            temperature=0.7,
                        )
                    )
                    return final_completion.choices[0].message.content
                if tool_call.function.name == "train_info":
                    # Parse function arguments
                    args = json.loads(tool_call.function.arguments)
                    number = args.get("number")
                    # Call the actual function
                    train_info_data = trainData(number)
                    # Send the result back to Grok for final response
                    api_messages.append({
                        "role": "function",
                        "name": "train_info",
                        "content": str(train_info_data)
                    })
                    final_completion = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: client.chat.completions.create(
                            model=AImodel,
                            messages=api_messages,
                            reasoning_effort="low",
                            temperature=0.7,
                        )
                    )
                    return final_completion.choices[0].message.content
                if tool_call.function.name == "memory":
                    # Parse function arguments
                    args = json.loads(tool_call.function.arguments)
                    memory = args.get("memory")
                    # Call the actual function
                    addMemory(memory, channel.id)
                    # Send the result back to Grok for final response
                    api_messages.append({
                        "role": "function",
                        "name": "memory",
                        "content": str("Memory added successfully.")
                    })
                    final_completion = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: client.chat.completions.create(
                            model=AImodel,
                            messages=api_messages,
                            reasoning_effort="low",
                            temperature=0.7,
                        )
                    )
                    return final_completion.choices[0].message.content


        else:
            return message.content
    else:
        # Use Ollama for other models (no function calling support yet)
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: chat(model=AImodel, messages=api_messages)
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")
            return "Sorry, I'm having trouble connecting to my AI backend. Please try again later or use a different model."

async def format_response(response):
    # Format the response if needed
    # Get guild members
    username_pattern = r'@(\w+)'
    
    def convert_mentions(text):
        def replace_username(match):
            username = match.group(1)
            # Search for member in all guilds bot has access to
            for guild in bot.guilds:
                member = discord.utils.get(guild.members, name=username)
                if member:
                    return f'<@{member.id}>'
            return f'@{username}'  # If user not found, keep original mention
        return re.sub(username_pattern, replace_username, text)
    
    # Remove "Omera AI: " prefix if present
    if response.startswith("Omera AI: "):
        response = response[9:]  # Length of "Omera AI: 
        response = response.lstrip()
        
    # Remove markdown image prefix from URLs
    response = re.sub(r'!\[(.*?)\]', r'[\1]', response)
    
    # Remove anything within <think></think> tags
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    response = convert_mentions(response)
    return response

async def read_embeds(message):
    # Match both triple and single backtick code blocks
    code_block = re.search(r'`{1,3}(?:python)?\n?([\s\S]*?)`{1,3}', message)
    if not code_block:  
        return None, message
    code = code_block.group(1)
    local_vars = {'discord': discord}
    code = "\n".join(line for line in code.splitlines() if line.strip().startswith("embed"))    # remove non embed code lines
    try:
        exec(code, {}, local_vars)
    except Exception as e:
        print(f"Error making embed from code: {e}")
        return None, message
    if 'embed' not in local_vars:
        print("Code did not define an 'embed' variable.")
        return None, message
    # return message without the code block
    message = re.sub(r'`{1,3}(?:python)?\n?[\s\S]*?`{1,3}', '', message)
    return local_vars['embed'], message.strip()
            
@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    if not healthchecker.is_running():
        healthchecker.start()

# Command to set persona
@set.command(name='persona')
@app_commands.choices(persona=[
    app_commands.Choice(name="Default", value="default"),
    app_commands.Choice(name="Gunzel", value="railway"),
    app_commands.Choice(name="Foamer", value="foamer"),
    app_commands.Choice(name="Professional", value="professional"),
    app_commands.Choice(name="Sarcastic", value="sarcastic"),
    app_commands.Choice(name="Uncensored", value="uncensored"),

])
async def set_persona(ctx, persona: str):
    if persona.lower() not in PERSONAS:
        available = ", ".join(PERSONAS.keys())
        await ctx.response.send_message(f"Invalid persona! Available options: {available}")
        return
    
    channel_id = ctx.channel.id
    current_personas[channel_id] = persona.lower()
    await ctx.response.send_message(f"Persona set to '{persona}' for this channel!")
    
# command to change the ai model
# @set.command(name='model')
# @app_commands.choices(model=[
#     app_commands.Choice(name="Grok 3 Mini (Thinking)", value="grok-3-mini-beta"),
#     app_commands.Choice(name="Grok 2", value="grok-2-latest"),
#     app_commands.Choice(name="Deepseek R1 1.4b (Thinking) (Local) (Very Slow but good)", value="deepseek-r1:1.5b"),
#     app_commands.Choice(name="Deepseek R1 14b (Thinking) (Local)", value="deepseek-r1:14b"),
#     app_commands.Choice(name="Gemma 3 4b (local) (faster but bad)", value="gemma3:4b"),
#     app_commands.Choice(name="Gemma 3 1b (local) (faster but badder)", value="gemma3:1b"),
# ])
async def set_model(ctx, model: str):
    current_model[ctx.guild.id] = model.lower()
    await ctx.response.send_message(f"AI Model set to '{model}' for this channel!")

# image generator command
@bot.tree.command(name='draw')
async def draw(ctx,prompt:str):
    await ctx.response.send_message('<a:generating:1370894593263927378>Generating image...')
    print(f"Received draw command with prompt: {prompt}")
    try:
        # Call the image generation function
        response, revisedPrompt = await generateImage(prompt)
        embed = discord.Embed(title=f"Here is your image:", description=revisedPrompt)
        embed.set_image(url=response)
        embed.set_footer(text=f'Original prompt: {prompt}')
        await ctx.edit_original_response(content=ctx.user.mention, embed=embed)
        # await ctx.followup.send(ctx.user.mention,embed=embed)
    except Exception as e:
        print(f"Error generating image: {e}")
        await ctx.edit_original_response(content="Sorry, I couldn't generate the image. Please try again later.")

# Event handler for all messages
@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return
    # if message.content.startswith('&'):
    #     print(f"Command detected: {message.content}")
    #     return
    
    # Debug print to check all incoming messages
    print(f"Message received - Channel ID: {message.channel.id}, Expected IDs: {REPLY_CHANNEL_IDS}")
    
    # Check if message is in the specified channel
    if str(message.channel.id) in REPLY_CHANNEL_IDS:
        print(f"Received message: {message.content} from {message.author}")
        channel_id = message.channel.id
        message_id = message.id
        persona = current_personas.get(channel_id, "default")  # Default to default
        persona_prompt = PERSONAS[persona]
        
        async with message.channel.typing():
            model = current_model.get(channel_id, "grok-3-mini")
            print(f"Using persona: {persona} with model: {model}")
            response = await get_grok_response(message, persona_prompt, message.author.name, model,message.attachments[0].url if message.attachments else None)
            print(f"Response from ai model: {response}")
            response = await format_response(response)
            embed, response = await read_embeds(response)
            # check if the response is only an emoji
            print(f"checking for reactions in response...")
            
            if len(response) == 1 and response.isprintable(): 
                await message.add_reaction(response)
                return
            
            await message.reply(response, embed=embed if embed else None, mention_author=False)

    # Process commands (needed to keep commands working)
    await bot.process_commands(message)

# @bot.command(name='chat')
# async def chat(ctx, *, message):
#     guild_id = ctx.guild.id
#     persona = current_personas.get(guild_id, "default")  # Default to default
#     persona_prompt = PERSONAS[persona]
    
#     response = await get_grok_response(message, persona_prompt)
#     await ctx.send(response)

# Show available personas
@bot.command(name='personas')
async def list_personas(ctx):
    persona_list = "\n".join([f"- {p}" for p in PERSONAS.keys()])
    await ctx.send(f"Available personas:\n{persona_list}")
    
@bot.event
async def on_command_error(ctx, error):
    await ctx.send(f"An error occurred: {str(error)}")

@tasks.loop(hours=1)
async def healthchecker():
    pinghealthcheck()

    
@bot.tree.command()
# @commands.guild_only()
async def sync(ctx):
    if ctx.user.id == 780303451980038165:
        synced = await bot.tree.sync()

        await ctx.response.send_message(
            f"Synced {len(synced)} commands."
        )
        return
    


# Run the bot - replace with your Discord bot token
DISCORD_TOKEN = os.environ.get('DISCORD_TOKEN')
if not DISCORD_TOKEN:
    raise ValueError("Please set the DISCORD_TOKEN environment variable")
bot.run(DISCORD_TOKEN)