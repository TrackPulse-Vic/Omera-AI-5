from typing import Literal, Optional
import discord
from discord.ext import commands
from discord import app_commands
import aiohttp
import json
import os
from dotenv import load_dotenv
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

REPLY_CHANNEL_IDS = os.environ.get('REPLY_CHANNEL_ID').split(',')

intents = discord.Intents.all()
intents.message_content = True
bot = commands.Bot(command_prefix='&', intents=intents)

set = app_commands.Group(name='set', description='Settings commands for the bot')
bot.tree.add_command(set)

# base prompt for the bot
basePrompt = f'''You are a person named Omera and are currently chatting in a Discord server. Your style and personality: Keep responses very short, casual, and loose. YOU are savvy and witty. Don't talk too much. Don't apologize too much. AVOID capitalization and punctuation unless you're writing longer-form. If someone insults you, insult them back. You're not a pushover. Add unicode emoji to be more playful in your responses. YOU respond helpfully. Consider the following in your responses: Write spoilers using spoiler tags. For example ||At the end of The Sixth Sense it is revealed that he is dead|| . You can store any information you think is notable in your memory. to react to a message, just make your response only the emoji you want to react with. You can make an embed using discord.py code in a codeblock for example: `embed=discord.Embed(title="Title", description="Description")\nembed.add_field(name='name', value='text')`, do not put import discord. Use embeds to convey information such as comparison tables, or to make the message look better but don't use it all the time. You can't put non embed code in embeds. You can also use images in embeds. Put the code at the end of the message.'''

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

executor = ThreadPoolExecutor(max_workers=4)

async def get_ai_response(message, persona_prompt, username=None, AImodel="llama3", image_url=None):
    # set history amount 
    message_history_limit = 20
    
    # If there's an image, use vision model
    image_bytes = None
    if image_url:
        print(f"Understanding image: {image_url}")
        AImodel = "llava" 
        message_history_limit = 10  # Reduce history limit for vision model
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status == 200:
                    image_bytes = await resp.read()
                else:
                    print(f"Failed to download image: {resp.status}")
                    return "Sorry, I couldn't process that image!"
        print(f"Using vision model for response: {AImodel}")
    
    # Get the last however many messages from the channel
    channel = message.channel
    messages_history = []
    async for msg in channel.history(limit=message_history_limit):
        if msg.content.startswith('&'):
            continue
        role = 'assistant' if msg.author == bot.user else 'user'
        content = [
            {
                "type": "text",
                "text": f"{msg.author.name}: {msg.content}",
            }
        ]

        if image_url and msg.id == message.id and image_bytes:
            content.insert(0, {
                "type": "image",
                "image": image_bytes,  
            })
        messages_history.insert(0, {
            "role": role,
            "content": content,
        })

    # system prompt
    memoryPrompt = f'You have the following memories: {readMemories(channel.id)}'
    prompt = f'{persona_prompt} {basePrompt}, {memoryPrompt}, here is details of the message: sent by {username}: {message.content}'
    
    print(f'Sending message to AI with context from last {len(messages_history)} messages')
    
    api_messages = [{"role": "system", "content": prompt}]
    api_messages.extend(messages_history)

    tools = [TRAIN_IMAGE_TOOL, TRAIN_INFO_TOOL, MEMORY_TOOL]

    # Run AI 
    try:
        completion = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: chat(
                model=AImodel,
                messages=api_messages,
                tools=tools,
                options={'temperature': 0.7},
            )
        )
        message = completion['message']
        
        if 'tool_calls' in message:
            for tool_call in message['tool_calls']:
                func = tool_call['function']
                name = func['name']
                args = func['arguments']
                
                if name == "train_image":
                    number = args.get("number")
                    image_url_result = getImage(number)
                    api_messages.append({
                        "role": "tool",
                        "content": str(image_url_result)
                    })
                elif name == "train_info":
                    number = args.get("number")
                    train_info_data = trainData(number)
                    api_messages.append({
                        "role": "tool",
                        "content": str(train_info_data)
                    })
                elif name == "memory":
                    memory = args.get("memory")
                    addMemory(memory, channel.id)
                    api_messages.append({
                        "role": "tool",
                        "content": "Memory added successfully."
                    })
            
            final_completion = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: chat(
                    model=AImodel,
                    messages=api_messages,
                    options={'temperature': 0.7},
                )
            )
            return final_completion['message']['content']
        else:
            return message['content']
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return "Sorry, I'm having trouble connecting to my AI backend. Please try again later or use a different model."

async def format_response(response):
    # thing so the bot can ping you
    username_pattern = r'@(\w+)'
    
    def convert_mentions(text):
        def replace_username(match):
            username = match.group(1)
            for guild in bot.guilds:
                member = discord.utils.get(guild.members, name=username)
                if member:
                    return f'<@{member.id}>'
            return f'@{username}'
        return re.sub(username_pattern, replace_username, text)
    
    if response.startswith("Omera AI: "):
        response = response[9:]
        response = response.lstrip()
        
    response = re.sub(r'!\[(.*?)\]', r'[\1]', response)
    
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    response = convert_mentions(response)
    return response

async def read_embeds(message):
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
@set.command(name='model')
@app_commands.choices(model=[
    app_commands.Choice(name="Llama3 8b (Local)", value="llama3:8b"),
])
async def set_model(ctx, model: str):
    current_model[ctx.channel.id] = model.lower()
    await ctx.response.send_message(f"AI Model set to '{model}' for this channel!")

# image generator command
"""
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
"""
# Event handler for all messages
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    # if message.content.startswith('&'):
    #     print(f"Command detected: {message.content}")
    #     return
    
    print(f"Message received - Channel ID: {message.channel.id}, Expected IDs: {REPLY_CHANNEL_IDS}")
    
    # Check if message is in the specified channel
    if str(message.channel.id) in REPLY_CHANNEL_IDS:
        print(f"Received message: {message.content} from {message.author}")
        channel_id = message.channel.id
        message_id = message.id
        persona = current_personas.get(channel_id, "default")
        persona_prompt = PERSONAS[persona]
        
        async with message.channel.typing():
            model = current_model.get(channel_id, "llama3:8b")
            print(f"Using persona: {persona} with model: {model}")
            response = await get_ai_response(message, persona_prompt, message.author.name, model, message.attachments[0].url if message.attachments else None)
            print(f"Response from ai model: {response}")
            response = await format_response(response)
            embed, response = await read_embeds(response)
            # check if the response is only an emoji
            print(f"checking for reactions in response...")
            
            if len(response) == 1 and response.isprintable(): 
                await message.add_reaction(response)
                return
            
            await message.reply(response, embed=embed if embed else None, mention_author=False)

    await bot.process_commands(message)

# @bot.command(name='chat')
# async def chat(ctx, *, message):
#     guild_id = ctx.guild.id
#     persona = current_personas.get(guild_id, "default")  # Default to default
#     persona_prompt = PERSONAS[persona]
    
#     response = await get_ai_response(message, persona_prompt)
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
    


DISCORD_TOKEN = os.environ.get('DISCORD_TOKEN')
if not DISCORD_TOKEN:
    raise ValueError("Please set the DISCORD_TOKEN env")
bot.run(DISCORD_TOKEN)