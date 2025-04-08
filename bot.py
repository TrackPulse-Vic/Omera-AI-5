from typing import Literal, Optional
import discord
from discord.ext import commands
from discord import app_commands
import aiohttp
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import re

load_dotenv()

intents = discord.Intents.all()  # Enable all intents for full message access
intents.message_content = True
bot = commands.Bot(command_prefix='&', intents=intents)

# base prompt for the bot
basePrompt = f'''You are sending messages in a discord server. You can use markdown formatting. and ping people. keep messages kina short like a chat. to react to a message, just make your response "react emoji" where emoji is the emoji you want to react with.'''

# Available personas
PERSONAS = {
    "default": "You are Grok, a chatbot developed by XAI. You are friendly and helpful",
    "professional": "You are a formal, professional assistant focused on clear, concise answers",
    "sarcastic": "You are a witty, sarcastic assistant who loves playful jabs",
    'railway': "You are a railway enthusiast, specifically focused on the Victorian railways. You are knowledgeable about trains, stations, and railway history. You cannot provide information about train schedules or other real time info, for that, tell the user to use the TrackPulse VIC bot. You are friendly and helpful.",
}   

# Store current persona per server
current_personas = {}

async def get_grok_response(message, persona_prompt, username=None):
    # Get the last 10 messages from the channel
    channel = message.channel
    messages_history = []
    async for msg in channel.history(limit=10):
        # Skip system messages and bot commands
        if msg.content.startswith('&'):
            continue
        
        role = "assistant" if msg.author == message.guild.me else "user"
        messages_history.insert(0, {
            "role": role,
            "content": f"{msg.author.name}: {msg.content}"
        })

    # Create the system prompt
    prompt = f'{persona_prompt} {basePrompt}, here is details of the message: sent by {username}: {message.content}'
    
    print(f'Sending message to Grok API with context from last {len(messages_history)} messages')
    
    XAI_API_KEY = os.getenv("XAI_API_KEY")
    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )

    # Add system prompt and conversation history
    api_messages = [{"role": "system", "content": prompt}]
    api_messages.extend(messages_history)

    completion = client.chat.completions.create(
        model="grok-2-latest",
        messages=api_messages
    )
    
    return completion.choices[0].message.content

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
    
    response = convert_mentions(response)
    return response
            
@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

# Command to set persona
@bot.tree.command(name='setpersona')
@app_commands.choices(persona=[
    app_commands.Choice(name="Default", value="default"),
    app_commands.Choice(name="Gunzel", value="railway"),
    app_commands.Choice(name="Professional", value="professional"),
    app_commands.Choice(name="Sarcastic", value="sarcastic"),

])
async def set_persona(ctx, persona: str):
    if persona.lower() not in PERSONAS:
        available = ", ".join(PERSONAS.keys())
        await ctx.response.send_message(f"Invalid persona! Available options: {available}")
        return
    
    current_personas[ctx.guild.id] = persona.lower()
    await ctx.response.send_message(f"Persona set to '{persona}' for this server!")

# Channel ID from .env
REPLY_CHANNEL_ID = int(os.environ.get('REPLY_CHANNEL_ID'))

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
    print(f"Message received - Channel ID: {message.channel.id}, Expected ID: {REPLY_CHANNEL_ID}")
    
    # Check if message is in the specified channel
    if message.channel.id == REPLY_CHANNEL_ID:
        print(f"Received message: {message.content} from {message.author}")
        guild_id = message.guild.id
        persona = current_personas.get(guild_id, "default")  # Default to default
        persona_prompt = PERSONAS[persona]
        
        async with message.channel.typing():
            response = await get_grok_response(message, persona_prompt, message.author.name)
            print(f"Response from Grok: {response}")
            response = await format_response(response)
            
            # check if the response is to react to a message
            print(f"checking for reactions in: {response}")
            
            if response.startswith("react"):
                emoji = response.split(" ")[1]
                await message.add_reaction(emoji)
                return
            
            await message.channel.send(response)

    # Process commands (needed to keep commands working)
    await bot.process_commands(message)

# # Keep the chat command for other channels
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
    
    
@bot.command()
@commands.guild_only()
async def sync(ctx: commands.Context, guilds: commands.Greedy[discord.Object], spec: Optional[Literal["~", "*", "^"]] = None) -> None:
    if ctx.author.id == 780303451980038165:

        if not guilds:
            if spec == "~":
                synced = await ctx.bot.tree.sync(guild=ctx.guild)
            elif spec == "*":
                ctx.bot.tree.copy_global_to(guild=ctx.guild)
                synced = await ctx.bot.tree.sync(guild=ctx.guild)
            elif spec == "^":
                ctx.bot.tree.clear_commands(guild=ctx.guild)
                await ctx.bot.tree.sync(guild=ctx.guild)
                synced = []
            else:
                synced = await ctx.bot.tree.sync()

            await ctx.send(
                f"Synced {len(synced)} commands {'globally' if spec is None else 'to the current guild.'}"
            )
            return

        ret = 0
        for guild in guilds:
            try:
                await ctx.bot.tree.sync(guild=guild)
            except discord.HTTPException as e:
                await print(f'Error: {e}')
            else:
                ret += 1

        await ctx.send(f"Synced the tree to {ret}/{len(guilds)}.")

# Run the bot - replace with your Discord bot token
DISCORD_TOKEN = os.environ.get('DISCORD_TOKEN')
if not DISCORD_TOKEN:
    raise ValueError("Please set the DISCORD_TOKEN environment variable")
bot.run(DISCORD_TOKEN)