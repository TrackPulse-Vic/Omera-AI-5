import json

def addMemory(memory, channel_id):
    try:
        try:
            with open('memory/memories.json', 'r', encoding='utf-8') as f:
                memories = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            memories = {}

        if str(channel_id) not in memories:
            memories[str(channel_id)] = []

        if memory not in memories[str(channel_id)]:
            memories[str(channel_id)].append(memory)

        with open('memory/memories.json', 'w', encoding='utf-8') as f:
            json.dump(memories, f, indent=4)
        return 'Memory added successfully.'
    except Exception as e:
        print(f"Error writing to JSON file: {e}")
        return(f"Error writing to JSON file: {e}")

def readMemories(channel_id):
    try:
        with open('memory/memories.json', 'r', encoding='utf-8') as f:
            memories = json.load(f)
            return memories.get(str(channel_id), [])
    except (FileNotFoundError, json.JSONDecodeError):
        return 'No memories found for this channel.'