tools = [
    {
        "name": "emotional_expert",
        "description": "情感识别专家",
        "parameters": {
            "type": "object",
            "properties": {
                "emotion": {
                    "description": "情感类型 e.g. 爱，聊天，作诗"
                }
            },
            "required": ['emotion']
        }
    },
    {
        "name": "flirting_hutao",
        "description": "扮演原神角色胡桃",
        "parameters": {
            "type": "object",
            "properties": {
                "emotion": {
                    "description": "情感类型 e.g. 爱，作诗，聊天"
                },
                "user": {
                    "description": "调情的对象 e.g. 旅行者，钟离"
                }
            },
            "required": ['emotion', 'user']
        }
    }
]


system_info = {"role": "system",
               "content": "Answer the following questions as best as you can. You have access to the following tools:",
               "tools": tools}
