from xushuhang.agents.api import *

def get_api_class(model_name):
    # a router to get correct API client class
    if 'qwen' in model_name:
        API_CLASS = WWXQ_API
    elif 'doubao' in model_name:
        API_CLASS = Volcano_API
    elif 'kimi' in model_name:
        API_CLASS = Volcano_API
    elif 'deepseek' in model_name:
        API_CLASS = Volcano_API
    elif 'gpt' in model_name:
        API_CLASS = OpenAI_API
    else:
        ## WWXQ by default
        API_CLASS = WWXQ_API
    
    return API_CLASS