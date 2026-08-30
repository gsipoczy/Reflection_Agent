from langchain.chat_models import init_chat_model

def get_model():

    model_name = "claude-sonnet-4-6"
    model_temperature = 0.5
    #model_max_tokens = 1024

    try:
        model = init_chat_model(
            model_name, 
            temperature = model_temperature,
            #max_tokens=con.model_max_tokens,)
        )
    except Exception as e: 
        print(e)
        model = None

    return model