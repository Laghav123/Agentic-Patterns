def get_completion_from_model(
        client,
        model: str,
        history: list,
) -> str:
    """
    This function is responsible for generating response based on current chat history

    Args: 
        client: groq in this case
        model: model of groq
        history: list of messages in current chat history
    
    Returns:
        str: the response returned by the groq's model
    """
    print("Making API call:")
    response = client.chat.completions.create(
            messages=history,
            model=model
        )
    return str(response.choices[0].message.content)