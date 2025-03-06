def build_prompt(
        prompt: str, 
        role: str
) -> dict :
    return {
        "role": role,
        "content" : prompt,
    }