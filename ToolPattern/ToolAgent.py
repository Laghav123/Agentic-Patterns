from groq import Groq
from utils.tools import tools, available_functions
from utils.roles import *
import json
from dotenv import load_dotenv

load_dotenv()

BASE_SYSTEM_PROMPT = """
Your task is to Generate the best content possible for the user's request.
You are provided with several functional tools with clear description. Use these function to get information required to generate best response. 
You must always output the revised content.
"""

class ToolAgent:

    def __init__(
            self,
            model: str = "llama-3.3-70b-versatile"
        ):
        self.client = Groq()
        self.model = model

    # imports calculate function from step 1
    def run(
            self,
            user_prompt: str,
            system_prompt: str = "",
        ):
        
        system_prompt += BASE_SYSTEM_PROMPT
        messages=[
            {
                "role": SYSTEM_ROLE,
                "content": system_prompt
            },
            {
                "role": USER_ROLE,
                "content": user_prompt,
            }
        ]
        
        # Make the initial API call to Groq
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages, # Conversation history
            stream=False,
            tools=tools, # Available tools (i.e. functions) for LLM to use
            tool_choice="auto", # Let LLM decide when to use tools
            max_completion_tokens=4096 # Maximum number of tokens to allow in response
        )

        # Extract the response and any tool call responses
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        if tool_calls:
            # Add the LLM's response to the conversation
            messages.append(response_message)

            # Process each tool call
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                print("trying to call "+ function_name + " with arguments: ")
                print(function_args)
                # Call the tool and get the response
                function_response = function_to_call(
                    **function_args
                )
                # Add the tool response to the conversation
                messages.append(
                    {
                        "tool_call_id": tool_call.id, 
                        "role": "tool", # Indicates this message is from tool use
                        "name": function_name,
                        "content": function_response,
                    }
                )
            # Make a second API call with the updated conversation
            second_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            # Return the final response
            return second_response.choices[0].message.content
        
        return response_message