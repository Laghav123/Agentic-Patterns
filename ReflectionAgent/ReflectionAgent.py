from groq import Groq
from utils.utils import *
from utils.roles import *
from utils.completion import *
from dotenv import load_dotenv

load_dotenv()

BASE_GENERATION_SYSTEM_PROMPT = """
Your task is to Generate the best content possible for the user's request.
If the user provides critique, respond with a revised version of your previous attempt.
You must always output the revised content.
"""

BASE_REFLECTION_SYSTEM_PROMPT = """
You are tasked with validating the user response and generating recommendations to the user's generated content.
If the user content has something wrong or something to be improved, output a list of recommendations
and critiques. If the user content is ok and there's nothing to change, include this '<OK>' in output else do not include it.
"""

class ReflectionAgent:

    def __init__(self, model : str = "llama-3.3-70b-versatile"):
        self.client = Groq()
        self.model = model

    def generate(
            self,
            generation_chat_history: list
    ) -> str :
        """
        Generates a response based on the provided generation history using the model.

        Args:
            generation_chat_history (list): A list of messages forming the conversation or generation history.

        Returns:
            str: The generated response.
        """
        
        # get response based on current chat
        generation = get_completion_from_model(self.client, self.model, generation_chat_history)
        
        # update the response in itselves chat history
        generation_chat_history.append(build_prompt(generation, ASSISTANT_ROLE))
        return generation
    
    def reflect(
            self,
            reflection_chat_history: list
    ) -> str :
        """
        Reflects upon the generated response based on provided reflection history using the model.

        Args:
            reflection_chat_history (list): A list of messages forming the conversation or reflection history.

        Returns:
            str: The reflected response.
        """
        
        # get response based on current chat
        reflection = get_completion_from_model(self.client, self.model, reflection_chat_history)

        # update the response in itselves chat history
        reflection_chat_history.append(build_prompt(reflection, ASSISTANT_ROLE))
        return reflection

    def run(
            self,
            user_msg : str,
            generation_system_prompt : str = "",
            reflection_system_prompt : str = "",
            max_iterations : int = 3,
    ) -> str :
        """
        Runs the ReflectionAgent over multiple steps, alternating between generating a response
        and reflecting on it for the specified number of steps.

        Args:
            user_msg (str): The user message or query that initiates the interaction.
            generation_system_prompt (str, optional): The system prompt for guiding the generation process.
            reflection_system_prompt (str, optional): The system prompt for guiding the reflection process.
            n_steps (int, optional): The number of generate-reflect cycles to perform. Defaults to 3.
            verbose (int, optional): The verbosity level controlling printed output. Defaults to 0.

        Returns:
            str: The final generated response after all cycles are completed.
        """

        generation_system_prompt += BASE_GENERATION_SYSTEM_PROMPT
        reflection_system_prompt += BASE_REFLECTION_SYSTEM_PROMPT

        generation_chat_history = [
            build_prompt(generation_system_prompt, SYSTEM_ROLE),
            build_prompt(user_msg, USER_ROLE),
        ]

        reflection_chat_history = [
            build_prompt(reflection_system_prompt, SYSTEM_ROLE)
        ]

        generation = ""

        for n in range(max_iterations):

            # 1. generate the response from user msg
            generation = self.generate(generation_chat_history)

            # add the generated response to the history of reflection model
            reflection_chat_history.append(build_prompt(generation, USER_ROLE))

            # 2. reflect on the generated response
            reflection = self.reflect(reflection_chat_history)

            # 3. if reflected response contains <OK> then break out of loop
            if "<OK>" == reflection:
                break
            
            # append in generation model's chat history
            generation_chat_history.append(build_prompt(reflection, USER_ROLE))
            

        return  generation;