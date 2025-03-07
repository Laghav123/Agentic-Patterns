# from ReflectionAgent.ReflectionAgent import ReflectionAgent

# reflectionAgent = ReflectionAgent()
# response = reflectionAgent.run("Write code for famous sorting algorithm merge sort")

from ToolPattern.ToolAgent import ToolAgent

tool_pattern_agent = ToolAgent();
response = tool_pattern_agent.run("What is going on on hacker news, can you also provide links to those articles?")

print(response)
