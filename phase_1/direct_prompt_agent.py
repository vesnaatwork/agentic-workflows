# Test script for DirectPromptAgent class

from utils import save_final_output
from workflow_agents.base_agents import DirectPromptAgent        
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# TODO: 2 - Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

# TODO: 3 - Instantiate the DirectPromptAgent as direct_agent
direct_agent = DirectPromptAgent(openai_api_key)

# TODO: 4 - Use direct_agent to send the prompt defined above and store the response
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(direct_agent_response)
save_final_output(
    "phase_1_agent_test_outputs.txt",
    agent_name="DirectPromptAgent",
    response=direct_agent_response,
    explanation="The DirectPromptAgent used its inherent knowledge from the training data provided by OpenAI to answer the prompt about the capital of France."
)
# TODO: 5 - Print an explanatory message describing the knowledge source used by the agent to generate the response
print("The DirectPromptAgent likely used its inherent knowledge from the training data provided by OpenAI to answer the prompt about the capital of France.")