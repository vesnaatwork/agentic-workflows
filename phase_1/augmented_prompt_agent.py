from workflow_agents.base_agents import AugmentedPromptAgent# TODO: 1 - Import the AugmentedPromptAgent class
import os
from dotenv import load_dotenv
from utils import save_final_output
# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

augmented_agent_response = AugmentedPromptAgent(openai_api_key, persona)

augmented_agent_response = augmented_agent_response.respond(prompt)

# Print the agent's response
print(augmented_agent_response)
save_final_output(
    "phase_1_agent_test_outputs.txt",
    agent_name="AugmentedPromptAgent",
    response=augmented_agent_response
)
# TODO: 4 - Add a comment explaining:
# - What knowledge the agent likely used to answer the prompt.
# - How the system prompt specifying the persona affected the agent's response.
print("The AugmentedPromptAgent likely used its inherent knowledge about world capitals to answer the prompt. " \
"The system prompt specifying the persona influenced the agent to frame its response in a formal and educational manner, " \
"beginning with 'Dear students,'.")