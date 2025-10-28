from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv
from utils import save_final_output

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# Create agent instance
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

# Get agent's response
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print("\n=== Agent Response ===")
print(augmented_agent_response)

# Print discussion about knowledge and persona
print("\n=== Knowledge and Persona Analysis ===")
print("Knowledge Source: The AugmentedPromptAgent likely used its inherent knowledge about world capitals to answer the prompt.")
print("Persona Impact: The system prompt specifying the professor persona influenced the agent to:")
print("- Frame its response in a formal, educational manner")
print("- Begin with 'Dear students,' as required")
print("- Potentially include additional educational context or explanations")

# Save outputs
save_final_output(
    "phase_1_agent_test_outputs.txt",
    agent_name="AugmentedPromptAgent",
    response=augmented_agent_response,
    knowledge_analysis="Used inherent knowledge about world capitals",
    persona_analysis="Professor persona shaped formal educational tone"
)