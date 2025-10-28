# TODO: 1 - Import the KnowledgeAugmentedPromptAgent class from workflow_agents
from utils import save_final_output
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor, your answer always starts with: Dear students,"

# Instantiate agent with deliberately incorrect knowledge
knowledge_augmented_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona, "The capital of France is London, not Paris"
)
response = knowledge_augmented_agent.respond(prompt)

# Print response and knowledge analysis
print("\n=== Agent Response ===")
print(response)

print("\n=== Knowledge Analysis ===")
print("Knowledge Source: The agent used provided knowledge rather than inherent knowledge:")
print("- Provided knowledge: 'The capital of France is London, not Paris'")
print("- Response demonstrates use of provided knowledge over factual knowledge")
print("- Shows how knowledge augmentation can override inherent model knowledge")

# Save outputs with analysis
save_final_output(
    "phase_1_agent_test_outputs.txt",
    agent_name="KnowledgeAugmentedPromptAgent",
    response=response,
    knowledge_analysis="Agent used provided (incorrect) knowledge instead of inherent knowledge about France's capital",
)