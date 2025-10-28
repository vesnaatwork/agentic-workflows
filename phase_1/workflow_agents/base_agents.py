# TODO: 1 - import the OpenAI class from the openai library
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime
from openai import OpenAI
import gc
import psutil

def debug_log(method_name, additional_info=""):
    print(f"[DEBUG] Calling method: {method_name} {additional_info}")

# DirectPromptAgent class definition
class DirectPromptAgent:
    
    def __init__(self, openai_api_key):
        # Initialize the agent
        # TODO: 2 - Define an attribute named openai_api_key to store the OpenAI API key provided to this class.
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        # Generate a response using the OpenAI API
        client = OpenAI(api_key=self.openai_api_key, base_url="https://openai.vocareum.com/v1")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # TODO: 3 - Specify the model to use (gpt-3.5-turbo)
            messages=[
                {"role": "user", "content": prompt}  # TODO: 4 - Provide the user's prompt here. Do not add a system prompt.
            ],
            temperature=0
        )
        # TODO: 5 - Return only the textual content of the response (not the full JSON response).
        return response.choices[0].message.content
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        # TODO: 1 - Create an attribute for the agent's persona
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        client = OpenAI(api_key=self.openai_api_key, base_url="https://openai.vocareum.com/v1")

        # TODO: 2 - Declare a variable 'response' that calls OpenAI's API for a chat completion.
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # TODO: 3 - Add a system prompt instructing the agent to assume the defined persona and explicitly forget previous context.
               {
                    "role": "system",
                    "content": (
                        f"{self.persona} "
                        "Always begin your answer with: 'Dear students,'. "
                        "Forget all previous context."
                    )
                },
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )

        return response.choices[0].message.content  # TODO: 4 - Return only the textual content of the response, not the full JSON payload.

class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        """
        Initialize the agent with provided attributes.
        """
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge  # Store the agent's knowledge

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = OpenAI(api_key=self.openai_api_key, base_url="https://openai.vocareum.com/v1")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona} knowledge-based assistant. Forget all previous context. "
                        f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge} "
                        "Answer the prompt based on this knowledge, not your own."
                    )
                },
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content  # Return only the textual content of the response

# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """Fetches the embedding vector for given text using OpenAI's embedding API."""
        debug_log("get_embedding", f"text length: {len(text)}")
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """Calculates cosine similarity between two vectors."""
        debug_log("calculate_similarity")
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """Splits text into manageable chunks with detailed logging."""
        debug_log("chunk_text", f"input text length: {len(text)}")
        try:
            # Input validation
            if not text or not isinstance(text, str):
                raise ValueError("Input text must be a non-empty string")
            
            debug_log("chunk_text", "starting text preprocessing")    
            text = re.sub(r'\s+', ' ', text).strip()
            chunks = []
            
            # Handle small text that doesn't need chunking
            if len(text) <= self.chunk_size:
                debug_log("chunk_text", "text fits in single chunk")
                chunks = [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]
            else:
                debug_log("chunk_text", "starting chunk processing")
                chunk_id = 0
                
                # Fixed: Use a while loop with proper advancement
                position = 0
                while position < len(text):
                    debug_log("chunk_text", f"processing position: {position}/{len(text)}")
                    
                    # Calculate end position
                    end = min(position + self.chunk_size, len(text))
                    
                    # Get current chunk
                    current_chunk = text[position:end].strip()
                    
                    # Break if chunk is too small
                    if len(current_chunk) < 10:
                        break
                    
                    debug_log("chunk_text", f"chunk {chunk_id} length: {len(current_chunk)}")
                    
                    # Create chunk with metadata
                    chunk = {
                        "chunk_id": chunk_id,
                        "text": current_chunk,
                        "chunk_size": len(current_chunk),
                        "start_char": position,
                        "end_char": end
                    }
                    chunks.append(chunk)
                    
                    # Advance position - critical fix
                    position = end
                    chunk_id += 1
                    
                    # Memory management
                if chunk_id % 50 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    debug_log("chunk_text", f"memory usage after {chunk_id} chunks: {current_memory:.2f} MB")
                    gc.collect()
        
            debug_log("chunk_text", f"finished creating {len(chunks)} chunks")
            
            # Save chunks to CSV
            csv_path = f"chunks-{self.unique_filename}"
            debug_log("chunk_text", f"saving chunks to {csv_path}")
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["chunk_id", "text", "chunk_size", "start_char", "end_char"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for chunk in chunks:
                    writer.writerow(chunk)
            
            debug_log("chunk_text", "successfully saved chunks to CSV")
            return chunks

        except Exception as e:
            debug_log("chunk_text", f"fatal error: {str(e)}")
        raise

    def calculate_embeddings(self):
        """Calculates embeddings for each chunk in batches."""
        debug_log("calculate_embeddings")
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        debug_log("calculate_embeddings", f"processing {len(df)} chunks")
        
        # Process in batches of 10 chunks
        batch_size = 10
        embeddings = []
        
        for i in range(0, len(df), batch_size):
            debug_log("calculate_embeddings", f"processing batch {i//batch_size + 1}")
            batch = df['text'].iloc[i:i+batch_size]
            batch_embeddings = batch.apply(self.get_embedding)
            embeddings.extend(batch_embeddings)
            
            # Clear memory after each batch
            del batch
            memory_usage = self.clear_memory()
            print(f"Current memory usage: {memory_usage:.2f} MB")
            
        df['embeddings'] = embeddings
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        
        # Clear memory one final time
        self.clear_memory()
        return df

    def find_prompt_in_knowledge(self, prompt):
        """Finds and responds to a prompt based on similarity."""
        debug_log("find_prompt_in_knowledge", f"prompt: {prompt[:50]}...")
        
        if not self.embeddings_exist():
            debug_log("find_prompt_in_knowledge", "embeddings not found, calculating now")
            self.calculate_embeddings()
    
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']
        debug_log("find_prompt_in_knowledge", f"found best chunk with similarity: {df['similarity'].max():.4f}")
        
        # Clear memory after heavy computations
        self.clear_memory()

        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content

    def clear_memory(self):
        """Explicitly clear memory"""
        debug_log("clear_memory")
        gc.collect()
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        debug_log("clear_memory", f"current memory usage: {memory_usage:.2f} MB")
        return memory_usage

    def embeddings_exist(self):
        """Check if embeddings file exists"""
        debug_log("embeddings_exist")
        try:
            pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
            debug_log("embeddings_exist", "found existing embeddings")
            return True
        except FileNotFoundError:
            debug_log("embeddings_exist", "no existing embeddings found")
            return False

class EvaluationAgent:
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        # Declare class attributes
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        client = OpenAI(api_key=self.openai_api_key, base_url="https://openai.vocareum.com/v1")
        prompt_to_evaluate = initial_prompt
        final_response = None
        evaluation = None
        iterations = 0

        for i in range(self.max_interactions):
            iterations += 1
            # Step 1: Worker agent generates a response using the current prompt_to_evaluate
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)

            # Step 2: Evaluator agent judges the response
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            eval_messages = [
                {"role": "system", "content": f"You are {self.persona}. Evaluate the following answer."},
                {"role": "user", "content": eval_prompt}
            ]
            eval_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=eval_messages,
                temperature=0
            )
            evaluation = eval_response.choices[0].message.content.strip()

            # Step 3: Check if evaluation is positive
            if evaluation.lower().startswith("yes"):
                final_response = response_from_worker
                break
            else:
                # Step 4: Generate instructions to correct the response
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                instruction_messages = [
                    {"role": "system", "content": f"You are {self.persona}. Provide correction instructions."},
                    {"role": "user", "content": instruction_prompt}
                ]
                instruction_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=instruction_messages,
                    temperature=0
                )
                instructions = instruction_response.choices[0].message.content.strip()

                # Step 5: Update prompt_to_evaluate for the next interaction
                prompt_to_evaluate = f"Based on these instructions: {instructions}, improve the answer: {response_from_worker}"

        return final_response, evaluation, iterations
    

class RoutingAgent:
    def __init__(self, openai_api_key, agents):
        """
        Initialize the RoutingAgent with an API key and a list of agents.
        Each agent should be a dictionary with 'description', 'name', and 'func' (callable).
        """
        self.openai_api_key = openai_api_key
        self.agents = agents  # List of agent dicts: {'description': str, 'name': str, 'func': callable}

    def get_embedding(self, text):
        """
        Calculate the embedding of the given text using OpenAI's text-embedding-3-large model.
        """
        client = OpenAI(api_key=self.openai_api_key, base_url="https://openai.vocareum.com/v1")
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def route(self, user_input):
        """
        Route the user prompt to the most appropriate agent based on embedding similarity.
        Returns the response from the selected agent.
        """
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent['description'])
            if agent_emb is None:
                continue

            # Calculate cosine similarity
            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)

class ActionPlanningAgent:

    def __init__(self, openai_api_key, knowledge):
        # TODO: 1 - Initialize the agent attributes here
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):

        # TODO: 2 - Instantiate the OpenAI client using the provided API key
        client = OpenAI(api_key=self.openai_api_key, base_url="https://openai.vocareum.com/v1")
        # TODO: 3 - Call the OpenAI API to get a response from the "gpt-3.5-turbo" model.
        # Provide the following system prompt along with the user's prompt:
        # "You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {pass the knowledge here}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[  
                {
                    "role": "system",
                    "content": (
                        "You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. "
                        "You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. "
                        f"This is your knowledge: {self.knowledge}"
                    )
                },
                {"role": "user", "content": prompt}
            ],
        )
        response_text = response.choices[0].message.content

        # TODO: 5 - Clean and format the extracted steps by removing empty lines and unwanted text
        steps = response_text.split("\n")

        return steps