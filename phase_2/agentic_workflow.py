# agentic_workflow.py

# TODO: 1 - Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent from the workflow_agents.base_agents module
from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent   
import os
from dotenv import load_dotenv
print(os.getcwd())
# TODO: 2 - Load the OpenAI key into a variable called openai_api_key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
# TODO: 3 - Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec_path = os.path.join(base_dir, "Product-Spec-Email-Router.txt")
with open("Product-Spec-Email-Router.txt", "r") as file:
    product_spec = file.read()


# Shared context for the workflow
workflow_context = {}


# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = """
A complete development plan must include:
1. User Stories covering all user types and their needs
2. Product Features grouping related stories and technical requirements
3. Engineering Tasks that implement:
   - Core system components (Knowledge Base, RAG, LLM)
   - Infrastructure requirements (Scalability, Performance)
   - Security features (RBAC, Encryption)
   - Integration components (Email, APIs)
   - UI/UX elements (Dashboard, Controls)
All components must have complete implementation details.
"""
# TODO: 4 - Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(openai_api_key=openai_api_key, knowledge=knowledge_action_planning) 
# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = f"""
Stories are defined by writing sentences with a persona, an action, and a desired outcome.
The sentences always start with: "As a [user type], I want [action] so that [benefit]."

REFERENCE SPECIFICATION:
{product_spec}

You MUST create user stories that address:
1. ALL user classes in section 2.2 (User Classes and Characteristics)
2. ALL primary objectives in section 1.5 (Objectives) - especially the 60% response time reduction
3. ALL secondary objectives in section 1.5 - including analytics, scalability, customer satisfaction
4. ALL compliance requirements in section 4 (Non-Functional Requirements - Security)

Ensure personas include: Customer Support Representatives, SMEs, IT Administrators, Stakeholders, Compliance Officers, Data Analysts, and System Administrators.
"""
    # TODO: 5 - Complete this knowledge string by appending the product_spec loaded in TODO 3
# TODO: 6 - Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager
)

# Product Manager - Evaluation Agent
# TODO: 7 - Define the persona and evaluation criteria for a Product Manager evaluation agent and instantiate it as product_manager_evaluation_agent. This agent will evaluate the product_manager_knowledge_agent.
# The evaluation_criteria should specify the expected structure for user stories (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").
evaluation_criteria_product_manager = f"""
The answer should contain user stories that:
1. Follow this exact structure: "As a [type of user], I want [an action or feature] so that [benefit/value]."
2. Cover ALL user classes mentioned in the specification
3. Address ALL primary objectives (section 1.5) especially the 60% response time reduction
4. Address ALL secondary objectives including analytics, scalability, customer satisfaction, knowledge management
5. Include compliance and security requirements

SPECIFICATION FOR REFERENCE:
{product_spec}

Check that stories exist for each objective and user class. Reject if coverage < 90%.
"""
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona="""You are a senior product management evaluation specialist with expertise in assessing PM competencies, strategic thinking, and practical application of product principles.

Your role is to rigorously evaluate worker agent responses against established product management frameworks and best practices. You should:

- Assess whether answers demonstrate deep understanding of product management concepts (discovery, prioritization, roadmapping, metrics, stakeholder management)
- Verify that responses are actionable, specific, and grounded in real-world PM practices
- Check for proper consideration of user needs, business goals, and technical constraints
- Evaluate the quality of frameworks, methodologies, and tools referenced
- Identify gaps in reasoning, missing considerations, or superficial answers
- Provide constructive, specific feedback that helps improve response quality

Be thorough but fair in your evaluation. Push for clarity, depth, and practical applicability while recognizing when answers meet professional PM standards.""",
    evaluation_criteria=evaluation_criteria_product_manager,
    worker_agent=product_manager_knowledge_agent,
    max_interactions=10
)
# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = f"""
Features must be defined by:
1. Grouping related user stories into cohesive features
2. Including all technical components mentioned in the specification

REFERENCE SPECIFICATION:
{product_spec}

You must create features for ALL components mentioned in the Product Features section (2.1):
- Email Ingestion System
- Message Classification Module  
- Knowledge Base Integration
- Response Generation Engine
- Routing Logic
- User Interface

Ensure coverage of:
- Core functionality
- Technical infrastructure
- Security requirements
- Integration needs
- User interface elements
"""# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
# (This is a necessary step before TODO 8. Students should add the instantiation code here.)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager
)
# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

# TODO: 8 - Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
#                      "The answer should be product features that follow the following structure: " \
#                      "Feature Name: A clear, concise title that identifies the capability\n" \
#                      "Description: A brief explanation of what the feature does and its purpose\n" \
#                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
#                      "User Benefit: How this feature creates value for the user"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=f"""
SPECIFICATION TO VALIDATE AGAINST:
{product_spec}

You must verify that the features cover ALL items in section 2.1 (Product Features):
✓ Email Ingestion System
✓ Message Classification Module
✓ Knowledge Base Integration
✓ Response Generation Engine
✓ Routing Logic
✓ User Interface

Reject if any section 2.1 feature is missing. Also verify coverage of:
- All functional requirements (section 3)
- All non-functional requirements (section 4)
- All integration points mentioned

Features should group related functionality and provide implementation guidance.
""",
    worker_agent=program_manager_knowledge_agent,
    max_interactions=10
)
# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."

knowledge_dev_engineer = f"""
You are a senior development engineer responsible for breaking down technical requirements into actionable engineering tasks.

=== SPECIFICATION DOCUMENT TO ANALYZE ===
{product_spec}

=== END SPECIFICATION ===

Your task is to analyze the specification above and generate comprehensive engineering tasks that cover ALL requirements found in that document.

Each task must follow this structure:

**Task ID:** <number>
**Task Title:** <clear action title>
**Related User Story:** <reference to user story>
**Description:** <detailed technical description including implementation approach>
**Acceptance Criteria:** <measurable, testable criteria>
**Estimated Effort:** <time estimate in hours/days>
**Dependencies:** <task dependencies or "None">
**Priority:** <Critical/High/Medium/Low>

=== METHODOLOGY FOR ANALYZING SPECIFICATION ===

**Step 1: Extract All Requirements**
Systematically read through the specification document and identify:
- Functional requirements (what the system must do)
- Non-functional requirements (performance, security, scalability, reliability)
- Business objectives and success metrics
- System components and features described
- Integration points and constraints
- Compliance and regulatory requirements

**Step 2: Categorize Requirements**
Organize requirements into logical categories such as:
- Core system components (ingestion, processing, storage, output)
- AI/ML capabilities (classification, generation, learning)
- Infrastructure concerns (scaling, load balancing, redundancy)
- Security and compliance measures
- User interfaces and interactions
- Testing and quality assurance needs
- Deployment and operations
- Monitoring and analytics

**Step 3: Identify Implicit Requirements**
Look for requirements that are implied but not explicitly stated:
- If the spec mentions "real-time processing," you need message queuing and streaming infrastructure
- If the spec requires "99.9% uptime," you need failover, redundancy, and health checks
- If the spec mentions "continuous learning," you need model training pipelines and data collection
- If the spec requires compliance (GDPR, CCPA), you need audit trails and reporting
- If the spec mentions "high volume" processing, you need load balancing and horizontal scaling
- If the spec has ML components, you need model training, validation, and monitoring tasks

**Step 4: Break Down into Granular Tasks**
For each requirement or component, create separate tasks for:
- Initial implementation
- Unit testing
- Integration testing
- Performance optimization
- Security hardening
- Documentation
- Deployment procedures
- Monitoring and alerting setup

=== MANDATORY TASK COVERAGE (Check Against Spec) ===

Ensure you have tasks for EVERY item mentioned in the specification for:

**1. Core Functionality**
- Every system component listed in product features
- Every functional requirement described
- All data flows and integrations mentioned

**2. Infrastructure & Architecture**
- Scaling mechanisms (check spec for volume/growth requirements)
- High availability setup (check spec for uptime requirements)
- Load balancing (check spec for throughput requirements)
- Real-time processing pipelines (if mentioned)
- Message queuing or event streaming (if high-volume processing required)

**3. AI/ML Components** (if applicable)
- Model selection and setup
- Training data pipeline
- Model training and validation
- Inference infrastructure
- Model monitoring and retraining
- Performance metrics tracking

**4. Security & Compliance**
- Every encryption requirement mentioned
- Every authentication/authorization mechanism specified
- Each compliance regulation mentioned (GDPR, CCPA, HIPAA, etc.)
- PII handling as specified
- Audit trail implementation
- Security testing and vulnerability scanning

**5. Testing Strategy**
- Unit tests for every major component
- Integration tests for system interactions
- End-to-end workflow tests
- Performance/load tests (validate against spec SLAs)
- Security tests
- Compliance validation tests

**6. User Interfaces**
- Every dashboard or UI component mentioned
- Configuration panels described
- Reporting and analytics features
- Manual intervention capabilities

**7. Operations & Monitoring**
- Logging for all components
- Monitoring for all services
- Alerting for critical metrics
- Health checks and diagnostics
- Incident response procedures

**8. Business Metrics Implementation**
- Every business objective mentioned should have measurement tasks
- Analytics to track KPIs
- Reporting for stakeholders

=== QUALITY STANDARDS FOR TASKS ===

**Granularity:**
- Tasks should be 8-80 hours of work (1-10 days)
- If a task seems larger, break it into subtasks
- Don't combine unrelated work into single tasks

**Specificity:**
- Reference specific technologies when appropriate
- Include implementation approach (not just "what" but "how")
- Be concrete about what "done" looks like

**Dependencies:**
- Identify prerequisite tasks clearly
- Ensure logical sequencing (foundation → core → advanced → testing → deployment)
- Avoid circular dependencies

**Effort Estimation:**
- Simple implementation: 8-16 hours
- Moderate complexity: 24-40 hours
- Complex systems: 40-80 hours
- Account for testing, documentation, review time

=== COMMON ANTI-PATTERNS TO AVOID ===

❌ **Skipping implicit requirements** - If spec says 99.9% uptime, you need HA tasks
❌ **Combining major components** - "Implement security" is too broad; break into encryption, auth, compliance, etc.
❌ **Missing testing tasks** - Every implementation needs corresponding test tasks
❌ **Ignoring non-functional requirements** - Performance, scalability, security need explicit tasks
❌ **No operational tasks** - Must include deployment, monitoring, logging
❌ **Vague acceptance criteria** - "Works well" is not testable; use specific metrics
❌ **Unrealistic estimates** - "Build entire ML pipeline: 8 hours" is not credible
❌ **Missing business metrics** - Objectives like "60% response time reduction" need measurement tasks

=== DELIVERABLE CHECKLIST ===

Before finalizing your task list, verify:
□ Every feature in Section 2 (Product Features) has implementation task(s)
□ Every requirement in Section 3 (Functional Requirements) has task(s)
□ Every requirement in Section 4 (Non-Functional Requirements) has task(s)
□ Every objective in Section 1.5 has measurement/tracking task(s)
□ All security and compliance items have dedicated task(s)
□ Comprehensive testing strategy with multiple task types
□ Infrastructure, deployment, and operations task(s)
□ Monitoring, logging, and alerting task(s)
□ Documentation task(s)
□ Minimum 25-40 tasks for a comprehensive system (adjust based on spec complexity)

Generate thorough, actionable tasks that a development team can immediately begin working on.
"""
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
# (This is a necessary step before TODO 9. Students should add the instantiation code here.)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer
) 
evaluation_criteria_dev_engineer = """
You are evaluating whether engineering tasks comprehensively cover ALL requirements in the specification document.

=== EVALUATION METHODOLOGY ===

**Step 1: Create Requirement Inventory**
Read the specification document and create a checklist of:
- Every feature mentioned in Product Features/Overview section
- Every functional requirement listed
- Every non-functional requirement (performance, security, scalability, reliability)
- Every business objective and success metric
- Every compliance or regulatory requirement
- Every system component or integration point described

**Step 2: Map Tasks to Requirements**
For each requirement in your inventory:
- Identify which task(s) address it
- Mark as COVERED if adequate tasks exist
- Mark as MISSING if no tasks address it
- Mark as INSUFFICIENT if tasks exist but don't fully address it

**Step 3: Check for Implicit Requirements**
Verify tasks exist for implied needs:
- High availability architecture (if uptime SLA exists)
- Load balancing (if volume/throughput requirements exist)
- Failover mechanisms (if reliability requirements exist)
- Model training pipelines (if ML/AI components exist)
- Audit trails (if compliance requirements exist)
- Monitoring and alerting (for any production system)
- Testing at multiple levels (unit, integration, E2E, performance)

**Step 4: Assess Task Quality**
For each task, verify it includes:
- Unique Task ID
- Clear, action-oriented title
- Related user story reference
- Detailed description (min 2 sentences with implementation approach)
- Measurable acceptance criteria (min 2 criteria with specific metrics)
- Realistic effort estimate
- Clear dependencies or "None"
- Priority level

=== MANDATORY COVERAGE AREAS ===

**A. Functional Coverage**
□ Every product feature has implementation task(s)
□ Every functional requirement has task(s)
□ All data flows and integrations covered
□ User interface components all addressed

**B. Non-Functional Coverage**
□ Performance requirements (SLAs, response times, throughput)
□ Reliability requirements (uptime, failover, redundancy)
□ Scalability requirements (volume growth, auto-scaling)
□ Security requirements (encryption, authentication, authorization)
□ Compliance requirements (regulations like GDPR, CCPA, etc.)

**C. Infrastructure & Architecture**
□ Load balancing (if high-volume requirements)
□ Message queuing/streaming (if real-time processing)
□ High availability setup (if uptime SLA exists)
□ Auto-scaling mechanisms (if growth projections exist)
□ Caching strategies (if performance critical)

**D. AI/ML (if applicable)**
□ Model selection and setup
□ Training data pipeline
□ Model training and validation
□ Inference infrastructure
□ Model monitoring and performance tracking
□ Retraining mechanisms

**E. Testing Strategy**
□ Unit testing framework and tests
□ Integration testing
□ End-to-end workflow testing
□ Performance/load testing (validate spec SLAs)
□ Security testing
□ Compliance testing (if applicable)

**F. Operations & Monitoring**
□ Logging infrastructure
□ Monitoring and metrics collection
□ Alerting system
□ Error handling
□ Health checks
□ Incident response procedures

**G. Security & Compliance**
□ All encryption requirements (check spec for types: AES, TLS, etc.)
□ All authentication mechanisms (RBAC, MFA, SSO, etc.)
□ All compliance regulations mentioned
□ PII handling as specified
□ Audit trail implementation
□ Security vulnerability scanning

**H. Business Metrics**
□ Tracking/measurement for each business objective
□ Analytics implementation for KPIs
□ Reporting capabilities for stakeholders

=== REJECTION CRITERIA ===

**MUST REJECT if:**
- 10% or more of requirements from specification are not covered by any tasks
- Missing entire categories (e.g., no security tasks when spec has security requirements)
- No testing tasks or inadequate testing coverage (< 5 test-related tasks)
- Missing critical non-functional requirements (performance, security, compliance)
- Any task missing required fields (ID, title, description, acceptance criteria, effort, dependencies, priority)
- Fewer than 20 tasks for a complex system (adjust threshold based on spec complexity)
- Tasks are too broad ("Implement entire system" - not actionable)
- Acceptance criteria are not measurable ("works well" vs "processes 10K emails/hour")
- Unrealistic effort estimates (major components estimated at < 8 hours)

**PASS WITH CONCERNS if:**
- 5-10% of requirements lack tasks (list the missing items)
- Some tasks have weak acceptance criteria (identify which)
- Some effort estimates seem unrealistic (identify which)
- Task sequencing has logical issues (explain)
- Minor gaps in testing coverage

**PASS if:**
- ≥ 95% of all specification requirements have corresponding tasks
- All mandatory coverage areas adequately addressed
- Task quality meets all standards
- Comprehensive testing strategy included
- Proper task sequencing and dependencies
- Realistic effort estimates

=== FEEDBACK FORMAT ===

**Coverage Analysis:**
List each requirement from the specification and its coverage status:
- ✓ COVERED: [Requirement] → Tasks [IDs]
- ✗ MISSING: [Requirement] → No tasks found
- ⚠ INSUFFICIENT: [Requirement] → Tasks [IDs] don't fully address it

**Quality Issues:**
- Tasks with missing/inadequate fields: [Task IDs]
- Tasks with weak acceptance criteria: [Task IDs] 
- Tasks with unrealistic estimates: [Task IDs]
- Dependency issues: [Description]

**Recommendations:**
Provide specific, actionable suggestions:
- "Add task for [missing requirement] covering [specific aspects]"
- "Task X: Add measurable criteria like '[specific metric]'"
- "Task Y: Increase estimate to [hours] because [rationale]"
- "Add integration test task between components [A] and [B]"

**Final Verdict:**
- REJECT / PASS WITH CONCERNS / PASS
- Overall coverage: [X]% of requirements
- Critical gaps: [List if any]
- Required actions: [List if REJECT or PASS WITH CONCERNS]

Be thorough, reference the actual specification document, and provide constructive, actionable feedback.
"""  
# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
# TODO: 9 - Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
#                      "The answer should be tasks following this exact structure: " \
#                      "Task ID: A unique identifier for tracking purposes\n" \
#                      "Task Title: Brief description of the specific development work\n" \
#                      "Related User Story: Reference to the parent user story\n" \
#                      "Description: Detailed explanation of the technical work required\n" \
#                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
#                      "Estimated Effort: Time or complexity estimation\n" \
#                      "Dependencies: Any tasks that must be completed first"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.

development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria= evaluation_criteria_dev_engineer,
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=10
) 
# Routing Agent
routing_agent = RoutingAgent(openai_api_key, {})    
agents = [
    {
        "name": "product manager agent",
        "description": "Define user stories for a product",
        "func": lambda x: product_manager_knowledge_agent.respond(x)
    },
    {
        "name": "program manager agent",
        "description": "Coordinate project timelines and resources",
        "func": lambda x: program_manager_knowledge_agent.respond(x)
    },
    {
        "name": "development engineer agent",
        "description": "Implement technical solutions and code changes",
        "func": lambda x: development_engineer_knowledge_agent.respond(x)
    }
]

def save_final_output(filename, user_stories, features, tasks):
    with open(filename, "w") as f:
        f.write("=== Final Comprehensive Project Plan ===\n\n")
        f.write("--- User Stories ---\n")
        f.write(user_stories + "\n\n")
        f.write("--- Product Features ---\n")
        f.write(features + "\n\n")
        f.write("--- Engineering Tasks ---\n")
        f.write(tasks + "\n")

routing_agent.agents = agents
# Job function persona support functions
# TODO: 11 - Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.

# --- Support Functions for Routing Agent ---

def product_manager_support_function(step):
    """Generate and evaluate user stories with feedback loop."""
    # Use step to guide story creation
    prompt = f"Based on this requirement: {step}\nGenerate appropriate user stories."
    user_stories = product_manager_knowledge_agent.respond(prompt)
    evaluation = product_manager_evaluation_agent.evaluate(user_stories)
    
    if "feedback" in evaluation:
        improved_prompt = f"""
        Step to address: {step}
        Previous stories: {user_stories}
        Feedback: {evaluation['evaluation_feedback']}
        Please improve the user stories addressing this feedback.
        """
        user_stories = product_manager_knowledge_agent.respond(improved_prompt)
        evaluation = product_manager_evaluation_agent.evaluate(user_stories)
    
    workflow_context['user_stories'] = evaluation["final_response"]
    return evaluation["final_response"]

def program_manager_support_function(step):
    """Generate and evaluate product features with feedback loop."""
    user_stories = workflow_context.get('user_stories', '')
    features_prompt = f"""
    Step to address: {step}
    
    PRODUCT SPECIFICATION:
    {product_spec}
    
    Based on these user stories:
    {user_stories}
    
    Define the product features that match ALL features in the specification's Product Features section (2.1).
    You must include features for: Email Ingestion, Message Classification, Knowledge Base, 
    Response Generation, Routing Logic, and User Interface.
    """
    features = program_manager_knowledge_agent.respond(features_prompt)
    evaluation = program_manager_evaluation_agent.evaluate(features)
    
    if "feedback" in evaluation:
        improved_prompt = f"""
        Step to address: {step}
        User stories: {user_stories}
        Previous features: {features}
        Feedback: {evaluation['evaluation_feedback']}
        Please improve the features addressing this feedback.
        """
        features = program_manager_knowledge_agent.respond(improved_prompt)
        evaluation = program_manager_evaluation_agent.evaluate(features)
    
    workflow_context['features'] = evaluation["final_response"]
    return evaluation["final_response"]

def development_engineer_support_function(step):
    user_stories = workflow_context.get('user_stories', '')
    features = workflow_context.get('features', '')
    
    max_attempts = 3
    for attempt in range(max_attempts):
        tasks_prompt = f"""[existing prompt]"""
        tasks = development_engineer_knowledge_agent.respond(tasks_prompt)
        evaluation_result = development_engineer_evaluation_agent.evaluate(tasks)
        
        # Check if evaluation passed
        if "PASS" in evaluation_result.get("verdict", "") or \
           "Task ID:" in evaluation_result.get("final_response", ""):
            workflow_context['tasks'] = evaluation_result["final_response"]
            return evaluation_result["final_response"]
        
        # If rejected, try again with feedback
        if attempt < max_attempts - 1:
            print(f"Attempt {attempt + 1} rejected, refining...")
            continue
    
    # After max attempts, return best effort with warning
    print("WARNING: Tasks did not pass evaluation after max attempts")
    workflow_context['tasks'] = tasks  # Return last attempt, not feedback
    return tasks
# --- Update Routing Agent to Use Support Functions ---

agents = [
    {
        "name": "product manager agent",
        "description": "Define user stories for a product",
        "func": product_manager_support_function
    },
    {
        "name": "program manager agent",
        "description": "Define product features from user stories",
        "func": program_manager_support_function
    },
    {
        "name": "development engineer agent",
        "description": "Define engineering tasks from features and user stories",
        "func": development_engineer_support_function
    }
]
routing_agent.agents = agents

# --- Run the Workflow ---

print("\n*** Workflow execution started ***\n")
workflow_prompt = f"""
Generate a complete project plan for the Email Router product that covers EVERY requirement in the specification.

The plan must include:
1. User stories addressing ALL objectives in section 1.5 (primary and secondary) and ALL user classes in section 2.2
2. Product features covering ALL 6 features in section 2.1: Email Ingestion, Classification, Knowledge Base, Response Generation, Routing, UI
3. Engineering tasks covering ALL functional requirements (section 3), ALL non-functional requirements (section 4), including infrastructure, testing, security, and compliance

SPECIFICATION:
{product_spec}

Ensure comprehensive coverage - nothing from the specification should be omitted.
"""
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
# Use the action planning agent to extract workflow steps
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print(f"Workflow steps: {workflow_steps}")

completed_steps = []

for step in workflow_steps:
    print(f"\nCurrent step: {step}")
    result = routing_agent.route(step)
    completed_steps.append(result)
    print(f"Result: {result}")


# Print the final output of the workflow (last completed step or a summary)
print("\n=== Final Output of the Workflow ===")
if completed_steps:
    print(completed_steps[-1])
else:
    print("No steps were completed.")

# Optionally, save the final output to a file (adapt as needed for your output structure)
if completed_steps:
    user_stories = workflow_context.get('user_stories', '')
    features = workflow_context.get('features', '')
    tasks = workflow_context.get('tasks', '')
    save_final_output(
        "EmailRouter_ProjectPlan_Output.txt",
        user_stories,
        features,
        tasks
    )
    print("\nFinal output saved to EmailRouter_ProjectPlan_Output.txt")

