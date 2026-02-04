from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from operator import add
from pydantic import BaseModel, Field

llm = ChatOllama(model='smollm2:135m')

class AgentState(TypedDict):
    messages: Annotated[list, add]
    user_input: str
    query_type: str
    confidence: str
    processing_step: str
    result: str
    route_taken: str
    llm_reasoning: str


class ClassificationResponse(BaseModel):
    category: Literal["greeting", "question", "command", "feedback", "fallback"] = Field(description='category name')
    confidence: float = Field(description='number between 0.0 and 1.0')
    reasoning: str = Field(description='brief explanation')

def input_node(state: AgentState) -> AgentState:
    classification_prompt = f"""
Analyze this user input and classify it into ONE of these categories:
- GREETING: Social pleasantries, hellos, goodbyes
- QUESTION: Information requests, queries about facts or how-to
- COMMAND: Action requests, instructions to do something
- FEEDBACK: Opinions, complaints, praise, suggestions
- UNCLEAR: Ambiguous or off-topic inputs

User input: "{state['user_input']}"
""".strip()
    
    structured_llm = llm.with_structured_output(ClassificationResponse)
    response: ClassificationResponse = structured_llm.invoke(classification_prompt)
    return {
        'messages': [HumanMessage(state['user_input'])],
        'query_type': response.category,
        'confidence': response.confidence,
        'llm_reasoning': response.reasoning,
        'processing_step': 'classified'
    }

def greeting_handler(state: AgentState) -> AgentState:
    prompt = f"""
Generate a friendly, warm greeting response to: "{state['user_input']}"
Keep it conversational and natural. 1-2 sentences maximum.
""".strip()
    
    response = llm.invoke(prompt)
    return {
        'messages': [AIMessage(response.content)],
        'result': response.content,
        'route_taken': 'greeting_path',
        'processing_step': 'greeting_completed'
    }

def question_handler(state: AgentState) -> AgentState:
    prompt = f"""
Answer this question thoroughly but concisely: "{state['user_input']}"
Provide a helpful, accurate response. Include examples if relevant. Keep it under 150 words.
""".strip()
    
    response = llm.invoke(prompt)
    return {
        'messages': [AIMessage(response.content)],
        'result': response.content,
        'route_taken': 'question_path',
        'processing_step': 'question_completed'
    }

def command_handler(state: AgentState) -> AgentState:
    prompt = f"""
The user requested: "{state['user_input']}"
Acknowledge the request and outline the steps you would take to fulfill it. Be specific but concise (2-3 sentences).
""".strip()
    
    response = llm.invoke(prompt)
    return {
        'messages': [AIMessage(response.content)],
        'result': response.content,
        'route_taken': 'command_path',
        'processing_step': 'command_completed'
    }

def feedback_handler(state: AgentState) -> AgentState:
    prompt = f"""
The user provided feedback: "{state['user_input']}"
Respond empathetically, acknowledging their input. Thank them or address their concern appropriately. Keep it brief and sincere.
""".strip()
    
    response = llm.invoke(prompt)
    return {
        'messages': [AIMessage(response.content)],
        'result': response.content,
        'route_taken': 'feedback_path',
        'processing_step': 'feedback_completed'
    }

def fallback_handler(state: AgentState) -> AgentState:
    prompt = f"""
The user said: "{state['user_input']}" but it's unclear what they want.
Politely ask for clarification. Be helpful and suggest what kind of information you can provide. Keep it brief.
""".strip()
    
    response = llm.invoke(prompt)
    return {
        'messages': [AIMessage(response.content)],
        'result': response.content,
        'route_taken': 'fallback_path',
        'processing_step': 'fallback_completed'
    }

def output_node(state: AgentState) -> AgentState:
    return {
        'messages': [AIMessage(f'[{state['route_taken']}] {state['result']}')],
        'processing_step': 'completed'
    }

def route_query(state: AgentState) -> Literal["greeting", "question", "command", "feedback", "fallback"]:
    query_type = state['query_type'].upper()
    confidence = state['confidence']

    if confidence < 0.6:
        return 'fallback'
    
    if query_type == 'GREETING':
        return 'greeting'
    if query_type == 'QUESTION':
        return 'question'
    if query_type == 'COMMAND':
        return 'command'
    if query_type == 'FEEDBACK':
        return 'feedback'
    else:
        return 'fallback'
    

workflow = StateGraph(AgentState)
INPUT = 'input'
GREETING = 'greeting'
QUESTION = 'question'
COMMAND = 'command'
FEEDBACK = 'feedback'
FALLBACK = 'fallback'
OUTPUT = 'output'

workflow.add_node(INPUT, input_node)
workflow.add_node(GREETING, greeting_handler)
workflow.add_node(QUESTION, question_handler)
workflow.add_node(COMMAND, command_handler)
workflow.add_node(FEEDBACK, feedback_handler)
workflow.add_node(FALLBACK, fallback_handler)
workflow.add_node(OUTPUT, output_node)

workflow.add_edge(START, INPUT)
workflow.add_conditional_edges(
    INPUT,
    route_query,
    {
        GREETING: GREETING,
        QUESTION: QUESTION,
        COMMAND: COMMAND,
        FEEDBACK: FEEDBACK,
        FALLBACK: FALLBACK
    }
)
workflow.add_edge(GREETING, OUTPUT)
workflow.add_edge(QUESTION, OUTPUT)
workflow.add_edge(COMMAND, OUTPUT)
workflow.add_edge(FEEDBACK, OUTPUT)
workflow.add_edge(FALLBACK, OUTPUT)
workflow.add_edge(OUTPUT, END)

app = workflow.compile()

def run_workflow(user_input: str):
    """
    Helper function to run the workflow
    """
    print(f"\n{'=' * 70}")
    print(f"RUNNING WORKFLOW: '{user_input}'")
    print(f"{'=' * 70}")

    initial_state = {
        "messages": [],
        "user_input": user_input,
        "query_type": "",
        "confidence": 0.0,
        "processing_step": "initialized",
        "result": "",
        "route_taken": "",
        "llm_reasoning": ""
    }

    final_state = app.invoke(initial_state)

    print(f"\n{'=' * 70}")
    print(f"WORKFLOW COMPLETED")
    print(f"{'=' * 70}")
    print(f" Results:")
    print(f"  - Query Type: {final_state['query_type']}")
    print(f"  - Confidence: {final_state['confidence']:.2f}")
    print(f"  - Route Taken: {final_state['route_taken']}")
    print(f"  - LLM Reasoning: {final_state['llm_reasoning']}")
    print(f"\n Final Response:")
    print(f"  {final_state['result']}")
    print(f"{'=' * 70}")

    print("$"*70)
    print(f" Messages:")
    for message in final_state['messages']:
        if isinstance(message, HumanMessage):
            print(f"\tHuman: {message.content}")
        else:
            print(f"\tAI: {message.content}")
    print("$"*70)

    return final_state


print("\n" + "=" * 70)
print("DEMO: Testing LLM-Powered Conditional Routing")
print("=" * 70)

# Test 1: Simple greeting
print("\n Test 1: Simple Greeting")
run_workflow("Hey there! How's it going?")

# Test 2: Technical question
print("\n Test 2: Technical Question")
run_workflow("What is LangGraph and how does it differ from LangChain?")

# Test 3: Action command
print("\n Test 3: Action Command")
run_workflow("Can you create a summary of the latest AI research papers?")

# Test 4: User feedback
print("\n Test 4: User Feedback")
run_workflow("This tool is amazing! The responses are really helpful.")

# Test 5: Complex question
print("\n Test 5: Complex Question")
run_workflow("How do conditional edges improve workflow efficiency compared to linear flows?")

# Test 6: Ambiguous input
print("\n Test 6: Ambiguous Input")
run_workflow("hmm...I see ")

# Test 7: Multi-intent input (tests LLM classification)
print("\n Test 7: Mixed Intent")
run_workflow("Hi! Can you explain how neural networks work?")
