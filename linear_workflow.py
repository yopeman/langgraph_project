from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from operator import add

class AgentSate(TypedDict):
    messages: Annotated[list, add]
    user_input: str
    processing_step: str
    result: str


def display(state_name, state: AgentSate):
    print('-'*75)
    print(state_name)
    print('User input:', state['user_input'])
    print('Messages:')
    for msg in state['messages']:
        if isinstance(msg, HumanMessage):
            print('\tUser:', msg.content)
        elif isinstance(msg, AIMessage):
            print('\tAI:', msg.content)
        else:
            print('Unknown:', msg)
    print('Processing step:', state['processing_step'])
    print('Result: ', state['result'])


def input_node(state: AgentSate) -> AgentSate:
    display("INPUT NODE:", state)

    return {
        'messages': [HumanMessage(state['user_input'])],
        'processing_step': 'input_received'
    }

def analyze_node(state: AgentSate) -> AgentSate:
    display("ANALYZE NODE:", state)
    usr_txt = state['user_input'].lower()

    if 'weather' in usr_txt:
        analysis = 'Weather related query detected'
    elif 'hello' in usr_txt or 'hi' in usr_txt:
        analysis = 'Greeting detected'
    else:
        analysis = 'General query detected'

    return {
        'messages': [AIMessage(f'Analysis: {analysis}')],
        'processing_step': 'analyzed'
    }

def process_node(state: AgentSate) -> AgentSate:
    display("PROCESS", state)
    usr_txt = state['user_input'].lower()

    if 'weather' in usr_txt:
        response = 'I would check the weather API for current conditions.'
    elif 'hello' in usr_txt or 'hi' in usr_txt:
        response = 'Hello! How can I help you today?'
    else:
        response = f'I receive your message: `{state['user_input']}`'

    return {
        'messages': [AIMessage(response)],
        'processing_step': 'processed',
        'result': response
    }

def output_node(state: AgentSate) -> AgentSate:
    display("OUTPUT NODE:", state)
    final_msg = f'Final Response: {state['result']}'
    return {
        'messages': [AIMessage(final_msg)],
        'processing_step': 'completed'
    }

workflow = StateGraph(AgentSate)
INPUT = 'input'
ANALYZE = 'analyze'
PROCESS = 'process'
OUTPUT = 'output'

workflow.add_node(INPUT, input_node)
workflow.add_node(ANALYZE, analyze_node)
workflow.add_node(PROCESS, process_node)
workflow.add_node(OUTPUT, output_node)

workflow.add_edge(START, INPUT)
workflow.add_edge(INPUT, ANALYZE)
workflow.add_edge(ANALYZE, PROCESS)
workflow.add_edge(PROCESS, OUTPUT)
workflow.add_edge(OUTPUT, END)

app = workflow.compile()

def run_workflow(usr_inp):
    print(f"\n{'=' * 75}")
    print(f"RUNNING WORKFLOW")
    print(f"{'=' * 70}")

    init_state: AgentSate = {
        'messages': [],
        'user_input': usr_inp,
        'processing_step': 'initialized',
        'result': ''
    }

    final_state = app.invoke(init_state)
    display("FINAL STATE:", final_state)
    return final_state


# # Test 1: Greeting
# print("\n Test 1: Greeting")
# run_workflow("Hello!")

# # Test 2: Weather query
# print("\n Test 2: Weather Query")
# run_workflow("What's the weather like?")

# # Test 3: General query
# print("\n Test 3: General Query")
# run_workflow("Tell me about LangGraph")


from IPython.display import Image, display
linear_workflow_diagram = Image(
        app.get_graph()
        .draw_mermaid_png()
    )