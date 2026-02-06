from typing import TypedDict, Annotated, List, Literal, Dict
from operator import add
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from approval_gui import ApprovalGUI
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from config import llm, config

wkp_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
ddg_search = DuckDuckGoSearchRun()

class SectionState(TypedDict):
    title: str
    raw_content: str
    draft_content: str
    final_content: str

class NoteState(TypedDict):
    topic: str
    sections_content: List[SectionState]
    current_section_index: int
    draft_note: str
    final_note: str


class PlanResponse(BaseModel):
    sections: List[str] = Field(description='List of ideas')

def planning_node(state: NoteState) -> NoteState:
    print("PLANING NODE:", end='')
    topic = state['topic']
    structured_llm = llm.with_structured_output(PlanResponse)
    response: PlanResponse = structured_llm.invoke(f"""
        You are expert content generator.
        tell me list of ideas of to create best content on the following topic
        TOPIC: "{topic}"
    """.strip())

    for section in response.sections:
        state['sections_content'].append(
            SectionState({
                'title': section,
                'raw_content': '',
                'draft_content': '',
                'final_content': ''
            })
        )
    print('pass')
    return state

def is_final_loop(state: NoteState) -> Literal['final_loop', 'not_final_loop']:
    print("\nIS FINAL LOOP NODE:", end='')
    total_section = len(state['sections_content'])
    current_section_index = state['current_section_index']

    print('pass')
    if current_section_index >= total_section:
        return 'final_loop'
    else:
        return 'not_final_loop'

class IsSearchNeedDecisionResponse(BaseModel):
    is_search_need: bool = Field(description='is searching is necessary')

def is_search_need(state: NoteState) -> Literal['need_search', 'not_need_search']:
    print("IS SEARCH NEED NODE:", end='')
    topic = state['topic']
    section = state['sections_content'][state['current_section_index']]['title']
    structured_llm = llm.with_structured_output(IsSearchNeedDecisionResponse)
    decision: IsSearchNeedDecisionResponse = structured_llm.invoke(f"""
        You are expert content generator.
        for the following topic and section is need further search?
        TOPIC: "{topic}"
        SECTION: "{section}"
    """.strip())

    print('pass')
    if decision.is_search_need:
        return 'need_search'
    else:
        return 'not_need_search'

class SearchTypeDecisionResponse(BaseModel):
    search_type: Literal['duck_duck_go', 'wikipedia', 'both'] = Field(description='Best search result')

def decide_search_type(state: NoteState) -> Literal['duck_duck_go', 'wikipedia', 'both']:
    print("DECIDE SEARCH TYPE NODE:", end='')
    topic = state['topic']
    section = state['sections_content'][state['current_section_index']]['title']
    structured_llm = llm.with_structured_output(SearchTypeDecisionResponse)
    decision: SearchTypeDecisionResponse = structured_llm.invoke(f"""
        You are expert content generator.
        which search tool is best for the following topic and section?
        TOPIC: "{topic}"
        SECTION: "{section}"
    """.strip())
    print('pass')
    return decision.search_type.lower()

def duck_duck_go_search_node(state: NoteState) -> NoteState:
    print("DUCK DUCK GO SEARCH NODE:", end='')
    topic = state['topic']
    section = state['sections_content'][state['current_section_index']]['title']
    response = llm.invoke(f"""
        You are expert content generator.
        for the following topic and section, 
        give me search query on duck duck go search engine.
        remember that the query is pure text (not markdown format and any description)
        TOPIC: "{topic}"
        SECTION: "{section}"
    """.strip())
    search_result = ddg_search.invoke(response.content)
    state['sections_content'][state['current_section_index']]['raw_content'] = f"[DucDucGo search result]: {search_result}"
    print('pass')
    return state

def wikipedia_search_node(state: NoteState) -> NoteState:
    print("WIKIPEDIA SEARCH NODE:", end='')
    topic = state['topic']
    section = state['sections_content'][state['current_section_index']]['title']
    response = llm.invoke(f"""
        You are expert content generator.
        for the following topic and section, 
        give me search query on Wikipedia encyclopedia.
        remember that the query is pure text (not markdown format and any description)
        TOPIC: "{topic}"
        SECTION: "{section}"
    """.strip())
    search_result = wkp_search.invoke(response.content)
    state['sections_content'][state['current_section_index']]['raw_content'] = f"[Wikipedia search result]: {search_result}"
    print('pass')
    return state

class SearchQueryResponse(BaseModel):
    duck_duck_go_search_query: str = Field(description='Query to search on DucDuckGo search engine')
    wikipedia_search_query: str = Field(description='Query to search on Wikipedia encyclopedia')

def both_search_node(state: NoteState) -> NoteState:
    print("BOTH SEARCH NODE:", end='')
    topic = state['topic']
    section = state['sections_content'][state['current_section_index']]['title']
    structured_llm = llm.with_structured_output(SearchQueryResponse)
    queries: SearchQueryResponse = structured_llm.invoke(f"""
        You are expert content generator.
        for the following topic and section, 
        give me search query for both DucDuckGo search and engine Wikipedia encyclopedia.
        remember that the query is pure text (not markdown format and any description)
        TOPIC: "{topic}"
        SECTION: "{section}"
    """.strip())
    ddg_search_result = ddg_search.invoke(queries.duck_duck_go_search_query)
    wkp_search_result = wkp_search.invoke(queries.wikipedia_search_query)
    state['sections_content'][state['current_section_index']]['raw_content'] = f"[DucDucGo search result]: {ddg_search_result}\n\n[Wikipedia search result]: {wkp_search_result}"
    print('pass')
    return state

def background_idea_generator_node(state: NoteState) -> NoteState:
    print("BACKGROUND IDEA NODE:", end='')
    topic = state['topic']
    section = state['sections_content'][state['current_section_index']]['title']
    response = llm.invoke(f"""
        You are expert content generator.
        for the following topic and section, 
        tell me background idea (generate maximum 3 paragraph)
        TOPIC: "{topic}"
        SECTION: "{section}"
    """.strip())
    state['sections_content'][state['current_section_index']]['raw_content'] = f"[Background idea]: {response.content}"
    print('pass')
    return state

def draft_content_generator_node(state: NoteState) -> NoteState:
    print("DRAFT CONTENT GENERATOR NODE:", end='')
    topic = state['topic']
    section = state['sections_content'][state['current_section_index']]['title']
    raw_content = state['sections_content'][state['current_section_index']]['raw_content']
    response = llm.invoke(f"""
        You are expert content generator.
        for the following topic, section and raw content, 
        tell me organized idea (generate maximum 5 paragraph)
        TOPIC: "{topic}"
        SECTION: "{section}"
        RAW CONTENT: "{raw_content}"
    """.strip())
    state['sections_content'][state['current_section_index']]['draft_content'] = response.content
    print('pass')
    return state
    
def section_human_approval_node(state: NoteState) -> NoteState:
    print("SECTION HUMAN APPROVAL NODE:", end='')
    section = state['sections_content'][state['current_section_index']]
    final_content = interrupt({'type': 'section_human_approval', 'section': section['title'], 'content': section['draft_content']})
    state['sections_content'][state['current_section_index']]['final_content'] = final_content
    state['current_section_index'] += 1
    print('pass')
    return state

def final_content_generator_node(state: NoteState) -> NoteState:
    print("FINAL CONTENT GENERATOR NODE:", end='')
    topic = state['topic']
    current_content = ''
    
    for i, section in enumerate(state['sections_content'], start=1):
        current_content += f"""
            ## Section {i}: {section['title']}
            {section['final_content']} \n\n
        """

    response = llm.invoke(f"""
        You are expert content generator.
        for the following topic and its content, 
        tell me organized and improved idea in proper markdown format
        TOPIC: "{topic}"
        CONTENT: "{current_content}"
    """.strip())
    state['draft_note'] = response.content
    print('pass')
    return state

def final_human_approval_node(state: NoteState) -> NoteState:
    print("FINAL HUMAN APPROVAL NODE:", end='')
    draft_note = state['draft_note']
    final_note = interrupt({'type': 'final_human_approval', 'content': draft_note})
    state['final_note'] = final_note
    print('pass')
    return state

def improve_markdown_node(state: NoteState) -> NoteState:
    print("IMPROVE MARKDOWN NODE:", end='')
    final_note = state['final_note']
    response = llm.invoke(f"""
        Improve the following markdown text:
        <text>
        {final_note}
        </text>
    """.strip())
    state['final_note'] = response.content
    print('pass')
    return state

def default_node(state: NoteState) -> NoteState:
    # print("DEFAULT NODE:")
    return state

workflow = StateGraph(NoteState)
PLAN = 'plan'
IS_FINAL = 'is_final_loop'
IS_SEARCH_NEED = 'is_search_need'
DECIDE_SEARCH_TYPE = 'decide_search_type'
DDG_SEARCH = 'ddg_search'
WKP_SEARCH = 'wkp_search'
BOTH_SEARCH = 'both_search'
BACKGROUND_IDEA = 'background_idea'
DRAFT_CONTENT = 'draft_content'
SECTION_HUMAN_APPROVAL = 'section_human_approval'
FINAL_CONTENT = 'final_content'
FINAL_HUMAN_APPROVAL = 'final_human_approval'
IMPROVE_MARKDOWN = 'improve_markdown'

workflow.add_node(PLAN, planning_node)
workflow.add_node(IS_FINAL, default_node)
workflow.add_node(IS_SEARCH_NEED, default_node)
workflow.add_node(DECIDE_SEARCH_TYPE, default_node)
workflow.add_node(DDG_SEARCH, duck_duck_go_search_node)
workflow.add_node(WKP_SEARCH, wikipedia_search_node)
workflow.add_node(BOTH_SEARCH, both_search_node)
workflow.add_node(BACKGROUND_IDEA, background_idea_generator_node)
workflow.add_node(DRAFT_CONTENT, draft_content_generator_node)
workflow.add_node(SECTION_HUMAN_APPROVAL, section_human_approval_node)
workflow.add_node(FINAL_CONTENT, final_content_generator_node)
workflow.add_node(FINAL_HUMAN_APPROVAL, final_human_approval_node)
workflow.add_node(IMPROVE_MARKDOWN, improve_markdown_node)

workflow.add_edge(START, PLAN)
workflow.add_edge(PLAN, IS_FINAL)
workflow.add_conditional_edges(
    IS_FINAL,
    is_final_loop,
    {
        'not_final_loop': IS_SEARCH_NEED,
        'final_loop': FINAL_CONTENT
    }
)
workflow.add_conditional_edges(
    IS_SEARCH_NEED,
    is_search_need,
    {
        'need_search': DECIDE_SEARCH_TYPE,
        'not_need_search': BACKGROUND_IDEA
    }
)
workflow.add_conditional_edges(
    DECIDE_SEARCH_TYPE,
    decide_search_type,
    {
        'duck_duck_go': DDG_SEARCH,
        'wikipedia': WKP_SEARCH,
        'both': BOTH_SEARCH
    }
)
workflow.add_edge(BACKGROUND_IDEA, DRAFT_CONTENT)
workflow.add_edge(DDG_SEARCH, DRAFT_CONTENT)
workflow.add_edge(WKP_SEARCH, DRAFT_CONTENT)
workflow.add_edge(BOTH_SEARCH, DRAFT_CONTENT)
workflow.add_edge(DRAFT_CONTENT, SECTION_HUMAN_APPROVAL)
workflow.add_edge(SECTION_HUMAN_APPROVAL, IS_FINAL)
workflow.add_edge(FINAL_CONTENT, FINAL_HUMAN_APPROVAL)
workflow.add_edge(FINAL_HUMAN_APPROVAL, IMPROVE_MARKDOWN)
workflow.add_edge(IMPROVE_MARKDOWN, END)

app = workflow.compile(checkpointer=MemorySaver())
def run_note_taker(topic:str):
    print("START:pass")
    initial_state = NoteState({
        'topic': topic,
        'sections_content': [],
        'current_section_index': 0,
        'draft_note': '',
        'final_note': ''
    })
    result = app.invoke(initial_state, config)

    while '__interrupt__' in result:
        payload = result['__interrupt__'][0].value
        match payload['type']:
            case 'section_human_approval':
                section = payload['section']
                content = payload['content']
                gui = ApprovalGUI(topic=topic, section=section, content=content)
                gui.run()
                improved_content = gui.content
            case 'final_human_approval':
                content = payload['content']
                gui = ApprovalGUI(topic=topic, content=content)
                gui.run()
                improved_content = gui.content
            case _:
                raise ValueError(f"Unknown interrupt type: {payload['type']}")
            
        result = app.invoke(Command(resume=improved_content), config)

    print("END:pass")
    print(result['final_note'])
    return result
