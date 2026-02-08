from typing import Literal
from pydantic import BaseModel, Field
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from approval_gui import ApprovalGUI
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from config import llm, get_config, app_diagram
import asyncio

wkp_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
ddg_search = DuckDuckGoSearchRun()

class SectionState(BaseModel):
    topic: str = ''
    title: str = ''
    raw_content: str = ''
    draft_content: str = ''
    final_content: str = ''


class IsSearchNeedDecisionResponse(BaseModel):
    is_search_need: bool = Field(description='is searching is necessary or not?')

async def is_search_need(state: SectionState) -> Literal['need_search', 'not_need_search']:
    print("IS SEARCH NEED NODE")
    structured_llm = llm.with_structured_output(IsSearchNeedDecisionResponse)
    decision: IsSearchNeedDecisionResponse = await structured_llm.ainvoke(f"""
        You are expert content generator.
        for the following general topic and its specific title is need further search?
        TOPIC: "{state.topic}"
        TITLE: "{state.title}"
    """.strip())

    if decision.is_search_need:
        return 'need_search'
    else:
        return 'not_need_search'

class SearchTypeDecisionResponse(BaseModel):
    search_type: Literal['duck_duck_go', 'wikipedia', 'both'] = Field(description='Best search tool')

async def decide_search_type(state: SectionState) -> Literal['duck_duck_go', 'wikipedia', 'both']:
    print("DECIDE SEARCH TYPE NODE")
    structured_llm = llm.with_structured_output(SearchTypeDecisionResponse)
    decision: SearchTypeDecisionResponse = await structured_llm.ainvoke(f"""
        You are expert content generator.
        which search tool is best for the following general topic and its specific title?
        TOPIC: "{state.topic}"
        TITLE: "{state.title}"
    """.strip())
    return decision.search_type.lower()

async def duck_duck_go_search_node(state: SectionState) -> SectionState:
    print("DUCK DUCK GO SEARCH NODE")
    response = await llm.ainvoke(f"""
        You are expert content generator.
        for the following general topic and its specific title, 
        give me search query on duck duck go search engine.
        remember that the query is pure text (not markdown format and any description)
        TOPIC: "{state.topic}"
        TITLE: "{state.title}"
    """.strip())
    search_result = await ddg_search.ainvoke(response.content)
    state.raw_content = f"[DucDucGo search result]: {search_result}"
    return state

async def wikipedia_search_node(state: SectionState) -> SectionState:
    print("WIKIPEDIA SEARCH NODE")
    response = await llm.ainvoke(f"""
        You are expert content generator.
        for the following general topic and its specific title, 
        give me search query on Wikipedia encyclopedia.
        remember that the query is pure text (not markdown format and any description)
        TOPIC: "{state.topic}"
        TITLE: "{state.title}"
    """.strip())
    search_result = await wkp_search.ainvoke(response.content)
    state.raw_content = f"[Wikipedia search result]: {search_result}"
    return state

class SearchQueryResponse(BaseModel):
    duck_duck_go_search_query: str = Field(description='Query to search on DucDuckGo search engine')
    wikipedia_search_query: str = Field(description='Query to search on Wikipedia encyclopedia')

async def both_search_node(state: SectionState) -> SectionState:
    print("BOTH SEARCH NODE")
    structured_llm = llm.with_structured_output(SearchQueryResponse)
    queries: SearchQueryResponse = await structured_llm.ainvoke(f"""
        You are expert content generator.
        for the following general topic and its specific title, 
        give me search query for both DucDuckGo search and engine Wikipedia encyclopedia.
        remember that the query is pure text (not markdown format and any description)
        TOPIC: "{state.topic}"
        TITLE: "{state.title}"
    """.strip())
    
    tasks = [
        asyncio.create_task(
            ddg_search.ainvoke(queries.duck_duck_go_search_query)
        ),
        asyncio.create_task(
            wkp_search.ainvoke(queries.wikipedia_search_query)
        ),
    ]
    ddg_search_result, wkp_search_result = await asyncio.gather(*tasks)

    state.raw_content = f"[DucDucGo search result]: {ddg_search_result}\n\n[Wikipedia search result]: {wkp_search_result}"
    return state

async def background_idea_generator_node(state: SectionState) -> SectionState:
    print("BACKGROUND IDEA NODE")
    response = await llm.ainvoke(f"""
        You are expert content generator.
        for the following general topic and its specific title, 
        tell me background idea
        TOPIC: "{state.topic}"
        TITLE: "{state.title}"
    """.strip())
    state.raw_content = f"[Background idea]: {response.content}"
    return state

async def draft_content_generator_node(state: SectionState) -> SectionState:
    print("DRAFT CONTENT GENERATOR NODE")
    response = await llm.ainvoke(f"""
        You are expert content generator.
        for the following general topic, its specific title and raw content, 
        organize the idea in proper markdown format
        TOPIC: "{state.topic}"
        TITLE: "{state.title}"
        RAW CONTENT: "{state.raw_content}"
    """.strip())
    state.draft_content = response.content
    return state

async def section_human_approval_node(state: SectionState) -> SectionState:
    print("SECTION HUMAN APPROVAL NODE")
    final_content = interrupt({
        'interrupt_state': state
    })
    state.final_content = final_content
    return state

async def default_node(state: SectionState) -> SectionState:
    # print("DEFAULT NODE")
    return state

IS_SEARCH_NEED = 'is_search_need'
DECIDE_SEARCH_TYPE = 'decide_search_type'
DDG_SEARCH = 'ddg_search'
WKP_SEARCH = 'wkp_search'
BOTH_SEARCH = 'both_search'
BACKGROUND_IDEA = 'background_idea'
DRAFT_CONTENT = 'draft_content'
SECTION_HUMAN_APPROVAL = 'section_human_approval'

section_graph = StateGraph(SectionState)
section_graph.add_node(IS_SEARCH_NEED, default_node)
section_graph.add_node(DECIDE_SEARCH_TYPE, default_node)
section_graph.add_node(DDG_SEARCH, duck_duck_go_search_node)
section_graph.add_node(WKP_SEARCH, wikipedia_search_node)
section_graph.add_node(BOTH_SEARCH, both_search_node)
section_graph.add_node(BACKGROUND_IDEA, background_idea_generator_node)
section_graph.add_node(DRAFT_CONTENT, draft_content_generator_node)
section_graph.add_node(SECTION_HUMAN_APPROVAL, section_human_approval_node)

section_graph.add_edge(START, IS_SEARCH_NEED)
section_graph.add_conditional_edges(
    IS_SEARCH_NEED,
    is_search_need,
    {
        'need_search': DECIDE_SEARCH_TYPE,
        'not_need_search': BACKGROUND_IDEA
    }
)
section_graph.add_conditional_edges(
    DECIDE_SEARCH_TYPE,
    decide_search_type,
    {
        'duck_duck_go': DDG_SEARCH,
        'wikipedia': WKP_SEARCH,
        'both': BOTH_SEARCH
    }
)
section_graph.add_edge(BACKGROUND_IDEA, DRAFT_CONTENT)
section_graph.add_edge(DDG_SEARCH, DRAFT_CONTENT)
section_graph.add_edge(WKP_SEARCH, DRAFT_CONTENT)
section_graph.add_edge(BOTH_SEARCH, DRAFT_CONTENT)
section_graph.add_edge(DRAFT_CONTENT, SECTION_HUMAN_APPROVAL)
section_graph.add_edge(SECTION_HUMAN_APPROVAL, END)

section_app = section_graph.compile(checkpointer=MemorySaver())
async def run_section_graph(state: SectionState) -> SectionState:
    print("START SECTION, Title:", state.title)
    config = get_config()
    result = await section_app.ainvoke(state, config)
    interrupt_state: SectionState = result['__interrupt__'][0].value['interrupt_state']
    final_content = interrupt_state.draft_content
    
    try:
        gui = ApprovalGUI(
            topic=interrupt_state.topic,
            section=interrupt_state.title,
            content=interrupt_state.draft_content
        )
        gui.run()
        final_content = gui.content
    except:
        pass

    if not final_content:
        final_content = interrupt_state.draft_content

    end_of_task = await section_app.ainvoke(Command(resume=final_content), config)
    final_state = SectionState(**end_of_task)

    print("END SECTION")
    return final_state


if __name__ == '__main__':
    final_state = asyncio.run(
        run_section_graph(SectionState(
            topic='Computer',
            title='Programming Language',
        ))
    )    
    # app_diagram(section_app, './note_taker/section_note_taker_app')
    print(final_state.final_content)
