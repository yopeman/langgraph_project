from typing import List
from pydantic import BaseModel, Field
from approval_gui import ApprovalGUI
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from section_graph import SectionState, run_section_graph
from config import llm, get_config, app_diagram
import asyncio

class NoteState(BaseModel):
    topic: str
    sections: List[SectionState] = Field(default_factory=list)
    draft_note: str = ''
    final_note: str = ''
    improved_note: str = ''


class PlanResponse(BaseModel):
    titles: List[str] = Field(description='List of related titles')

async def planning_node(state: NoteState) -> NoteState:
    print("PLANING NODE:", end='')
    structured_llm = llm.with_structured_output(PlanResponse)
    response: PlanResponse = await structured_llm.ainvoke(f"""
        You are expert content generator.
        tell me list of related titles of the following topic
        TOPIC: "{state.topic}"
    """.strip())

    for title in response.titles:
        state.sections.append(
            SectionState(
                topic=state.topic,
                title=title
            )
        )
    return state

async def section_content_generator_node(state: NoteState) -> NoteState:
    print("SECTION CONTENT GENERATOR NODE:", end='')
    tasks = [
        asyncio.create_task(
            run_section_graph(section)
        ) 
        for section in state.sections
    ]
    sections = await asyncio.gather(*tasks)
    state.sections = sections
    return state

async def draft_note_generator_node(state: NoteState) -> NoteState:
    print("FINAL CONTENT GENERATOR NODE:", end='')
    current_content = ''
    
    for i, section in enumerate(state.sections, start=1):
        current_content += f"""
            ## Section {i}: {section.title}
            {section.final_content} \n\n
        """

    response = await llm.ainvoke(f"""
        You are expert content generator.
        for the following topic and its content, 
        tell me organized and improved idea in proper markdown format
        TOPIC: "{state.topic}"
        CONTENT: "{current_content}"
    """.strip())
    state.draft_note = response.content
    return state

async def final_human_approval_node(state: NoteState) -> NoteState:
    print("FINAL HUMAN APPROVAL NODE:", end='')
    final_note = interrupt({
        'interrupt_state': state
    })
    state.final_note = final_note
    return state

async def improve_markdown_node(state: NoteState) -> NoteState:
    print("IMPROVE MARKDOWN NODE:", end='')
    response = await llm.ainvoke(f"""
        Improve the following markdown text:
        <text>
        {state.final_note}
        </text>
    """.strip())
    state.improved_note = response.content
    return state

PLAN = 'plan'
SECTION_CONTENT_GENERATOR = 'section_content_generator'
DRAFT_NOTE = 'draft_note'
FINAL_HUMAN_APPROVAL = 'final_human_approval'
IMPROVE_MARKDOWN = 'improve_markdown'

note_graph = StateGraph(NoteState)
note_graph.add_node(PLAN, planning_node)
note_graph.add_node(SECTION_CONTENT_GENERATOR, section_content_generator_node)
note_graph.add_node(DRAFT_NOTE, draft_note_generator_node)
note_graph.add_node(FINAL_HUMAN_APPROVAL, final_human_approval_node)
note_graph.add_node(IMPROVE_MARKDOWN, improve_markdown_node)

note_graph.add_edge(START, PLAN)
note_graph.add_edge(PLAN, SECTION_CONTENT_GENERATOR)
note_graph.add_edge(SECTION_CONTENT_GENERATOR, DRAFT_NOTE)
note_graph.add_edge(DRAFT_NOTE, FINAL_HUMAN_APPROVAL)
note_graph.add_edge(FINAL_HUMAN_APPROVAL, IMPROVE_MARKDOWN)
note_graph.add_edge(IMPROVE_MARKDOWN, END)

note_app = note_graph.compile(checkpointer=MemorySaver())
async def run_note_graph(state: NoteState):
    print("START NOTE")
    config = get_config()
    result = await note_app.ainvoke(state, config)
    interrupt_state: NoteState = result['__interrupt__'][0].value['interrupt_state']
    final_note = interrupt_state.draft_note
    
    try:
        gui = ApprovalGUI(
            topic=interrupt_state.topic,
            content=interrupt_state.draft_note
        )
        gui.run()
        final_note = gui.content
    except:
        pass

    if not final_note:
        final_note = interrupt_state.final_note

    end_of_task = await note_app.ainvoke(Command(resume=final_note), config)
    final_state = NoteState(**end_of_task)

    print("END NOTE")
    return final_state


if __name__ == '__main__':
    final_state = asyncio.run(
        run_note_graph(NoteState(
            topic='Artificial Intelligence (AI)'
        ))
    )    
    # app_diagram(note_app, './note_taker/note_taker_app')
    print(final_state.improved_note)
