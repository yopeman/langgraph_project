from note_taker import app
from IPython.display import Image

workflow_diagram = Image(
    app.get_graph().draw_mermaid_png()
)

with open("./workflow.png", "wb") as f:
    f.write(workflow_diagram.data)
