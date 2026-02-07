
from langchain_ollama import ChatOllama
import tkinter as tk
import uuid

# llm = ChatOllama(model='smollm2:135m')
# llm = ChatOllama(model='gemma3:4b')
# llm = ChatOllama(model='llama3.1:8b')
llm = ChatOllama(model='llama3.2:3b')

def get_config():
    config = {'configurable': {'thread_id': str(uuid.uuid4())}}
    return config

def app_diagram(app, filename):
    from IPython.display import Image as IP_Image
    from PIL import ImageTk, Image as PIL_Image
    import io

    diagram = IP_Image(
        app.get_graph()
        .draw_mermaid_png()
    )
    bytes = diagram.data

    with open(f"{filename}.png", "wb") as f:
        f.write(bytes)

    root = tk.Tk()
    root.title(filename)
    frame = tk.Frame(root)
    pil_img = PIL_Image.open(io.BytesIO(bytes))
    tk_img = ImageTk.PhotoImage(pil_img)
    label = tk.Label(frame, image=tk_img)
    label.pack()
    root.mainloop()
