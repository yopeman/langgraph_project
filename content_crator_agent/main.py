import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from note_graph import note_app, run_note_graph, NoteState
from section_graph import section_app
from IPython.display import Image as IP_Image
from PIL import ImageTk, Image as PIL_Image
import asyncio
import io


class NoteTakerAgentGUI:
    def __init__(self):
        root = tk.Tk()
        root.title("Content Generator Agent")
        # root.geometry("600x450")
        self.content = None

        container = tk.Frame(root, padx=30, pady=30)
        container.pack(fill="both", expand=True)

        topic_label = tk.Label(container, text="Topic Name")
        topic_label.pack(anchor="w", pady=(0, 5))

        self.topic_entry = tk.Entry(container)
        self.topic_entry.pack(fill="x", pady=(0, 10))

        generate_button = tk.Button(
            container,
            text="Generate",
            command=self.generate_content,
            width=15
        )
        generate_button.pack(anchor="w", pady=(0, 15))

        button_frame = tk.Frame(container)
        button_frame.pack(fill="x")

        save_button = tk.Button(
            button_frame,
            text="Save",
            command=self.save_content,
            width=12
        )
        save_button.pack(side="left", padx=(0, 10))

        discard_button = tk.Button(
            button_frame,
            text="Discard",
            command=self.discard_all,
            width=12
        )
        discard_button.pack(side="left")


        note_taker_workflow_diagram = IP_Image(section_app.get_graph().draw_mermaid_png())
        image_bytes = note_taker_workflow_diagram.data
        pil_image = PIL_Image.open(io.BytesIO(image_bytes))
        tk_image = ImageTk.PhotoImage(pil_image)
        img_label = tk.Label(container, image=tk_image)
        img_label.pack(anchor='center')

        root.mainloop()

    def generate_content(self):
        topic = self.topic_entry.get().strip()
        if not topic:
            messagebox.showwarning("Missing Topic", "Please enter a topic name.")
            return

        messagebox.showinfo("Acknowledgement", f"Generating Content: \n\n{topic}")
        content = asyncio.run(run_note_graph(NoteState(topic=topic)))
        self.content = content.improved_note
        messagebox.showinfo("Congratulation", f"Generate: \n\n{self.content[:250]}...")

    def save_content(self):
        content = self.content
        topic = self.topic_entry.get().strip()

        if not content:
            messagebox.showwarning("Nothing to Save", "No generated content found.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = topic[:15].replace(" ", "_")
        default_filename = f"{safe_topic}_{timestamp}.md"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".md",
            initialfile=default_filename,
            filetypes=[("Markdown Files", "*.md"), ("All Files", "*.*")]
        )

        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Saved", "File saved successfully.")

    def discard_all(self):
        self.topic_entry.delete(0, tk.END)
        self.content = ''

NoteTakerAgentGUI()