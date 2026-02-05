import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from note_taker import app

class NoteTakerAgentGUI:
    def __init__(self):
        root = tk.Tk()
        root.title("Content Generator Agent")
        # root.geometry("600x450")
        self.content = None

        container = tk.Frame(root, padx=30, pady=30)
        container.pack(fill="both", expand=True)

        topic_label = tk.Label(container, text="Topic Name:")
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
        root.mainloop()

    def generate_content(self):
        topic = self.topic_entry.get().strip()
        if not topic:
            messagebox.showwarning("Missing Topic", "Please enter a topic name.")
            return

        content = app.invoke({'topic': topic})
        self.content = content['final_note']
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