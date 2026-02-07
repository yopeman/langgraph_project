import tkinter as tk
from tkinter import messagebox
from tkhtmlview import HTMLLabel
import markdown
from config import llm

class ApprovalGUI:
    def __init__(self, topic, content, section = None):
        title = "Section Approval" if section else "Final Approval"
        self.root = tk.Tk()
        self.root.title(title)
        # root.geometry("520x450")
        self.topic = topic
        self.section = section
        self.content = content

        container = tk.Frame(self.root, padx=15, pady=15)
        container.pack(fill="both", expand=True)

        topic_label = tk.Label(
            container,
            text=f"📝 Topic: {topic} ({title})",
            font=("Arial", 16, "bold")
        )
        topic_label.pack(anchor="w", pady=(0, 10))

        if section:
            section_label = tk.Label(
                container,
                text=f"✍️ Section: {section}",
                font=("Arial", 14, "bold")
            )
            section_label.pack(anchor="w", pady=(0, 10))

        markdown_text = f"{self.content}"

        html_content = markdown.markdown(markdown_text)

        self.html_label = HTMLLabel(
            container,
            html=html_content,
            width=60
        )
        self.html_label.pack(fill="x", pady=(0, 12))

        self.text_area = tk.Text(container, height=8)
        self.text_area.pack(fill="both", expand=True, pady=(0, 12))

        button_frame = tk.Frame(container)
        button_frame.pack(fill="x")

        approve_button = tk.Button(
            button_frame,
            text="Approve",
            command=self.approve_action,
            width=12
        )
        approve_button.pack(side="left", padx=(0, 10))

        improve_button = tk.Button(
            button_frame,
            text="Improve",
            command=self.improve_action,
            width=12
        )
        improve_button.pack(side="left")

    def approve_action(self):
        messagebox.showinfo("Approved", f"Approved: \n\n{self.content[:250]}...")
        # self.root.quit()
        self.root.destroy()

    def improve_action(self):
        feedback = self.text_area.get("1.0", tk.END).strip()
        messagebox.showinfo("Improve", f"Content improving based on feedback: \n\n{feedback}")
        response = llm.invoke(f"""
            You are expert content generator.
            for the following topic, section and content, 
            improve the content based on user feedback
            TOPIC: "{self.topic}" {f'\nSECTION: "{self.section}"' if self.section else ''}
            CONTENT: "{self.content}"
            FEEDBACK: "{feedback}
        """.strip())
        self.content = response.content
        self.update_content_label()
        self.text_area.delete("1.0", tk.END)
        messagebox.showinfo("Improve", f"Content improved!")

    def update_content_label(self):
        html_content = markdown.markdown(self.content)
        self.html_label.set_html(html_content)

    def run(self):
        self.root.mainloop()
