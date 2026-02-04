import tkinter as tk
from PIL import ImageTk, Image
import io
from linear_workflow import linear_workflow_diagram
from conditional_workflow import conditional_workflow_diagram

root = tk.Tk()
root.title("Lang Graph")

# Create a frame to hold the images in a column layout
frame = tk.Frame(root)
frame.pack(pady=10)

# Convert IPython display Image to PIL Image
# The diagram is a PNG image in bytes format
image_bytes1 = linear_workflow_diagram.data
pil_image1 = Image.open(io.BytesIO(image_bytes1))

# Convert PIL Image to Tkinter PhotoImage
tk_img1 = ImageTk.PhotoImage(pil_image1)
label1 = tk.Label(frame, image=tk_img1)
label1.grid(row=0, column=0, padx=10, pady=5)

# Convert IPython display Image to PIL Image
# The diagram is a PNG image in bytes format
image_bytes2 = conditional_workflow_diagram.data
pil_image2 = Image.open(io.BytesIO(image_bytes2))

# Convert PIL Image to Tkinter PhotoImage
tk_img2 = ImageTk.PhotoImage(pil_image2)
label2 = tk.Label(frame, image=tk_img2)
label2.grid(row=0, column=1, padx=10, pady=5)

root.mainloop()
