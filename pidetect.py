import tkinter as tk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
import json
import time
from picamera2 import Picamera2
import threading

model = tf.saved_model.load("/home/varshith/Downloads/plant_model_saved")
infer = model.signatures["serving_default"]

with open("/home/varshith/Downloads/class_names.json", "r") as f:
    class_names = json.load(f)

with open("/home/varshith/Downloads/plantinfo.json", "r") as f:
    plant_info = json.load(f)

THRESHOLD = 0.65

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 400)}))
picam2.start()
time.sleep(2)

latest_frame = None
is_detecting = False

root = tk.Tk()
root.title("Plant Toxicity Detector")
root.geometry("800x480")
root.configure(bg="#1a1a1a")
root.attributes("-fullscreen", True)

camera_label = tk.Label(root, bg="#1a1a1a")
camera_label.place(x=0, y=0, width=640, height=400)

result_frame = tk.Frame(root, bg="#1a1a1a")
result_frame.place(x=640, y=0, width=160, height=480)

result_title = tk.Label(result_frame, text="RESULT",
                         font=("Courier", 11, "bold"),
                         bg="#1a1a1a", fg="#00ff88")
result_title.pack(pady=(15, 5))

plant_label = tk.Label(result_frame, text="--",
                        font=("Courier", 10, "bold"),
                        bg="#1a1a1a", fg="white",
                        wraplength=150, justify="center")
plant_label.pack(pady=5)

score_label = tk.Label(result_frame, text="",
                        font=("Courier", 9),
                        bg="#1a1a1a", fg="#aaaaaa",
                        wraplength=150, justify="center")
score_label.pack(pady=2)

status_label = tk.Label(result_frame, text="",
                         font=("Courier", 10, "bold"),
                         bg="#1a1a1a", fg="white",
                         wraplength=150, justify="center")
status_label.pack(pady=5)

remedy_label = tk.Label(result_frame, text="",
                         font=("Courier", 8),
                         bg="#1a1a1a", fg="#cccccc",
                         wraplength=150, justify="center")
remedy_label.pack(pady=5)

bottom_frame = tk.Frame(root, bg="#111111")
bottom_frame.place(x=0, y=400, width=640, height=80)

status_bar = tk.Label(bottom_frame,
                       text="Point camera at a plant and press CAPTURE",
                       font=("Courier", 9), bg="#111111", fg="#888888")
status_bar.pack(pady=5)

capture_btn = tk.Button(bottom_frame, text="CAPTURE",
                         font=("Courier", 14, "bold"),
                         bg="#00ff88", fg="#000000",
                         activebackground="#00cc66",
                         relief="flat", cursor="hand2")
capture_btn.pack(pady=5, ipadx=20, ipady=5)

quit_btn = tk.Button(result_frame, text="QUIT",
                      font=("Courier", 9, "bold"),
                      bg="#ff4444", fg="white",
                      activebackground="#cc0000",
                      relief="flat", cursor="hand2",
                      command=root.destroy)
quit_btn.pack(side="bottom", pady=15, ipadx=10, ipady=5)

def run_detection():
    global is_detecting, latest_frame
    if latest_frame is None:
        return

    frame = latest_frame.copy()

    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    input_tensor = tf.constant(img)
    output = infer(input_tensor)

    output_key = list(output.keys())[0]
    prediction = output[output_key].numpy()

    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    def update_ui():
        global is_detecting
        if confidence < THRESHOLD:
            plant_label.config(text="Unknown", fg="#ffaa00")
            score_label.config(text="")
            status_label.config(text="Not Recognized", fg="#ffaa00")
            remedy_label.config(text="Try again with\na clearer image")
            status_bar.config(text="Unknown plant detected")
        else:
            plant_name = class_names[str(class_index)]
            info = plant_info.get(plant_name)
            plant_label.config(text=plant_name, fg="white")
            score_label.config(text=f"{round(confidence*100,2)}%")
            if info:
                if info["toxic"]:
                    status_label.config(text="POISONOUS", fg="#ff4444")
                else:
                    status_label.config(text="NON-POISONOUS", fg="#00ff88")
                remedy_label.config(text=info["remedy"])
            status_bar.config(text=f"Detected: {plant_name}")

        capture_btn.config(state="normal", text="CAPTURE",
                           bg="#00ff88")
        is_detecting = False

    root.after(0, update_ui)

def on_capture():
    global is_detecting
    if is_detecting:
        return
    is_detecting = True
    capture_btn.config(state="disabled", text="Detecting...",
                       bg="#888888")
    status_bar.config(text="Analysing plant...")
    threading.Thread(target=run_detection, daemon=True).start()

capture_btn.config(command=on_capture)

def update_frame():
    global latest_frame
    frame = picam2.capture_array()
    latest_frame = frame

    img = Image.fromarray(frame).resize((640, 400))
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)
    root.after(33, update_frame) 
update_frame()
root.mainloop()
picam2.close()