import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import numpy as np
import threading
import os
import config
from predict import load_model, predict_digit

# Modern, clean, cool-toned redesign (dark + minimal)
# No borders, no icons, no emojis, no top title bar.
# Smooth aqua/blue accents, flat minimal design.

class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("")  # No title text
        self.root.geometry("800x600")
        self.root.configure(bg="#1e1e24")  # Dark modern background

        self.model = None
        self.current_image = None
        self.current_image_path = None

        self.setup_gui()
        self.load_model_async()

    def setup_gui(self):
        main_container = tk.Frame(self.root, bg="#1e1e24")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # LEFT PANEL
        left_panel = tk.Frame(main_container, bg="#1e1e24")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Drop Zone (no border)
        self.drop_zone = tk.Label(
            left_panel,
            text="Drag & Drop Image Here\n\nor\n\nClick to Browse",
            font=("Inter", 14),
            bg="#2a2a33",  # Subtle dark card
            fg="#cfcfd6",
            borderwidth=0,
            cursor="hand2"
        )
        self.drop_zone.pack(fill=tk.BOTH, expand=True, pady=10)

        self.drop_zone.drop_target_register(DND_FILES)
        self.drop_zone.dnd_bind("<<Drop>>", self.drop_file)
        self.drop_zone.bind("<Button-1>", lambda e: self.browse_image())

        # Browse Button
        browse_btn = tk.Button(
            left_panel,
            text="Browse Image",
            font=("Inter", 13, "bold"),
            bg="#3b82f6",      # Modern blue
            fg="white",
            activebackground="#2563eb",
            activeforeground="white",
            cursor="hand2",
            relief="flat",
            height=2
        )
        browse_btn.config(command=self.browse_image)
        browse_btn.pack(fill=tk.X, pady=(10, 0))

        # RIGHT PANEL
        right_panel = tk.Frame(main_container, bg="#1e1e24")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Preview container
        self.preview_label = tk.Label(
            right_panel,
            text="No image loaded",
            bg="#2a2a33",
            fg="#b0b0b8",
            font=("Inter", 11),
            borderwidth=0
        )
        self.preview_label.pack(fill=tk.X, pady=10)

        # Result text box (no border)
        result_frame = tk.Frame(right_panel, bg="#1e1e24")
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(
            result_frame,
            font=("Inter", 12),
            bg="#2a2a33",
            fg="#e6e6eb",
            wrap=tk.WORD,
            borderwidth=0,
            relief="flat",
            padx=15,
            pady=15
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.END, "Load an image to see predictions...")
        self.result_text.config(state=tk.DISABLED)

        # Predict Button — redesigned
        self.predict_btn = tk.Button(
            right_panel,
            text="Predict Digit",
            font=("Inter", 14, "bold"),
            bg="#10b981",      # Modern green
            fg="white",
            activebackground="#059669",
            activeforeground="white",
            cursor="hand2",
            relief="flat",
            state=tk.DISABLED,
            height=2
        )
        self.predict_btn.config(command=self.predict_image)
        self.predict_btn.pack(fill=tk.X, pady=(10, 0))

        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Loading model...",
            font=("Inter", 10),
            bg="#1e1e24",
            fg="#8a8a92",
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, padx=20, pady=(0, 10))

    def load_model_async(self):
        def load():
            self.model = load_model()
            if self.model:
                self.root.after(0, lambda: self.status_bar.config(text="Model loaded. Ready."))
            else:
                self.root.after(0, lambda: self.status_bar.config(text="Model not found."))
                self.root.after(0, lambda: messagebox.showerror(
                    "Model Not Found",
                    "Please train the model first: python train.py"
                ))

        thread = threading.Thread(target=load, daemon=True)
        thread.start()

    def drop_file(self, event):
        file_path = event.data
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]
        self.load_image(file_path)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        try:
            self.current_image_path = file_path
            img = Image.open(file_path)
            self.current_image = img.copy()

            img_display = img.copy()
            img_display.thumbnail((320, 320), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_display)

            self.preview_label.config(image=photo, text="", bg="#2a2a33")
            self.preview_label.image = photo

            self.drop_zone.config(
                text=f"Image Loaded\n{os.path.basename(file_path)}\nSize: {img.size}\nMode: {img.mode}",
                fg="#4ade80",   # Soft lime
                bg="#2a2a33"
            )

            self.predict_btn.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Loaded: {os.path.basename(file_path)}")

            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Click Predict Digit to analyze...")
            self.result_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            self.status_bar.config(text="Error loading image")

    def predict_image(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded!")
            return
        if not self.current_image:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        try:
            self.status_bar.config(text="Analyzing...")
            self.predict_btn.config(state=tk.DISABLED, text="Processing...")
            self.root.update()

            prediction, confidence = predict_digit(self.model, self.current_image)

            if confidence >= config.CONFIDENCE_THRESHOLD:
                self.display_results(prediction, confidence)
            else:
                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Low confidence prediction\n\n")
                self.result_text.insert(tk.END, f"Prediction: {prediction}\n")
                self.result_text.insert(tk.END, f"Confidence: {confidence*100:.1f}%\n\n")
                self.result_text.insert(tk.END, "Try a clearer image.")
                self.result_text.config(state=tk.DISABLED)
                self.status_bar.config(text="Low confidence")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
            self.status_bar.config(text="Prediction failed")
        finally:
            self.predict_btn.config(state=tk.NORMAL, text="Predict Digit")

    def display_results(self, prediction, confidence):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        self.result_text.insert(tk.END, f"Prediction: {prediction}\n", "bold")
        word = config.NUMBER_TO_WORD[prediction]
        self.result_text.insert(tk.END, f"In Words: {word}\n", "bold")
        self.result_text.insert(tk.END, f"Confidence: {confidence*100:.1f}%\n", "bold")

        self.result_text.tag_config("bold", font=("Inter", 14, "bold"))

        self.result_text.config(state=tk.DISABLED)
        self.status_bar.config(text=f"Prediction: {prediction} ({word})")


def main():
    try:
        root = TkinterDnD.Tk()
        app = DigitRecognizerGUI(root)
        root.mainloop()
    except ImportError:
        print("Error: tkinterdnd2 not installed!")
        print("Run: pip install tkinterdnd2")   
if __name__ == "__main__":
    main()
