import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import numpy as np
import threading
import os
import config
from predict import load_model, predict_digit

class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition")
        self.root.geometry("900x650")
        self.root.configure(bg='#0d1117')
        
        self.model = None
        self.current_image = None
        self.current_image_path = None
        
        self.setup_gui()
        self.load_model_async()
    
    def setup_gui(self):
        main_container = tk.Frame(self.root, bg='#0d1117')
        main_container.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)
        
        left_panel = tk.Frame(main_container, bg='#161b22')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))
        
        upload_label = tk.Label(
            left_panel,
            text="UPLOAD",
            font=('Segoe UI', 11, 'bold'),
            bg='#161b22',
            fg='#58a6ff'
        )
        upload_label.pack(pady=(20, 12), padx=20, anchor=tk.W)
        
        self.drop_zone = tk.Label(
            left_panel,
            text="Drag and drop image here\nor click to browse",
            font=('Segoe UI', 12),
            bg='#0d1117',
            fg='#8b949e',
            relief=tk.FLAT,
            cursor='hand2'
        )
        self.drop_zone.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        self.drop_zone.drop_target_register(DND_FILES)
        self.drop_zone.dnd_bind('<<Drop>>', self.drop_file)
        self.drop_zone.bind('<Button-1>', lambda e: self.browse_image())
        self.drop_zone.bind('<Enter>', lambda e: self.drop_zone.config(bg='#1c2128'))
        self.drop_zone.bind('<Leave>', lambda e: self.drop_zone.config(bg='#0d1117'))
        
        browse_btn = tk.Button(
            left_panel,
            text="Browse Files",
            font=('Segoe UI', 10, 'bold'),
            bg='#238636',
            fg='#ffffff',
            activebackground='#2ea043',
            activeforeground='#ffffff',
            cursor='hand2',
            command=self.browse_image,
            relief=tk.FLAT,
            height=1,
            width=15
        )
        browse_btn.pack(pady=(0, 20))
        
        right_panel = tk.Frame(main_container, bg='#161b22')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(12, 0))
        
        result_label = tk.Label(
            right_panel,
            text="RESULTS",
            font=('Segoe UI', 11, 'bold'),
            bg='#161b22',
            fg='#58a6ff'
        )
        result_label.pack(pady=(20, 12), padx=20, anchor=tk.W)
        
        preview_container = tk.Frame(right_panel, bg='#0d1117', height=220)
        preview_container.pack(fill=tk.X, padx=20, pady=(0, 15))
        preview_container.pack_propagate(False)
        
        self.preview_label = tk.Label(
            preview_container,
            text="No image",
            bg='#0d1117',
            fg='#8b949e',
            font=('Segoe UI', 10)
        )
        self.preview_label.pack(expand=True)
        
        result_frame = tk.Frame(right_panel, bg='#0d1117')
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        self.result_text = tk.Text(
            result_frame,
            font=('Consolas', 10),
            bg='#0d1117',
            fg='#c9d1d9',
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=15,
            pady=15,
            borderwidth=0,
            insertbackground='#58a6ff',
            selectbackground='#1c2128'
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.END, "Load an image to begin")
        self.result_text.config(state=tk.DISABLED)
        
        btn_container = tk.Frame(right_panel, bg='#161b22')
        btn_container.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        self.predict_btn = tk.Button(
            btn_container,
            text="PREDICT DIGIT",
            font=('Segoe UI', 11, 'bold'),
            bg='#1f6feb',
            fg='#ffffff',
            activebackground='#388bfd',
            activeforeground='#ffffff',
            cursor='hand2',
            command=self.predict_image,
            relief=tk.FLAT,
            height=1,
            width=18,
            state=tk.DISABLED
        )
        self.predict_btn.pack()
        
        footer = tk.Frame(self.root, bg='#161b22', height=40)
        footer.pack(fill=tk.X)
        footer.pack_propagate(False)
        
        self.status_bar = tk.Label(
            footer,
            text="Loading model...",
            font=('Segoe UI', 9),
            bg='#161b22',
            fg='#8b949e',
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.BOTH, padx=25, pady=10)
    
    def load_model_async(self):
        def load():
            self.model = load_model()
            if self.model:
                self.root.after(0, lambda: self.status_bar.config(text="Model ready", fg='#3fb950'))
            else:
                self.root.after(0, lambda: self.status_bar.config(text="Model not found - Run: python train.py", fg='#f85149'))
                self.root.after(0, lambda: messagebox.showerror(
                    "Model Not Found",
                    "Please train the model first:\npython train.py"
                ))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def drop_file(self, event):
        file_path = event.data
        if file_path.startswith('{') and file_path.endswith('}'):
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
            img_display.thumbnail((380, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_display)
            
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo
            
            filename = os.path.basename(file_path)
            size = f"{img.size[0]}x{img.size[1]}"
            
            self.drop_zone.config(
                text=f"IMAGE LOADED\n\n{filename}\n{size} | {img.mode}",
                fg='#3fb950'
            )
            
            self.predict_btn.config(state=tk.NORMAL, bg='#1f6feb')
            self.status_bar.config(text=f"Image loaded: {filename}", fg='#58a6ff')
            
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Click predict to analyze")
            self.result_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            self.status_bar.config(text="Error loading image", fg='#f85149')
    
    def predict_image(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded")
            return
        
        if not self.current_image:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            self.status_bar.config(text="Processing...", fg='#d29922')
            self.predict_btn.config(state=tk.DISABLED, text="PROCESSING...")
            self.root.update()
            
            prediction, confidence = predict_digit(self.model, self.current_image)
            
            if confidence >= config.CONFIDENCE_THRESHOLD:
                self.display_results(prediction, confidence)
            else:
                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "LOW CONFIDENCE\n\n")
                self.result_text.insert(tk.END, f"Prediction: {prediction}\n")
                self.result_text.insert(tk.END, f"Confidence: {confidence*100:.1f}%\n\n")
                self.result_text.insert(tk.END, "Try a clearer image")
                self.result_text.config(state=tk.DISABLED)
                self.status_bar.config(text="Low confidence", fg='#d29922')
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
            self.status_bar.config(text="Prediction failed", fg='#f85149')
        
        finally:
            self.predict_btn.config(state=tk.NORMAL, text="PREDICT DIGIT")
    
    def display_results(self, prediction, confidence):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        self.result_text.insert(tk.END, "PREDICTION\n", 'header')
        self.result_text.insert(tk.END, "_" * 40 + "\n\n", 'line')
        
        self.result_text.insert(tk.END, "Detected Digit\n", 'label')
        self.result_text.insert(tk.END, f"{prediction}\n\n", 'digit')
        
        word = config.NUMBER_TO_WORD[prediction]
        self.result_text.insert(tk.END, "Word Form\n", 'label')
        self.result_text.insert(tk.END, f"{word.upper()}\n\n", 'word')
        
        self.result_text.insert(tk.END, "Confidence\n", 'label')
        self.result_text.insert(tk.END, f"{confidence*100:.1f}%\n", 'confidence')
        
        self.result_text.tag_config('header', font=('Segoe UI', 12, 'bold'), foreground='#58a6ff')
        self.result_text.tag_config('line', foreground='#30363d')
        self.result_text.tag_config('label', font=('Segoe UI', 9), foreground='#8b949e')
        self.result_text.tag_config('digit', font=('Segoe UI', 42, 'bold'), foreground='#3fb950')
        self.result_text.tag_config('word', font=('Segoe UI', 20, 'bold'), foreground='#1f6feb')
        self.result_text.tag_config('confidence', font=('Segoe UI', 16, 'bold'), foreground='#58a6ff')
        
        self.result_text.config(state=tk.DISABLED)
        
        self.status_bar.config(text=f"Prediction: {prediction} ({word}) - {confidence*100:.1f}%", fg='#3fb950')

def main():
    try:
        root = TkinterDnD.Tk()
        app = DigitRecognizerGUI(root)
        root.mainloop()
    except ImportError:
        print("Error: tkinterdnd2 not installed")
        print("Run: pip install tkinterdnd2")

if __name__ == "__main__":
    main()