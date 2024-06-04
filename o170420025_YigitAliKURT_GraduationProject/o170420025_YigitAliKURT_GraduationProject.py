import os
import pickle
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

class ImageCaptioningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Second School")
        self.root.geometry("400x600")
        
        # Set window icon
        self.root.iconphoto(True, tk.PhotoImage(file="Resources/logo.png"))
        
        self.create_widgets()
        
        self.inception_model = self.load_inception_model()
        self.model = load_model('Resources/image_captioning_model.keras')
        self.tokenizer = self.load_tokenizer('Resources/tokenizer.pkl')

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=400, height=600)
        self.canvas.pack(fill="both", expand=True)
        
        self.bg_image = Image.open("Resources/main_bg.jpg")
        self.bg_image = self.bg_image.resize((400, 600), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")
        
        self.photo_button_img = Image.open("Resources/photo_bg.jpg")
        self.photo_button_img = self.photo_button_img.resize((150, 150), Image.LANCZOS)
        self.photo_button_photo = ImageTk.PhotoImage(self.photo_button_img)
        
        self.caption_button_img = Image.open("Resources/pencil_bg.jpg")
        self.caption_button_img = self.caption_button_img.resize((150, 150), Image.LANCZOS)
        self.caption_button_photo = ImageTk.PhotoImage(self.caption_button_img)
        
        self.okay_button_img = Image.open("Resources/okay_bg.png")
        self.okay_button_img = self.okay_button_img.resize((160, 160), Image.LANCZOS)
        self.okay_button_photo = ImageTk.PhotoImage(self.okay_button_img)
        
        self.image_button = tk.Button(self.root, image=self.photo_button_photo, command=self.browse_image, bd=0)
        self.image_button_window = self.canvas.create_window(200, 200, window=self.image_button)
        
        self.generate_button = tk.Button(self.root, image=self.caption_button_photo, command=self.generate_caption, bd=0)
        self.generate_button_window = self.canvas.create_window(200, 380, window=self.generate_button)

    def load_tokenizer(self, tokenizer_path):
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
    
    def load_inception_model(self):
        inception_model = InceptionV3(include_top=False, weights='imagenet')
        new_input = inception_model.input
        hidden_layer = inception_model.layers[-1].output
        pooling_layer = GlobalAveragePooling2D()(hidden_layer)
        return Model(inputs=new_input, outputs=pooling_layer)
    
    def browse_image(self):
        image_file = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        self.image_path = image_file

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(299, 299))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def extract_image_features(self, image_path, model):
        img = self.preprocess_image(image_path)
        feature = model.predict(img, verbose=0)
        return feature

    def generate_sentence_from_features(self, model, tokenizer, img_feature, max_length):
        in_text = ''
        words = []
        for _ in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([img_feature, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = tokenizer.index_word.get(yhat)
            if word is None:
                break
            if len(in_text) == 0:
                word = word.capitalize()
            in_text += word + ' '
            if word == '<end>':
                break
            words.append(word)
            if len(' '.join(words)) > 29:
                in_text += '\n'
                words = []
        return in_text.strip()
    
    def generate_caption(self):
        if not hasattr(self, 'image_path') or not self.image_path:
            self.show_caption_message("No Image Selected", "")
            return
        
        feature = self.extract_image_features(self.image_path, self.inception_model)
        caption = self.generate_sentence_from_features(self.model, self.tokenizer, feature, 34)
        
        self.show_caption_message(os.path.basename(self.image_path), caption)

    def show_caption_message(self, image_name, caption):
        self.caption_window = tk.Toplevel(self.root)
        self.caption_window.title("Second Teacher")
        self.caption_window.geometry("650x450")

        bg_image = Image.open("Resources/output_bg.png")
        bg_image = bg_image.resize((650, 450), Image.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_image)

        canvas = tk.Canvas(self.caption_window, width=650, height=450)
        canvas.pack(fill="both", expand=True)
        canvas.create_image(0, 0, image=bg_photo, anchor="nw")

        caption_text = tk.Text(self.caption_window, height=5, width=30, font=("Segoe Print", 16))
        caption_text_window = canvas.create_window(550, 50, window=caption_text, anchor="ne")
        caption_text.insert(tk.END, f'{caption}\n')

        caption_window_okay_button = tk.Button(self.caption_window, image=self.okay_button_photo, command=self.close_caption_window, bd=0)
        caption_window_okay_button_window = canvas.create_window(550, 350, window=caption_window_okay_button)

        self.caption_window.bg_photo = bg_photo

    def close_caption_window(self):
        self.caption_window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptioningApp(root)
    root.mainloop()
