import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string

class KeywordExtractor:
    """Extracteur de mots-clés simplifié"""
    
    def __init__(self):
        # Charger directement le modèle
        try:
            self.model = load_model("./model/Classification/Précis/keyword_extraction_model_cleaned.keras", compile=False)
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            with open("./model/Classification/Précis/keyword_tokenizer_cleaned.pkl", 'rb') as f:
                self.tokenizer = pickle.load(f)
        except:
            try:
                self.model = load_model("./model/Classification/Précis/keyword_extraction_model_20250626_201852.keras", compile=False)
                self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                
                with open("./model/Classification/Précis/keyword_tokenizer_20250626_201852.pkl", 'rb') as f:
                    self.tokenizer = pickle.load(f)
            except:
                self.model = None
                self.tokenizer = None
        
        # Mots vides basiques
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 
            'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
            'has', 'had', 'having', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'one', 'two', 'three'
        }
    
    def extract_keywords(self, text, threshold=0.4, maxlen=128):
        """Extraction des mots-clés"""
        if not self.model or not text.strip():
            return []
        
        try:
            # Nettoyage et tokenisation
            text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\'\"]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            tokens = re.findall(r'\b\w+\b', text.lower())
            tokens = [token for token in tokens if len(token) >= 2]
            
            if not tokens:
                return []
            
            # Prédiction
            sequence = self.tokenizer.texts_to_sequences([" ".join(tokens)])[0]
            if not sequence:
                return []
            
            X = pad_sequences([sequence], maxlen=maxlen, padding='post')
            predictions = self.model.predict(X, verbose=0)
            y_pred = (predictions > threshold).astype(int)
            
            # Extraction des mots-clés
            keywords = []
            for i, (token, pred) in enumerate(zip(tokens, y_pred[0])):
                if i < len(tokens) and i < len(y_pred[0]) and pred[0] == 1:
                    keywords.append(token)
            
            # Post-traitement
            cleaned = []
            for keyword in keywords:
                keyword = keyword.lower().strip()
                if (len(keyword) >= 2 and 
                    keyword not in self.stop_words and
                    not keyword.isdigit() and
                    not all(c in string.punctuation for c in keyword)):
                    cleaned.append(keyword)
            
            return list(dict.fromkeys(cleaned))[:10]
            
        except Exception as e:
            return []

class ChatbotGUI_Keywords:
    def __init__(self, root):
        self.root = root
        self.root.title("Akinator for KeyWords")
        self.root.geometry("700x500")
        self.root.configure(bg="#2C3E50")
        self.root.minsize(400, 200)
        
        # Initialiser l'extracteur
        self.extractor = KeywordExtractor()

        # Style personnalisé
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12), padding=5)
        self.style.configure("TLabel", font=("Helvetica", 12), background="#2C3E50", foreground="white")
        self.style.configure("TEntry", font=("Helvetica", 12))

        # Titre
        self.title_label = ttk.Label(root, text="🔍 Akinator for KeyWords", font=("Helvetica", 16, "bold"), foreground="#ECF0F1")
        self.title_label.pack(pady=10)

        # Zone de chat avec frame
        self.chat_frame = tk.Frame(root, bg="#34495E", bd=2, relief="flat")
        self.chat_frame.pack(padx=20, pady=10, fill="both", expand=True)

        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=20, 
            bg="#ECF0F1", 
            fg="#2C3E50", 
            font=("Helvetica", 11),
            bd=0,
            highlightthickness=0
        )
        self.chat_display.pack(padx=10, pady=10, fill="both", expand=True)

        # Zone d'entrée
        self.input_frame = tk.Frame(root, bg="#2C3E50")
        self.input_frame.pack(padx=20, pady=10, fill="x")

        self.input_label = ttk.Label(self.input_frame, text="Your text :")
        self.input_label.pack(side="left", padx=5)

        self.input_field = ttk.Entry(self.input_frame, width=50)
        self.input_field.pack(side="left", padx=5, fill="x", expand=True)
        self.input_field.bind("<Return>", self.process_input)

        self.send_button = ttk.Button(
            self.input_frame, 
            text="Extract", 
            command=self.process_input,
            style="Custom.TButton"
        )
        self.style.configure("Custom.TButton", background="#27AE60", foreground="white")
        self.send_button.pack(side="right", padx=5)

    def process_input(self, event=None):
        user_input = self.input_field.get().strip()
        if user_input:
            # Afficher le message utilisateur avec style
            self.chat_display.insert(tk.END, "You : ", ("user",))
            self.chat_display.insert(tk.END, user_input + "\n")
            
            # Obtenir et afficher la réponse
            response = self.get_keywords(user_input)
            self.chat_display.insert(tk.END, "Bot ML Keywords: ", ("bot",))
            self.chat_display.insert(tk.END, str(response) + "\n\n")
            
            self.input_field.delete(0, tk.END)
            self.chat_display.see(tk.END)

    def get_keywords(self, query):
        if not isinstance(query, str) or query == "":
            return "Error: the request must be a string."
        
        try:
            keywords = self.extractor.extract_keywords(query)
            
            if keywords:
                return "Keywords: " + " • ".join(keywords)
            else:
                return "No significant keywords found."
                
        except Exception as e:
            return "Error during extraction: " + str(e)

    def configure_tags(self):
        # Configurer les styles pour les messages
        self.chat_display.tag_configure("user", foreground="#3498DB", font=("Helvetica", 11, "bold"))
        self.chat_display.tag_configure("bot", foreground="#27AE60", font=("Helvetica", 11, "bold"))

def main_keykey():
    root = tk.Tk()
    app = ChatbotGUI_Keywords(root)
    app.configure_tags()
    root.mainloop()

if __name__ == "__main__":
    main_keykey()