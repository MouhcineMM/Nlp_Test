import tkinter as tk
from tkinter import ttk, scrolledtext
import wikipedia
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Télécharger les ressources NLTK nécessaires (exécuter une seule fois)
# nltk.download('punkt')
#Pour import BART
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Unintelligent wiki")
        self.root.geometry("700x500")
        self.root.configure(bg="#2C3E50")  # Fond bleu foncé
        self.root.minsize(400, 200)

        # Style personnalisé
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12), padding=5)
        self.style.configure("TLabel", font=("Helvetica", 12), background="#2C3E50", foreground="white")
        self.style.configure("TEntry", font=("Helvetica", 12))

        # Titre
        self.title_label = ttk.Label(root, text="Le chat qui ne GPT pas bien", font=("Helvetica", 16, "bold"), foreground="#ECF0F1")
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

        self.input_label = ttk.Label(self.input_frame, text="Your message :")
        self.input_label.pack(side="left", padx=5)

        self.input_field = ttk.Entry(self.input_frame, width=50)
        self.input_field.pack(side="left", padx=5, fill="x", expand=True)
        self.input_field.bind("<Return>", self.process_input)

        self.send_button = ttk.Button(
            self.input_frame, 
            text="Send", 
            command=self.process_input,
            style="Custom.TButton"
        )
        self.style.configure("Custom.TButton", background="#3498DB", foreground="#2C3E50")
        self.send_button.pack(side="right", padx=5)

    def process_input(self, event=None):
        user_input = self.input_field.get().strip()
        if user_input:
            # Afficher le message utilisateur avec style
            self.chat_display.insert(tk.END, "You : ", ("user",))
            self.chat_display.insert(tk.END, user_input + "\n")
            
            # Obtenir et afficher la réponse
            response = self.get_wikipedia_summary(user_input)
            self.chat_display.insert(tk.END, "Bot : ", ("bot",))
            self.chat_display.insert(tk.END, response + "\n\n")
            
            self.input_field.delete(0, tk.END)
            self.chat_display.see(tk.END)

    def get_wikipedia_summary(self, query):
        try:
            wikipedia.set_lang("en")
            full_summary = wikipedia.summary(query)
            short_summary = self.condense_summary(full_summary,num_sentences=3)
            return short_summary
            
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Your request is ambiguous. Please specify. Possible options : {e.options[:3]}"
        except wikipedia.exceptions.PageError:
            return "Sorry, I did not find any information on this topic on Wikipedia."
        except Exception as e:
            return f"An error has occurred : {str(e)}"

    def condense_summary(self, text, num_sentences=3):
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=200,
            min_length=80,
            num_beams=4,
            early_stopping=True,
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        sentences = sent_tokenize(summary)
        condensed = ' '.join(sentences[:num_sentences])
        
        return condensed

    def configure_tags(self):
        # Configurer les styles pour les messages
        self.chat_display.tag_configure("user", foreground="#3498DB", font=("Helvetica", 11, "bold"))
        self.chat_display.tag_configure("bot", foreground="#E74C3C", font=("Helvetica", 11, "bold"))

def main_resum():
    root = tk.Tk()
    app = ChatbotGUI(root)
    app.configure_tags()  # Configurer les tags après l'initialisation
    root.mainloop()

if __name__ == "__main__":
    main_resum()