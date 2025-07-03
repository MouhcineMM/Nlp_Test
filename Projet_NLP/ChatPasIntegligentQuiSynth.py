import tkinter as tk
from tkinter import ttk, scrolledtext

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

class ChatbotGUI_Synth:
    def __init__(self, root):
        self.root = root
        self.root.title("Unintelligent synthesize")
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
            response = self.get_synth(user_input)
            self.chat_display.insert(tk.END, "Bot ML TextRank: ", ("bot",))
            self.chat_display.insert(tk.END, str(response) + "\n\n")
            
            self.input_field.delete(0, tk.END)
            self.chat_display.see(tk.END)

    def get_synth(self, query):
        nb_query = query.count(".")
        if isinstance(query, str) or query == "":
            if nb_query<= 5:
                return self.synthesize(query, 1)
            else: max_sentences = round(nb_query*0.2)
            return self.synthesize(query, max_sentences)
        else:
            return "Error: the request must be a string."
        
    def synthesize(self, query, max_sentences):
        try:
            # Créer un parser et un tokenizer
            parser = PlaintextParser.from_string(query, Tokenizer("english"))
            # Créer un résumeur
            summarizer = TextRankSummarizer()
            # Résumer le texte
            summary = summarizer(parser.document, max_sentences)  # Résumer en 2 phrases
            return " ".join(str(sentence) for sentence in summary)
        except Exception as e:
            return "Error during synthesis: " + str(e)

    def configure_tags(self):
        # Configurer les styles pour les messages
        self.chat_display.tag_configure("user", foreground="#3498DB", font=("Helvetica", 11, "bold"))
        self.chat_display.tag_configure("bot", foreground="#E74C3C", font=("Helvetica", 11, "bold"))

def main_synth():
    root = tk.Tk()
    app = ChatbotGUI_Synth(root)
    app.configure_tags()  # Configurer les tags après l'initialisation
    root.mainloop()

if __name__ == "__main__":
    main_synth()