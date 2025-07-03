import tkinter as tk
from PIL import ImageTk, Image
from ChatPasIntegligentQuiResume import *
from ChatPasIntegligentQuiClassV2 import *
from ChatPasIntegligentQuiSynth import *
from Chat_bot_KeyWords_27062025 import *
# Créer une fenêtre
window = tk.Tk()
from scipy import triu
# Titre de la fenêtre
window.title("Menu de l'application")
window.geometry("700x400")
window.minsize(400, 200)

# Icon de la fenêtre (nécessite un format .ico)
ico_path = "Unintelligent_cat.ico"
window.iconbitmap(ico_path)

#Création des frames
Frame_h = tk.Frame(window,bg="#2C3E50",bd=2,width=700, height=100)
Frame_b = tk.Frame(window,bg="#2C3E50",bd=1,width=700, height=100)

Frame_h.pack(side="top",fill="both", expand=True)
Frame_b.pack(side="bottom",fill="both", expand=True)

#Création d'une image
image_path = "Unintelligent_cat.png"
img = ImageTk.PhotoImage(Image.open(image_path).resize((400, 300)))


#Ajout image et texte
label = tk.Label(Frame_h, text="The Unintelligent Cat", font=("Arial", 40),bg="#2C3E50", fg="white")
label.pack(side="top", pady=50)

label_image = tk.Label(Frame_h, image=img, bg="#2C3E50")
label_image.pack(side="top")

sub_title = tk.Label(Frame_h, text="What do you want to do?", font=("Arial", 20),bg="#2C3E50", fg="white")
sub_title.pack(side="bottom", pady=10)

##Boutons du bas
#Configure les colonnes pour qu'elles s'étendent également
Frame_b.grid_columnconfigure(0, weight=1)
Frame_b.grid_columnconfigure(5, weight=1)

#Créeation des boutons
button_w = tk.Button(Frame_b, text="Wikipedia", font=("#2C3E50", 20), bg="white", fg="#2C3E50", command=main_resum)
button_w.grid(row=0, column=1, padx=40)

button_s = tk.Button(Frame_b, text="Synthesize", font=("#2C3E50", 20), bg="white", fg="#2C3E50", command=main_synth)
button_s.grid(row=0, column=2, padx=40)

button_c = tk.Button(Frame_b, text="Classify", font=("#2C3E50", 20), bg="white", fg="#2C3E50", command=main_class)
button_c.grid(row=0, column=3, padx=40)

# Nouveau bouton ajouté sans fonction
button_new = tk.Button(Frame_b, text="Keyword extract", font=("#2C3E50", 20), bg="white", fg="#2C3E50"
                       ,command=main_keykey)
button_new.grid(row=0, column=4, padx=40)

# Boucle principale d'affichage de la fenêtre
window.mainloop()
