from tkinter import *
from tkinter import ttk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
import json
import numpy as np
import threading
import requests

model = load_model("NTv1.h5")

with open("tokenizer.json","r", encoding="utf-8") as f:
    tokenizer_json = f.read()
        
tokenizer = tokenizer_from_json(json.loads(tokenizer_json))

max_lenghth = len(tokenizer.word_index)

root = Tk()

root.geometry("700x700")
root.title("AI Chatbot")
# Dark theme colors
background_color = "#2C3E50"  
text_color = "#D3D3D3"        
button_color = "#FFA07A"      
enter_bg_color = "#34495E"    
enter_text_color = "#FFFFFF"  

input_text = StringVar()

root.configure(background=background_color)


def generate_prediction():
    print("inside gen predict")
    tokenizer.word_index
    for i in range(2):

        token_text = tokenizer.texts_to_sequences([input_text.get()])[0]
        token_text = pad_sequences([token_text], maxlen = max_lenghth, padding = 'pre')
        prob = model.predict(token_text)
        pos = np.argmax(prob)
        text = input_text.get()
        for word, index in tokenizer.word_index.items():
            if index == pos:
                text = text + " " + word
                input_text.set(text)
                print(text)
        
        text_box.delete("1.0", END)
        text_box.insert(END, text)

question_var = StringVar()

scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)

text_box = Text(root, height=10, width=0, font=('Microsoft YaHei Light', 14, "bold"),yscrollcommand=scrollbar.set, bg=enter_bg_color, fg=text_color, insertbackground=text_color)
text_box.pack(pady=10, fill=BOTH, expand=True)

question_label = Label(root, text="Enter your question:", font=('Microsoft YaHei Light', 14, "bold"),bg=background_color, fg=text_color)
question_label.pack(pady=5)

question_entry = Entry( root, textvariable=input_text, font=('Microsoft YaHei Light', 14, "bold"),bg=enter_bg_color, fg=enter_text_color, insertbackground=enter_text_color)
question_entry.pack(pady=5)

ask_button = Button(root, text="Generate Response",command = generate_prediction,font=("Microsoft YaHei Light", 14, "bold"), bg=button_color, fg=text_color)
ask_button.pack(pady=10)

root.mainloop()
