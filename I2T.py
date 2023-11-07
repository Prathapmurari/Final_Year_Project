from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image
from pytesseract import image_to_string
from gtts import gTTS
from googletrans import Translator
from playsound import playsound
import os
import time
import pyttsx3
filename=""
root=Tk()
Options=["Tamil","Hindi","Telugu","Kanadam","Malayalam"]
var1=StringVar(root)
var1.set(Options[0])
w=OptionMenu(root,var1,*Options).place(x=400,y=20)

def clr():
    os.remove("Tamil.mp3")
    os.remove("Hindi.mp3")
    os.remove("Telugu.mp3")
    os.remove("Kanadam.mp3")
    os.remove("Malayalam.mp3")

def ok():
    dumm=var1.get()
    if(dumm=="Tamil"):
        translator = Translator()
        txt=texta.get("1.0","end-1c")
      
        translated = translator.translate(txt, src='english', dest='tamil')
        t=str(translated)
        texta_1.delete(1.0,END)
        texta_1.insert(END,t)
        txt1=texta_1.get("1.0","end-1c")
        obj=gTTS(text=txt1,lang='ta',slow=False)
        obj.save('Tamil.mp3')
        
    elif(dumm=="Hindi"):
        translator = Translator()
        txt=texta.get("1.0","end-1c")
        translated = translator.translate(txt, src='english', dest='hindi')
        t=str(translated)
        texta_1.delete(1.0,END)
        texta_1.insert(END,t)
        txt1=texta_1.get("1.0","end-1c")
        obj=gTTS(text=txt1,lang='hi',slow=False)
        obj.save('Hindi.mp3')
        time.sleep(10)
        
    elif(dumm=="Telugu"):
        translator = Translator()
        txt=texta.get("1.0","end-1c")
        translated = translator.translate(txt, src='english', dest='telugu')
        t=str(translated)
        texta_1.delete(1.0,END)
        texta_1.insert(END,t)
        txt1=texta_1.get("1.0","end-1c")
        obj=gTTS(text=txt1,lang='te',slow=False)
        obj.save('Telugu.mp3')
        time.sleep(10)
        
    elif(dumm=="Kanadam"):
        translator = Translator()
        txt=texta.get("1.0","end-1c")
        translated = translator.translate(txt, src='english', dest='kannada')
        t=str(translated)
        texta_1.delete(1.0,END)
        texta_1.insert(END,t)
        txt1=texta_1.get("1.0","end-1c")
        obj=gTTS(text=txt1,lang='kn',slow=False)
        obj.save('Kanadam.mp3')
        time.sleep(10)
        
    else:
        translator = Translator()
        txt=texta.get("1.0","end-1c")
        translated = translator.translate(txt, src='english', dest='malayalam')
        t=str(translated)
        texta_1.delete(1.0,END)
        texta_1.insert(END,t)
        txt1=texta_1.get("1.0","end-1c")
        obj=gTTS(text=txt1,lang='ml',slow=False)
        obj.save('Malayalam.mp3')
        time.sleep(10)
        
def play():
    dumm=var1.get()
    print(dumm)
    if(dumm=="Tamil"):
        tam()
    elif(dumm=="Hindi"):
        hin()
    elif(dumm=="Malayalam"):
        mal()
    elif(dumm=="Telugu"):
        Tel()
    else:
        kan()
        
        
du=Button(root,text="GO",bd=6,command=ok).place(x=500,y=20)
du1=Button(root,text="Clear",bd=6,command=clr).place(x=600,y=20)

def eng():
    playsound('english.mp3')


def tam():
    playsound('Tamil.mp3')
    
def hin():
    playsound('hindi.mp3')
def mal():
    playsound('Malayalam.mp3')
def Tel():
    playsound('Telugu.mp3')
    
def kan():
    playsound('Kanadam.mp3')
def Igm():
    filename=askopenfilename()
    img=filename
    image = Image.open(img, mode='r')
    txt=image_to_string(image,lang='eng')
    texta.insert(END,txt)
    print(filename)

root.geometry("1350x900")
root.configure(background="#307678")
root.title("Image to Text - Text to Voice")
root.resizable(width=False, height=False)
Lb1=Label(root,text="CHOOSE IMAGE TO CONVERT: ",bd=6)
Lb1.place(x=20,y=20)

Btn=Button(root,text="Choose File To Upload",bd=6,command=Igm)
Btn.place(x=250,y=20)
Lb2=Label(root,text="The Text: ",bd=6)
Lb2.place(x=0,y=70)
texta=Text(root,bd=7,width=45,height=20)
texta.place(x=150,y=70)
Lb3=Label(root,text="The Converted Text: ",bd=6)
Lb3.place(x=560,y=70)
texta_1=Text(root,bd=7,width=45,height=20)
texta_1.place(x=690,y=70)


#Btn1=Button(root,text="English",bd=6,command=eng).place(x=20,y=500)
Btn2=Button(root,text="PLAY",bd=6,command=play)
Btn2.place(x=660,y=20)
#Btn2.pack()
#Btn3=Button(root,text="Hindi",bd=6,command=hin).place(x=160,y=500)


def m():
    return root