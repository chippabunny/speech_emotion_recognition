import tkinter as tk
from tkinter import *
from tensorflow.keras.models import model_from_json

import numpy as np

import pandas as pd
import numpy as np

import librosa
import librosa.display

import speech_recognition as sr
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from tensorflow.keras.models import model_from_json



sample_rate=22050







def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def load_model(json_file,weights_file):
    with open(json_file,'r') as file:
        loaded_model=file.read()
        model=model_from_json(loaded_model)
    model.load_weights(weights_file)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model








def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally
    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result


















top=tk.Tk()
top.geometry('800x600')
top.title("speech emotion detection")
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD',font=('arial',15,'bold'))


def record():
    # Initialize recognizer
    global label 
    label.config(text="speak now")
    top.update_idletasks()
    r = sr.Recognizer()

    # Use microphone as source
    with sr.Microphone() as source:
        print("Speak now...")
        audio = r.listen(source)

    # Save audio to file
    with open("audio.wav", "wb") as f:
        f.write(audio.get_wav_data())




def find_emotion(to_pred):
    pred=encoder.inverse_transform(to_pred)
    print(pred)
    predicted=[]
    for i in pred:
        predicted.append(i[0])


    d={}
    for i in predicted:
        if i in d:
            d[i]+=1
        else:
            d[i]=1
    result=""
    for i in d:
        if d[i]==3 or d[i]==2:
            result=i
            break

    print("predicted emotion is "+result)
    return result



def record_detect():
    global label
    
    label.config(text="   ")
    top.update_idletasks()
    record()
    label.config(text="   detecting  ")
    top.update_idletasks()
    features=get_features('audioav')
    pred=model.predict(features)
    emotion=find_emotion(pred)
    label.config(text=emotion)



model=load_model('model_architecture.json','model_weights.h5')

scalar=StandardScaler()
encoder=OneHotEncoder()


y=pd.read_csv('features.csv')
y=y.labels
y= encoder.fit_transform(np.array(y).reshape(-1,1)).toarray()



upload=Button(top,text="record and detect",command=record_detect,padx=10,pady=5)
upload.configure(background='#364156',foreground='white',font=('arial',30,'bold'))
upload.pack(side='bottom',pady=50)
label.pack(side='bottom',expand=True)
heading=Label(top,text='speech emotion detection',padx=30,font=('arial',30,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()

top.mainloop()



