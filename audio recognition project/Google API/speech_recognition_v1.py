#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 00:26:37 2018

@author: calvin
"""



import speech_recognition as sr
import speech_recognition
import time 
import os
import pyaudio
import wave 


def Voice_To_Text():
    
    r=sr.Recognizer() 
    with sr.Microphone() as source: 
     ## https://blog.gtwang.org/programming/python-with-context-manager-tutorial/
        print("please talk:")                               # print
        r.adjust_for_ambient_noise(source)     # adjust the environmentalnoise
        audio = r.listen(source)
    try:
        Text = r.recognize_google(audio, language="zh-TW")     
    except r.UnknowValueError:
        Text = "cannot translate"
    except sr.RequestError as e:
        Text = "cannot translate".format(e)
              # if it cannot translate, print this

    return Text



Text = Voice_To_Text()
print(Text)