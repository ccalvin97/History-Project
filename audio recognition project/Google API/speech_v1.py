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
     ## 介紹一下 with XXX as XX 這個指令
     ## XXX 是一個函數或動作 然後我們把他 的output 放在 XX 裡
     ## with 是在設定一個範圍 讓本來的 source 不會一直進行
     ## 簡單的應用，可以參考
     ## https://blog.gtwang.org/programming/python-with-context-manager-tutorial/
        print("please speak something:")                               # print 一個提示 提醒你可以講話了
        r.adjust_for_ambient_noise(source)     # 函數調整麥克風的噪音:
        audio = r.listen(source)
     ## with 的功能結束 source 會不見 
     ## 接下來我們只會用到 audio 的結果
    try:
        Text = r.recognize_google(audio, language="en-US")   
#        Text = r.recognize_google(audio, language="zh-TW")   
        	
              ##將剛說的話轉成  zh-TW 繁體中文 的 字串
              ## recognize_google 指得是使用 google 的api 
              ## 也就是用google 網站看到的語音辨識啦~~
              ## 雖然有其他選擇  但人家是大公司哩 當然優先用他的囉dd
    except sr.UnknowValueError:
        Text = "cannot be translated"
    except sr.RequestError as e:
        Text = "cannot be translated{0}".format(e)
              # 兩個 except 是當語音辨識不出來的時候 防呆用的 

    return Text
### fun定義結束

##讓我們實際利用看看吧~
Text = Voice_To_Text()
print(Text)