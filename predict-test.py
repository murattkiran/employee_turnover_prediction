#!/usr/bin/env python
# coding: utf-8


import requests


url = 'http://localhost:9696/predict'

employee = {                                  
    "education": "Masters",        
    "joiningyear": 2017,           
    "city": "Bangalore",           
    "paymenttier": 3,              
    "age": 34,                     
    "gender": "Male",              
    "everbenched": "No",           
    "experienceincurrentdomain": 0 
}                                  


response = requests.post(url, json=employee).json()
print(response)