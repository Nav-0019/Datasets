import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



os.chdir("B:/Github/Datasets/Uber_Dataset")

df = pd.read_csv("Uber_Dataset.csv")

df.columns = df.columns.str.strip().str.replace(" ", "_")

print(df.head(), df.info(), df.isna().sum())

df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour

print(df['Booking_Status'].unique())

Encoder_status = pd.get_dummies(df['Booking_Status'], prefix='Booking_Status', drop_first=True)
df = pd.concat([df, Encoder_status], axis=1)
df.drop(columns = ['Booking_Status'], inplace=True)

print(df.info())