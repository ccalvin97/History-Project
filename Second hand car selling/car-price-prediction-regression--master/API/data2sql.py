import sqlite3
import pandas as pd


data = pd.read_csv('testing_v1.csv')

conn = sqlite3.connect("data.db")
cur = conn.cursor()

data.to_sql("car_data", conn, if_exists="replace")

cur.close()
conn.close()