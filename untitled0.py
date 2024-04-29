from flask import Flask
from pymongo import MongoClient


# MongoDB connection string
# Replace <password> with your actual password
client = MongoClient("mongodb+srv://user1:12345@atlascluster.nq1qhdg.mongodb.net/?ssl=true&ssl_cert_reqs=CERT_OPTIONAL")

# Select the database to use
db = client["lunge"]

data = db.users.find()
    
# Iterate over the data and print each document
for doc in data:
    print(doc)
