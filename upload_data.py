import os
import sys
import pandas as pd
import pymongo
import certifi
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.environ.get("MONGODB_URL")
DATABASE_NAME = "insurance_db"
COLLECTION_NAME = "insurance_data"

def upload_data(file_path: str):
    try:
        # Read CSV
        print(f"Reading data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")

        # Drop index column if exists
        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)

        # Convert to dictionary records
        records = df.to_dict(orient="records")
        print(f"Total records to upload: {len(records)}")

        # Connect to MongoDB
        print("Connecting to MongoDB Atlas...")
        ca = certifi.where()
        client = pymongo.MongoClient(MONGODB_URL, tlsCAFile=ca)

        # Select database and collection
        database = client[DATABASE_NAME]
        collection = database[COLLECTION_NAME]

        # Drop existing data if any
        collection.drop()
        print("Existing collection dropped")

        # Insert records
        collection.insert_many(records)
        print(f"✅ Successfully uploaded {len(records)} records to MongoDB")
        print(f"Database: {DATABASE_NAME}")
        print(f"Collection: {COLLECTION_NAME}")

        # Verify
        count = collection.count_documents({})
        print(f"✅ Verified: {count} documents in collection")

        client.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        raise e

if __name__ == "__main__":
    # Put the path to your downloaded train.csv here
    file_path = "train.csv"
    upload_data(file_path)