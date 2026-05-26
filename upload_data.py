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
        if not MONGODB_URL:
            raise Exception("MONGODB_URL environment variable is not set in .env file")

        print(f"📂 Reading data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded. Shape: {df.shape}")

        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)
            print("🗑️  Dropped 'id' column")

        records = df.to_dict(orient="records")
        print(f"📦 Total records to upload: {len(records)}")

        print("🔌 Connecting to MongoDB Atlas...")
        ca = certifi.where()
        client = pymongo.MongoClient(MONGODB_URL, tlsCAFile=ca)

        database = client[DATABASE_NAME]
        collection = database[COLLECTION_NAME]

        collection.drop()
        print("🗑️  Existing collection cleared")

        collection.insert_many(records)
        print(f"✅ Successfully uploaded {len(records)} records")
        print(f"📁 Database : {DATABASE_NAME}")
        print(f"📁 Collection: {COLLECTION_NAME}")

        count = collection.count_documents({})
        print(f"✅ Verified : {count} documents in collection")

        client.close()
        print("\n🎉 Data upload complete!")

    except Exception as e:
        print(f"❌ Error during upload: {e}")
        raise e


if __name__ == "__main__":
    file_path = "train.csv"

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("Please place train.csv in the root of the project")
        sys.exit(1)

    upload_data(file_path)