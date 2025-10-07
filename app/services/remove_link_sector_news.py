from pymongo import MongoClient
from datetime import date, timedelta

# MongoDB setup
client = MongoClient(
    "mongodb+srv://smartnest26:smartnest26@cluster0.ceho9t7.mongodb.net/"
    "?retryWrites=true&w=majority",
    tls=True,
    tlsAllowInvalidCertificates=True,
    tlsAllowInvalidHostnames=True,
    serverSelectionTimeoutMS=10000,
    connectTimeoutMS=10000
)
db = client["sector_news"]
collection = db["sector_articles"]

# Calculate the threshold date for deletion (7 days ago)
threshold_date = (date.today() - timedelta(days=6)).strftime("%Y-%m-%d")

# Remove all documents with Date < threshold_date
result = collection.delete_many({"Date": {"$lt": threshold_date}})
print(f"Deleted {result.deleted_count} documents older than {threshold_date}.")
