import requests
from urllib.parse import quote
import sys
import argparse

def query_couchdb(doc_id):
    base_url = "http://apoco:apoco2023@couchdb.apoco.com.cn/design_mate/_design/materials_view/_view/materials_view?"
    url = f"{base_url}startkey=%5B%22{doc_id}%22%5D&endkey=%5B%22{doc_id}%22%2C%7B%7D%5D"
    #print("http://apoco:apoco2023@couchdb.apoco.com.cn/design_mate/_design/materials_view/_view/materials_view?startkey=%5B%2217000447665715886%22%5D&endkey=%5B%2217000447665715886%22%2C%7B%7D%5D")

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(data)
        i = 0
        for row in data.get("rows", []):
            i += 1
            print(f"row key {row['key']} value {row['value']} index = {i}")
    else:
        print(f"Error: {response.status_code}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve document by ID from CouchDB")
    parser.add_argument("document_id", help="ID of the document")
    args = parser.parse_args()
    doc_id = args.document_id
    query_couchdb(doc_id)
