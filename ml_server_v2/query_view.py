import requests
import json
import sys
import argparse


base_url = "http://apoco:apoco2023@couchdb.apoco.com.cn/design_mate/_design/materials_view/_view/materials_view?id="

def get_document_by_id(document_id):
    url = f"{base_url}\"{document_id}\""
    print(url)
    response = requests.get(url)
    data = response.json()

    i = 0
    for row in data.get("rows", []):
        i += 1
        print(f"row key {row['key']} value {row['value']} index = {i}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve document by ID from CouchDB")
    parser.add_argument("document_id", help="ID of the document")
    args = parser.parse_args()
    document_id = args.document_id
    get_document_by_id(document_id)

