# test.py
import requests
import json

def main():
    base_url = 'http://127.0.0.1:5000/vectorstore'
    headers = {'Content-Type': 'application/json'}

    # Clear the collection
    print("About to clear the collection...\n")
    clear_payload = {
        "type": "clear",
        "collection": "amygdala"
    }
    response = requests.post(base_url, headers=headers, data=json.dumps(clear_payload))
    print(response.json())

    # Load data into the collection
    print("\nAbout to perform load...\n")
    load_payload = {
        "type": "load",
        "text": "I am thirsty and want a drink of water",
        "collection": "amygdala"
    }
    response = requests.post(base_url, headers=headers, data=json.dumps(load_payload))
    print(response.json())

    # Perform a search
    print("\nAbout to perform search...\n")
    search_payload = {
        "type": "search",
        "query": "I want a drink",
        "collection": "amygdala"
    }
    response = requests.post(base_url, headers=headers, data=json.dumps(search_payload))
    print(response.json())

if __name__ == "__main__":
    main()
