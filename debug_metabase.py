import os
import requests
import configparser
from pathlib import Path

# Load Configuration
config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

BASE_URL = config.get("metabase", "url", fallback="http://localhost:3000").rstrip('/') + '/api'
API_KEY = config.get("metabase", "api_key", fallback=None)

headers = {
    'Content-Type': 'application/json',
    'x-api-key': API_KEY
}

def list_dashboards():
    print(f"🔍 Connecting to {BASE_URL}...")
    try:
        # 1. Check User
        res = requests.get(f"{BASE_URL}/user/current", headers=headers)
        res.raise_for_status()
        print(f"✅ Authenticated as: {res.json().get('common_name')}")

        # 2. List Dashboards
        print("\n📋 Fetching Dashboards...")
        res = requests.get(f"{BASE_URL}/dashboard", headers=headers)
        res.raise_for_status()
        dashboards = res.json()
        
        print(f"{'ID':<5} | {'Name'}")
        print("-" * 30)
        for d in dashboards:
            print(f"{d['id']:<5} | {d['name']}")
            
        print("\n👉 Please check if ID 97 is in this list.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    list_dashboards()
