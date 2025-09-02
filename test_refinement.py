#!/usr/bin/env python3
"""
Simple test script to verify the prompt refinement tool works
"""

import requests
import json

def test_refinement():
    url = "http://127.0.0.1:5001/refine"
    
    test_data = {
        "prompt": "Write a function to calculate factorial",
        "attachments": "",
        "model": "gpt-4o",
        "temperature": 1.0,
        "max_tokens": 6000,
        "max_iterations": 2,
        "review_prompt": ""
    }
    
    print("🧪 Testing prompt refinement...")
    print(f"📝 Test prompt: {test_data['prompt']}")
    
    try:
        response = requests.post(url, json=test_data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! {result['iterations']} iterations completed")
            print(f"📄 Final refined prompt: {result['refined_prompt'][:100]}...")
            print(f"💾 Saved to: {result.get('saved_file', 'No file saved')}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False

if __name__ == "__main__":
    test_refinement()
