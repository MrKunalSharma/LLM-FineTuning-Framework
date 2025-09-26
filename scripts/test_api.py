"""Test script for API endpoints."""
import requests
import json
import time
import argparse
from typing import Dict, Any

class APITester:
    """Test API endpoints."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def test_health(self) -> bool:
        """Test health endpoint."""
        print("\n🏥 Testing health endpoint...")
        try:
            response = requests.get(f"{self.base_url}/health")
            data = response.json()
            print(f"✅ Health Status: {data['status']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            print(f"   Device: {data['device']}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Health check failed: {str(e)}")
            return False
    
    def test_model_info(self) -> bool:
        """Test model info endpoint."""
        print("\n📊 Testing model info endpoint...")
        try:
            response = requests.get(
                f"{self.base_url}/model/info",
                headers=self.headers
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model Type: {data['model_type']}")
                print(f"   Total Parameters: {data['parameters']['total']:,}")
                print(f"   Device: {data['device']}")
                return True
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Model info request failed: {str(e)}")
            return False
    
    def test_generate(self) -> bool:
        """Test generate endpoint."""
        print("\n🤖 Testing generate endpoint...")
        
        test_cases = [
            {
                "instruction": "Translate to French",
                "input_text": "Hello, how are you?",
                "max_new_tokens": 50
            },
            {
                "instruction": "Summarize the following text",
                "input_text": "Artificial intelligence is transforming the way we work and live.",
                "temperature": 0.5
            },
            {
                "instruction": "Write a Python function to calculate fibonacci numbers",
                "max_new_tokens": 150
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n  Test case {i}:")
            print(f"  Instruction: {test_case['instruction']}")
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/generate",
                    headers=self.headers,
                    json=test_case
                )
                
                if response.status_code == 200:
                    data = response.json()
                    elapsed = time.time() - start_time
                    
                    print(f"  ✅ Response: {data['response'][:100]}...")
                    print(f"     Tokens: {data['tokens_generated']}")
                    print(f"     Time: {elapsed:.2f}s")
                else:
                    print(f"  ❌ Failed: {response.status_code}")
                    print(f"     Error: {response.json()}")
                    
            except Exception as e:
                print(f"  ❌ Request failed: {str(e)}")
                return False
        
        return True
    
    def test_batch(self) -> bool:
        """Test batch endpoint."""
        print("\n📦 Testing batch endpoint...")
        
        batch_request = {
            "requests": [
                {"instruction": "Translate to Spanish", "input_text": "Good morning"},
                {"instruction": "What is 2+2?"},
                {"instruction": "Name three colors"}
            ]
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/batch",
                headers=self.headers,
                json=batch_request
            )
            
            if response.status_code == 200:
                data = response.json()
                elapsed = time.time() - start_time
                
                print(f"✅ Batch processed: {data['batch_size']} requests")
                print(f"   Total time: {elapsed:.2f}s")
                print(f"   Avg time per request: {elapsed/data['batch_size']:.2f}s")
                
                for i, resp in enumerate(data['responses'], 1):
                    print(f"\n   Response {i}: {resp['response'][:50]}...")
                
                return True
            else:
                print(f"❌ Batch failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Batch request failed: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all tests."""
        print(f"🧪 Testing API at {self.base_url}")
        print("=" * 50)
        
        tests = [
            self.test_health,
            self.test_model_info,
            self.test_generate,
            self.test_batch
        ]
        
        results = []
        for test in tests:
            results.append(test())
        
        print("\n" + "=" * 50)
        print("📋 Test Summary:")
        passed = sum(results)
        total = len(results)
        print(f"   Passed: {passed}/{total}")
        print(f"   Status: {'✅ All tests passed!' if passed == total else '❌ Some tests failed'}")

def main():
    parser = argparse.ArgumentParser(description="Test LLM API endpoints")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", type=str, default="your-secure-api-key", help="API key")
    
    args = parser.parse_args()
    
    tester = APITester(args.url, args.api_key)
    tester.run_all_tests()

if __name__ == "__main__":
    main()
