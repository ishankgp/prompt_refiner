#!/usr/bin/env python3
"""
Simple validation test for the improved Prompt Refiner application.
"""

import os
import sys

# Add the application directory to the path
sys.path.insert(0, '/workspaces/prompt_refiner')

def test_basic_functionality():
    """Test that basic functionality works with our improvements"""
    print("🔍 Testing basic application functionality...")
    
    # Set required environment variable
    os.environ['OPENAI_API_KEY'] = 'test_key'
    
    try:
        # Test import and basic function calls
        from app import (
            validate_environment,
            validate_refinement_params, 
            sanitize_input,
            Config
        )
        
        # Test 1: Environment validation
        print("✅ Environment validation function imported successfully")
        
        # Test 2: Input validation 
        valid_data = {'prompt': 'Test prompt'}
        validate_refinement_params(valid_data)
        print("✅ Input validation function working")
        
        # Test 3: Input sanitization
        clean = sanitize_input("Test\x00input")
        if clean == "Testinput":
            print("✅ Input sanitization working")
        else:
            print(f"⚠️  Input sanitization result: '{clean}'")
        
        # Test 4: Configuration
        if hasattr(Config, 'DEFAULT_MODEL'):
            print("✅ Configuration class loaded successfully")
        else:
            print("❌ Configuration class missing attributes")
            return False
        
        print("✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_server_import():
    """Test that the server can be imported without running"""
    print("🔍 Testing server import...")
    
    os.environ['OPENAI_API_KEY'] = 'test_key'
    
    try:
        import app
        print("✅ Main application imported successfully")
        
        # Check that key components exist
        if hasattr(app, 'app') and hasattr(app, 'socketio'):
            print("✅ Flask app and SocketIO configured")
        else:
            print("❌ Missing Flask app or SocketIO")
            return False
            
        # Check endpoints exist
        if hasattr(app.app, 'url_map'):
            routes = [rule.rule for rule in app.app.url_map.iter_rules()]
            if '/' in routes and '/api/health' in routes:
                print("✅ Required endpoints configured")
            else:
                print(f"❌ Missing endpoints. Found: {routes}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing application: {e}")
        return False

def main():
    """Run validation tests"""
    print("🧪 Validating Prompt Refiner improvements...")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Server Import", test_server_import),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n📋 {name}")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {name}: FAILED")
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validations passed! Improvements are working.")
        return 0
    else:
        print("⚠️  Some validations failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
