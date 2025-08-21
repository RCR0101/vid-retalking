#!/usr/bin/env python3
"""
Test script for Qwen TTS integration
Tests the TTS module independently before full pipeline testing
"""

import os
import sys
from utils.qwen_tts import QwenTTS, text_to_audio

def test_api_key():
    """Test if API key is available"""
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("❌ DASHSCOPE_API_KEY not found in environment variables")
        print("Please set your API key: export DASHSCOPE_API_KEY=your_api_key")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    return True

def test_tts_class():
    """Test QwenTTS class functionality"""
    try:
        print("🔧 Testing QwenTTS class initialization...")
        tts = QwenTTS()
        
        print("📋 Available voices:")
        voices = tts.list_voices()
        for voice, description in voices.items():
            print(f"  - {voice}: {description}")
        
        return tts
    except Exception as e:
        print(f"❌ TTS class initialization failed: {e}")
        return None

def test_tts_synthesis(tts):
    """Test actual TTS synthesis"""
    test_text = "Hello, this is a test of Qwen TTS integration."
    
    try:
        print(f"🎤 Testing TTS synthesis...")
        print(f"Text: {test_text}")
        
        audio_path = tts.synthesize(
            text=test_text,
            voice="Dylan",
            output_path="temp_tts_test.wav"
        )
        
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            print(f"✅ TTS synthesis successful!")
            print(f"Audio file: {audio_path}")
            print(f"File size: {file_size} bytes")
            
            # Cleanup
            os.remove(audio_path)
            print("🧹 Test file cleaned up")
            return True
        else:
            print("❌ Audio file was not created")
            return False
            
    except Exception as e:
        print(f"❌ TTS synthesis failed: {e}")
        return False

def test_convenience_function():
    """Test the convenience function"""
    try:
        print("🎯 Testing convenience function...")
        
        audio_path = text_to_audio(
            "Testing convenience function",
            voice="Cherry",
            output_path="temp_convenience_test.wav"
        )
        
        if os.path.exists(audio_path):
            print("✅ Convenience function works!")
            os.remove(audio_path)
            print("🧹 Test file cleaned up")
            return True
        else:
            print("❌ Convenience function failed")
            return False
            
    except Exception as e:
        print(f"❌ Convenience function failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Qwen TTS Integration Test Suite")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: API Key
    print("\n[Test 1/4] Checking API Key...")
    if test_api_key():
        tests_passed += 1
    
    # Test 2: TTS Class
    print("\n[Test 2/4] Testing TTS Class...")
    tts = test_tts_class()
    if tts is not None:
        tests_passed += 1
    
    # Test 3: TTS Synthesis (only if class initialization succeeded)
    if tts is not None:
        print("\n[Test 3/4] Testing TTS Synthesis...")
        if test_tts_synthesis(tts):
            tests_passed += 1
    else:
        print("\n[Test 3/4] Skipping TTS Synthesis (class init failed)")
    
    # Test 4: Convenience Function
    print("\n[Test 4/4] Testing Convenience Function...")
    if test_convenience_function():
        tests_passed += 1
    
    # Results
    print(f"\n{'='*40}")
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! TTS integration is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Run: python text_to_video_example.py")
        print("2. Or use: python inference_with_tts.py --face video.mp4 --text 'Your text here'")
    else:
        print("❌ Some tests failed. Please resolve issues before using the pipeline.")
        
        if tests_passed == 0:
            print("\n💡 Troubleshooting tips:")
            print("- Ensure DASHSCOPE_API_KEY is set correctly")
            print("- Check internet connection")
            print("- Verify API key is valid and has quota")

if __name__ == "__main__":
    main()