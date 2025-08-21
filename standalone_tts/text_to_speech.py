#!/usr/bin/env python3
"""
Standalone Text-to-Speech Script using Coqui TTS
Generates audio files that can be used with video-retalking

Usage:
    python text_to_speech.py --text "Hello world" --output audio.wav
    python text_to_speech.py --text "Hello" --speaker_wav voice_sample.wav --output cloned_audio.wav
"""

import os
import sys
import argparse
import tempfile
import torch
from pathlib import Path
from TTS.api import TTS
import warnings
warnings.filterwarnings("ignore")


class StandaloneTTS:
    """Standalone TTS using Coqui TTS"""
    
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"🔊 Loading TTS model: {model_name}")
        print(f"🖥️  Using device: {self.device}")
        
        if "xtts" in model_name.lower():
            print("📥 First run will download XTTS-v2 model (~1GB)")
        
        try:
            self.tts = TTS(model_name).to(self.device)
            print("✅ TTS model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load TTS model: {str(e)}")
            sys.exit(1)
    
    def text_to_speech(self, text, output_path, language="en", speaker_wav=None):
        """Convert text to speech"""
        print(f"🎤 Converting text to speech...")
        print(f"📝 Text: {text}")
        print(f"🌍 Language: {language}")
        
        try:
            if "xtts" in self.model_name.lower():
                if speaker_wav and os.path.exists(speaker_wav):
                    print(f"🎭 Using voice cloning with: {speaker_wav}")
                    self.tts.tts_to_file(
                        text=text,
                        speaker_wav=speaker_wav,
                        language=language,
                        file_path=output_path
                    )
                else:
                    print("⚠️  XTTS requires a speaker audio file for voice cloning")
                    print("💡 Switching to fallback model for default voice...")
                    # Use a simpler model as fallback
                    fallback_tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(self.device)
                    fallback_tts.tts_to_file(text=text, file_path=output_path)
            else:
                # Other TTS models
                self.tts.tts_to_file(text=text, file_path=output_path)
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✅ Audio generated successfully!")
                print(f"📁 Output: {output_path}")
                print(f"📊 File size: {file_size:,} bytes")
                return True
            else:
                print("❌ Failed to generate audio file")
                return False
                
        except Exception as e:
            print(f"❌ TTS synthesis failed: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Text-to-Speech using Coqui TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python text_to_speech.py --text "Hello world" --output hello.wav

  # Voice cloning
  python text_to_speech.py --text "Hello world" --speaker_wav speaker.wav --output cloned.wav

  # Different language
  python text_to_speech.py --text "Hola mundo" --language es --output spanish.wav

  # Use simple model (faster, English only)
  python text_to_speech.py --text "Hello" --model simple --output simple.wav
        """
    )
    
    parser.add_argument("--text", type=str, required=True, 
                        help="Text to convert to speech")
    parser.add_argument("--output", type=str, required=True, 
                        help="Output audio file path (.wav)")
    parser.add_argument("--language", type=str, default="en",
                        choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"],
                        help="Language code (default: en)")
    parser.add_argument("--speaker_wav", type=str, 
                        help="Speaker audio file for voice cloning (3+ seconds recommended)")
    parser.add_argument("--model", type=str, default="xtts", 
                        choices=["xtts", "simple"],
                        help="TTS model to use (default: xtts)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], 
                        help="Device to use (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.speaker_wav and not os.path.exists(args.speaker_wav):
        print(f"❌ Speaker audio file not found: {args.speaker_wav}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Select model
    if args.model == "xtts":
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    else:  # simple
        model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        if args.language != "en":
            print("⚠️  Simple model only supports English. Switching to English.")
            args.language = "en"
    
    print("🎯 Standalone Text-to-Speech")
    print("=" * 40)
    
    try:
        # Initialize TTS
        tts = StandaloneTTS(model_name=model_name, device=args.device)
        
        # Generate speech
        success = tts.text_to_speech(
            text=args.text,
            output_path=args.output,
            language=args.language,
            speaker_wav=args.speaker_wav
        )
        
        if success:
            print("\n🎉 Success! You can now use this audio file with video-retalking:")
            print(f"   python inference.py --face video.mp4 --audio {args.output} --outfile result.mp4")
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()