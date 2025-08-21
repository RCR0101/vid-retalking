#!/usr/bin/env python3
"""
Text-to-Video Pipeline Example using Qwen TTS + Video Retalking

This example demonstrates how to use the integrated pipeline to:
1. Convert text to speech using Qwen TTS
2. Apply lip sync to a video using the generated audio

Usage:
    python text_to_video_example.py

Prerequisites:
    1. Set DASHSCOPE_API_KEY environment variable
    2. Have necessary model checkpoints in 'checkpoints/' directory
    3. Install required dependencies: pip install -r requirements.txt
"""

import os
import subprocess
import sys

def check_requirements():
    """Check if all requirements are met"""
    
    # Check API key
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("❌ Error: DASHSCOPE_API_KEY environment variable not set")
        print("Please set your Qwen API key: export DASHSCOPE_API_KEY=your_api_key")
        return False
    
    # Check if example video exists
    example_video = "examples/face/1.mp4"
    if not os.path.exists(example_video):
        print(f"❌ Error: Example video not found at {example_video}")
        print("Please ensure example videos are available")
        return False
    
    # Check essential checkpoints
    essential_checkpoints = [
        "checkpoints/DNet.pt",
        "checkpoints/LNet.pth", 
        "checkpoints/ENet.pth",
        "checkpoints/face3d_pretrain_epoch_20.pth"
    ]
    
    missing_checkpoints = []
    for checkpoint in essential_checkpoints:
        if not os.path.exists(checkpoint):
            missing_checkpoints.append(checkpoint)
    
    if missing_checkpoints:
        print("❌ Error: Missing required checkpoints:")
        for checkpoint in missing_checkpoints:
            print(f"  - {checkpoint}")
        print("Please download the required model checkpoints")
        return False
    
    print("✅ All requirements met!")
    return True


def run_text_to_video_example():
    """Run the text-to-video example"""
    
    # Example configuration
    text = "Hello! This is a demonstration of the Qwen TTS integration with video retalking."
    input_video = "examples/face/1.mp4"
    output_video = "results/text_to_video_output.mp4"
    voice = "Dylan"  # Options: Cherry, Ethan, Chelsie, Serena, Dylan, Jada, Sunny
    
    print(f"🎬 Running Text-to-Video Pipeline")
    print(f"📝 Text: {text}")
    print(f"🎥 Input video: {input_video}")
    print(f"🔊 TTS Voice: {voice}")
    print(f"💾 Output: {output_video}")
    print()
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run the inference
    command = [
        "python", "inference_with_tts.py",
        "--face", input_video,
        "--text", text,
        "--tts_voice", voice,
        "--outfile", output_video
    ]
    
    try:
        print("🚀 Starting inference...")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("✅ Pipeline completed successfully!")
        print(f"📁 Output saved to: {output_video}")
        
        # Check if output file was created
        if os.path.exists(output_video):
            file_size = os.path.getsize(output_video) / (1024 * 1024)  # MB
            print(f"📊 Output file size: {file_size:.1f} MB")
        
    except subprocess.CalledProcessError as e:
        print("❌ Pipeline failed:")
        print(f"Error: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def show_usage_examples():
    """Show various usage examples"""
    print("\n📚 Usage Examples:")
    print()
    
    print("1. Basic text-to-video:")
    print('   python inference_with_tts.py --face video.mp4 --text "Hello world" --outfile output.mp4')
    print()
    
    print("2. With different voice:")
    print('   python inference_with_tts.py --face video.mp4 --text "Hello" --tts_voice Cherry --outfile output.mp4')
    print()
    
    print("3. Traditional audio input (still supported):")
    print('   python inference_with_tts.py --face video.mp4 --audio audio.wav --outfile output.mp4')
    print()
    
    print("4. Available TTS voices:")
    voices = ["Cherry", "Ethan", "Chelsie", "Serena", "Dylan", "Jada", "Sunny"]
    for voice in voices:
        print(f"   - {voice}")
    print()


def main():
    print("🎯 Qwen TTS + Video Retalking Integration")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        show_usage_examples()
        return
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please resolve the issues above before running the pipeline")
        return
    
    print()
    
    # Run example
    run_text_to_video_example()
    
    # Show additional usage examples
    show_usage_examples()


if __name__ == "__main__":
    main()