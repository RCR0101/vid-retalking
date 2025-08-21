# Quick Start Guide

This guide gets you up and running with text-to-video generation in 5 minutes.

## 🚀 Setup (One-time)

```bash
# 1. Setup TTS
cd standalone_tts/
pip install -r requirements.txt

# 2. Setup Video-Retalking  
cd ../
pip install -r requirements.txt

# 3. Download video-retalking model checkpoints (if not already done)
# Place these in checkpoints/ directory:
# - DNet.pt, LNet.pth, ENet.pth, face3d_pretrain_epoch_20.pth
# - GFPGANv1.3.pth, shape_predictor_68_face_landmarks.dat
```

## 🎬 Generate Your First Video

### Basic Example
```bash
# Step 1: Generate speech from text
cd standalone_tts/
python text_to_speech.py \
    --text "Hello! Welcome to our AI-powered presentation." \
    --output ../demo_audio.wav

# Step 2: Create lip-synced video
cd ../
python inference.py \
    --face examples/face/1.mp4 \
    --audio demo_audio.wav \
    --outfile my_first_video.mp4
```

### Voice Cloning Example
```bash
# Step 1: Generate speech with voice cloning (need speaker audio sample)
cd standalone_tts/
python text_to_speech.py \
    --text "This sounds exactly like my voice!" \
    --speaker_wav your_voice_sample.wav \
    --output ../cloned_audio.wav

# Step 2: Create video
cd ../
python inference.py \
    --face your_video.mp4 \
    --audio cloned_audio.wav \
    --outfile cloned_video.mp4
```

## 🌍 Multiple Languages

```bash
# Spanish
cd standalone_tts/
python text_to_speech.py --text "Hola, bienvenidos" --language es --output ../spanish.wav
cd ../
python inference.py --face video.mp4 --audio spanish.wav --outfile spanish_video.mp4

# French  
cd standalone_tts/
python text_to_speech.py --text "Bonjour tout le monde" --language fr --output ../french.wav
cd ../
python inference.py --face video.mp4 --audio french.wav --outfile french_video.mp4
```

## 📝 Notes

- **First TTS run**: Downloads ~1GB model, subsequent runs are fast
- **Speaker audio**: Use 3-10 seconds of clear speech for voice cloning
- **GPU recommended**: Much faster than CPU for both TTS and video processing
- **File paths**: Use absolute paths if having issues with relative paths

## 🔧 Troubleshooting

**TTS fails**: Install with `pip install TTS torch torchaudio`
**Video-retalking fails**: Ensure model checkpoints are in `checkpoints/` directory
**Out of memory**: Add `--device cpu` to TTS command, reduce video resolution
**Audio quality**: Use WAV format, ensure no background noise in speaker samples

## 🎯 Tips for Best Results

1. **Voice Cloning**: Record 5-10 seconds of natural speech
2. **Text**: Use punctuation for better prosody  
3. **Video**: Use videos with clear facial features
4. **Audio**: Generate audio at same sample rate as video (usually 16kHz or 44.1kHz)

Ready to create amazing AI videos! 🎉