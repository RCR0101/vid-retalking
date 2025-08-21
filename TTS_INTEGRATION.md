# Text-to-Speech + Video Retalking Workflow

This guide shows how to use the standalone TTS script to generate audio, then use that audio with the video-retalking system. The TTS and video-retalking components are completely separate, avoiding dependency conflicts.

## Workflow Overview

1. **Step 1**: Use standalone TTS script to generate audio from text
2. **Step 2**: Use generated audio with video-retalking system
3. **Result**: Lip-synced video with speech from your text

## Benefits

- **Clean Separation**: No dependency conflicts between TTS and video-retalking
- **Voice Cloning**: Clone any voice from just 3 seconds of audio
- **17 Languages**: Multi-language support 
- **No API Keys**: Completely free and runs locally
- **Flexible**: Use any audio source with video-retalking

## Available Voice Options

### XTTS-v2 (Default - Voice Cloning Model)
| Voice Preset | Description |
|-------------|-------------|
| female_1    | Clear female voice |
| male_1      | Clear male voice |
| female_2    | Expressive female voice |
| male_2      | Expressive male voice |
| custom      | Use your own speaker audio file |

### Supported Languages (XTTS-v2)
English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko)

## Setup

### 1. Setup TTS Environment

```bash
cd standalone_tts/
pip install -r requirements.txt
```

**Note**: First run will download the XTTS-v2 model (~1GB).

### 2. Setup Video-Retalking Environment

```bash
cd ../  # Back to main directory
pip install -r requirements.txt
```

Ensure you have the required model checkpoints in the `checkpoints/` directory:
- `DNet.pt`, `LNet.pth`, `ENet.pth`, `face3d_pretrain_epoch_20.pth`
- `GFPGANv1.3.pth`, `shape_predictor_68_face_landmarks.dat`

## Usage

### Complete Workflow

**Step 1: Generate Audio**
```bash
cd standalone_tts/
python text_to_speech.py --text "Hello! This is a demonstration." --output ../generated_audio.wav
```

**Step 2: Create Video**
```bash
cd ../
python inference.py --face input_video.mp4 --audio generated_audio.wav --outfile result.mp4
```

### Voice Cloning Workflow

**Step 1: Generate Audio with Voice Cloning**
```bash
cd standalone_tts/
python text_to_speech.py \
    --text "Hello! I'm speaking with a cloned voice." \
    --speaker_wav speaker_sample.wav \
    --output ../cloned_audio.wav
```

**Step 2: Create Video**
```bash
cd ../
python inference.py --face input_video.mp4 --audio cloned_audio.wav --outfile cloned_result.mp4
```

### Multi-language Workflow

**Step 1: Generate Spanish Audio**
```bash
cd standalone_tts/
python text_to_speech.py \
    --text "Hola, esto es una demostración en español." \
    --language es \
    --output ../spanish_audio.wav
```

**Step 2: Create Video**
```bash
cd ../
python inference.py --face input_video.mp4 --audio spanish_audio.wav --outfile spanish_result.mp4
```

## Command Line Reference

### TTS Script Options

```bash
cd standalone_tts/
python text_to_speech.py [OPTIONS]
```

**Required Arguments:**
- `--text "Your text here"` - Text to synthesize
- `--output path/to/output.wav` - Output audio file

**Optional Arguments:**
- `--language en` - Language code (default: en)
- `--speaker_wav speaker.wav` - Speaker audio for voice cloning
- `--model xtts` - Model type (xtts or simple)
- `--device cuda` - Device to use (cuda or cpu)

### Video-Retalking Options

```bash
python inference.py [OPTIONS]
```

**Key Arguments:**
- `--face input_video.mp4` - Input video file
- `--audio generated_audio.wav` - Audio file (from TTS)
- `--outfile result.mp4` - Output video file

## Directory Structure

```
video-retalking/
├── standalone_tts/           # TTS Environment
│   ├── text_to_speech.py    # Main TTS script
│   ├── requirements.txt     # TTS dependencies
│   └── README.md           # TTS documentation
├── inference.py            # Video-retalking script
├── requirements.txt        # Video-retalking dependencies
├── models/                 # Neural network models
├── utils/                  # Utility functions
├── third_part/            # Third-party code
└── examples/              # Sample videos/audio

## Pipeline Flow

1. **Text Input**: Provide text to standalone TTS script
2. **TTS Generation**: Text is converted to speech and saved as WAV file
3. **Audio Input**: Generated WAV file is passed to video-retalking
4. **Video Processing**: Video-retalking processes video with audio
5. **Output**: Final lip-synced video is saved

## Error Handling

Common issues and solutions:

### Model Download Issues
```
Failed to load TTS model
```
**Solution**: Ensure stable internet connection and sufficient disk space (~1GB)

### Voice Cloning Issues
```
Voice 'custom' not available and no speaker_wav provided
```
**Solution**: Provide `--speaker_wav` with a 3+ second audio sample when using custom voice

### Memory Issues
```
CUDA out of memory
```
**Solution**: Use CPU mode or reduce batch sizes, or try the simple TTS model

## Performance Notes

- **First Run**: Model download (~1GB) may take 5-15 minutes
- **Subsequent Runs**: TTS generation adds ~5-15 seconds to processing time
- **Voice Cloning**: Requires 3+ second speaker audio sample for best results
- **GPU vs CPU**: GPU significantly faster for inference
- **Storage**: Models cached locally after first download

## Examples

### English Example
```bash
# Step 1: Generate English audio
cd standalone_tts/
python text_to_speech.py \
    --text "Welcome to the future of video generation!" \
    --output ../english_audio.wav

# Step 2: Create video
cd ../
python inference.py \
    --face examples/face/1.mp4 \
    --audio english_audio.wav \
    --outfile english_demo.mp4
```

### Spanish Example
```bash
# Step 1: Generate Spanish audio
cd standalone_tts/
python text_to_speech.py \
    --text "¡Bienvenido al futuro de la generación de video!" \
    --language es \
    --output ../spanish_audio.wav

# Step 2: Create video
cd ../
python inference.py \
    --face examples/face/2.mp4 \
    --audio spanish_audio.wav \
    --outfile spanish_demo.mp4
```

### Voice Cloning Example
```bash
# Step 1: Generate cloned voice audio
cd standalone_tts/
python text_to_speech.py \
    --text "This is my cloned voice speaking!" \
    --speaker_wav my_voice_sample.wav \
    --output ../cloned_audio.wav

# Step 2: Create video
cd ../
python inference.py \
    --face examples/face/3.mp4 \
    --audio cloned_audio.wav \
    --outfile cloned_demo.mp4
```

## Troubleshooting

1. **ModuleNotFoundError: TTS**: Install with `pip install TTS`
2. **Model download fails**: Check internet connection and disk space
3. **Voice cloning quality**: Use high-quality, 3+ second speaker samples with clear speech
4. **Long processing times**: Use GPU, reduce video resolution, or try simple TTS model
5. **Memory issues**: Reduce batch sizes or use CPU mode

## Deployment to SSH Server

### Method 1: Copy Entire Directory
```bash
# Copy project to server
scp -r video-retalking/ user@server:/path/to/destination/

# SSH to server and setup
ssh user@server
cd /path/to/destination/video-retalking/

# Setup TTS environment
cd standalone_tts/
python -m venv tts_env
source tts_env/bin/activate
pip install -r requirements.txt

# Setup video-retalking environment
cd ../
python -m venv video_env  
source video_env/bin/activate
pip install -r requirements.txt
```

### Method 2: Separate Upload
```bash
# Upload TTS component
scp -r standalone_tts/ user@server:/path/to/tts/

# Upload video-retalking component (excluding TTS files)
rsync -av --exclude='standalone_tts/' video-retalking/ user@server:/path/to/video-retalking/
```

## Advanced Usage

### Batch Processing
```bash
# Generate multiple audio files
cd standalone_tts/
python text_to_speech.py --text "First sentence" --output ../audio1.wav
python text_to_speech.py --text "Second sentence" --output ../audio2.wav
python text_to_speech.py --text "Third sentence" --output ../audio3.wav

# Process multiple videos
cd ../
python inference.py --face video1.mp4 --audio audio1.wav --outfile result1.mp4
python inference.py --face video2.mp4 --audio audio2.wav --outfile result2.mp4
python inference.py --face video3.mp4 --audio audio3.wav --outfile result3.mp4
```

### Environment Separation Benefits

1. **No Dependency Conflicts**: TTS and video-retalking use different Python versions
2. **Easy Deployment**: Upload components separately to different environments
3. **Modular Updates**: Update TTS or video-retalking independently
4. **Flexible Usage**: Use any audio source with video-retalking
5. **Clean Development**: Work on components without version hell