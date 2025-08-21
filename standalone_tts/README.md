# Standalone Text-to-Speech

This is a completely independent TTS script using Coqui TTS. It generates audio files that can be used with the video-retalking system.

## Setup

1. **Create virtual environment** (recommended):
```bash
python -m venv tts_env
source tts_env/bin/activate  # On Windows: tts_env\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Text-to-Speech
```bash
python text_to_speech.py --text "Hello world!" --output audio.wav
```

### Voice Cloning
```bash
python text_to_speech.py \
    --text "Hello, this is my cloned voice!" \
    --speaker_wav speaker_sample.wav \
    --output cloned_audio.wav
```

### Different Languages
```bash
# Spanish
python text_to_speech.py --text "Hola mundo" --language es --output spanish.wav

# French
python text_to_speech.py --text "Bonjour le monde" --language fr --output french.wav

# Chinese
python text_to_speech.py --text "你好世界" --language zh-cn --output chinese.wav
```

### Simple Model (Faster, English Only)
```bash
python text_to_speech.py --text "Hello world" --model simple --output simple.wav
```

## Supported Languages

- **English** (en) - Default
- **Spanish** (es)
- **French** (fr) 
- **German** (de)
- **Italian** (it)
- **Portuguese** (pt)
- **Polish** (pl)
- **Turkish** (tr)
- **Russian** (ru)
- **Dutch** (nl)
- **Czech** (cs)
- **Arabic** (ar)
- **Chinese** (zh-cn)
- **Japanese** (ja)
- **Hungarian** (hu)
- **Korean** (ko)

## Voice Cloning Tips

1. **Speaker Audio Requirements**:
   - 3+ seconds of clear speech
   - Good audio quality (no background noise)
   - Single speaker only
   - WAV format recommended

2. **Best Results**:
   - Use 5-10 seconds of audio
   - Multiple sentences work better than single words
   - Clear pronunciation and consistent volume

## Models

### XTTS-v2 (Default)
- **Size**: ~1GB download on first run
- **Features**: Voice cloning, 17 languages
- **Quality**: High
- **Speed**: Moderate

### Simple Model
- **Size**: ~100MB
- **Features**: English only, no voice cloning
- **Quality**: Good
- **Speed**: Fast

## Integration with Video-Retalking

Once you generate an audio file, use it with the video-retalking system:

```bash
# Generate audio
python text_to_speech.py --text "Your text here" --output generated_audio.wav

# Use with video-retalking
cd ../  # Go back to main directory
python inference.py --face input_video.mp4 --audio standalone_tts/generated_audio.wav --outfile result.mp4
```

## Examples

### Example 1: Basic English
```bash
python text_to_speech.py \
    --text "Welcome to our presentation today." \
    --output presentation_intro.wav
```

### Example 2: Voice Cloning
```bash
# First, record a 5-second sample of your voice as "my_voice.wav"
python text_to_speech.py \
    --text "This presentation will cover the latest developments in AI." \
    --speaker_wav my_voice.wav \
    --output cloned_presentation.wav
```

### Example 3: Multilingual
```bash
# Generate in multiple languages
python text_to_speech.py --text "Hello everyone" --language en --output intro_en.wav
python text_to_speech.py --text "Hola a todos" --language es --output intro_es.wav
python text_to_speech.py --text "Bonjour tout le monde" --language fr --output intro_fr.wav
```

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Ensure sufficient disk space (~1GB)
   - Try again (downloads can resume)

2. **CUDA Out of Memory**
   - Use `--device cpu` to force CPU mode
   - Try the simple model with `--model simple`

3. **Poor Voice Cloning Quality**
   - Use longer speaker audio (5-10 seconds)
   - Ensure speaker audio is clear and noise-free
   - Try different speaker samples

4. **Slow Performance**
   - Use GPU if available
   - Try simple model for faster results
   - Reduce text length for testing

### Performance Tips

- **GPU Recommended**: Significantly faster than CPU
- **First Run**: Model download takes time, subsequent runs are fast
- **Batch Processing**: Generate multiple files in one session to avoid model reload