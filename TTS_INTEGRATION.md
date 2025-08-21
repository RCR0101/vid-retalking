# Open-Source TTS Integration for Video Retalking

This integration allows you to create lip-synced videos directly from text input using open-source Text-to-Speech (TTS) models, eliminating the need for pre-recorded audio files or API keys.

## Features

- **Text-to-Speech**: Convert any text to high-quality speech using Coqui TTS
- **Voice Cloning**: Clone any voice from just 3 seconds of audio (XTTS-v2)
- **17 Languages**: Multi-language support including English, Spanish, French, German, Italian, Portuguese, and more
- **No API Keys**: Completely free and runs locally
- **Multiple Models**: Choose from various TTS models based on your needs

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

### 1. Install Dependencies

```bash
pip install TTS torch torchaudio
```

**Note**: First run will automatically download the XTTS-v2 model (~1GB). Ensure you have sufficient disk space and internet connection.

### 2. Download Model Checkpoints

Ensure you have the required model checkpoints in the `checkpoints/` directory:
- `DNet.pt`
- `LNet.pth`
- `ENet.pth`
- `face3d_pretrain_epoch_20.pth`
- `GFPGANv1.3.pth`
- `shape_predictor_68_face_landmarks.dat`

## Usage

### Basic Text-to-Video

```bash
python inference_with_tts.py \
    --face input_video.mp4 \
    --text "Hello! This is a text-to-speech demonstration." \
    --tts_voice female_1 \
    --language en \
    --outfile output_with_tts.mp4
```

### With Voice Cloning

```bash
python inference_with_tts.py \
    --face person.mp4 \
    --text "Hello! I'm speaking with a cloned voice." \
    --tts_voice custom \
    --speaker_wav speaker_sample.wav \
    --language en \
    --outfile cloned_voice_output.mp4
```

### Multi-language Support

```bash
python inference_with_tts.py \
    --face person.mp4 \
    --text "Hola, esto es una demostración en español." \
    --tts_voice male_1 \
    --language es \
    --outfile spanish_output.mp4
```

### Run Example Script

```bash
python text_to_video_example.py
```

## API Reference

### OpenSourceTTS Class

```python
from utils.opensource_tts import OpenSourceTTS

# Initialize TTS
tts = OpenSourceTTS()  # Uses XTTS-v2 by default

# Generate speech with voice cloning
audio_path = tts.synthesize(
    text="Hello world",
    voice="custom",
    speaker_wav="speaker.wav",
    language="en",
    output_path="output.wav"
)

# List available voices and languages
voices = tts.list_voices()
languages = tts.list_languages()
```

### Convenience Function

```python
from utils.opensource_tts import text_to_audio

# Simple text-to-audio conversion
audio_path = text_to_audio(
    text="Hello world", 
    voice="female_1",
    language="en"
)

# With voice cloning
audio_path = text_to_audio(
    text="Hello world",
    voice="custom",
    speaker_wav="speaker.wav",
    language="en"
)
```

### Simple TTS (Lightweight Alternative)

```python
from utils.simple_tts import simple_text_to_audio

# Quick setup with lighter model (English only)
audio_path = simple_text_to_audio("Hello world")
```

## Command Line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--text` | Text to convert to speech | Required if no `--audio` | Any text |
| `--audio` | Audio file path | Required if no `--text` | Path to audio file |
| `--tts_voice` | TTS voice preset | female_1 | female_1, male_1, female_2, male_2, custom |
| `--speaker_wav` | Speaker audio for cloning | None | Path to 3+ second audio file |
| `--language` | Language code | en | en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko |
| `--tts_model` | TTS model to use | xtts_v2 | Any supported TTS model |
| `--face` | Input video/image path | Required | Path to video/image |
| `--outfile` | Output video path | results/output_with_tts.mp4 | Output path |

All other arguments from the original inference script are supported.

## Pipeline Flow

1. **Text Input**: Provide text via `--text` parameter
2. **TTS Generation**: Text is converted to speech using Qwen TTS
3. **Audio Processing**: Generated audio is processed for lip sync
4. **Video Processing**: Original video retalking pipeline processes the video
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

### English Text
```bash
python inference_with_tts.py \
    --face examples/face/1.mp4 \
    --text "Welcome to the future of video generation!" \
    --tts_voice male_1 \
    --language en \
    --outfile english_demo.mp4
```

### Spanish Text
```bash
python inference_with_tts.py \
    --face examples/face/2.mp4 \
    --text "¡Bienvenido al futuro de la generación de video!" \
    --tts_voice female_2 \
    --language es \
    --outfile spanish_demo.mp4
```

### Voice Cloning Example
```bash
python inference_with_tts.py \
    --face examples/face/3.mp4 \
    --text "This is my cloned voice speaking!" \
    --tts_voice custom \
    --speaker_wav my_voice_sample.wav \
    --language en \
    --outfile cloned_voice_demo.mp4
```

## Troubleshooting

1. **ModuleNotFoundError: TTS**: Install with `pip install TTS`
2. **Model download fails**: Check internet connection and disk space
3. **Voice cloning quality**: Use high-quality, 3+ second speaker samples with clear speech
4. **Long processing times**: Use GPU, reduce video resolution, or try simple TTS model
5. **Memory issues**: Reduce batch sizes or use CPU mode

## Integration with Existing Workflows

The TTS integration is backward compatible. Existing scripts using `--audio` parameter will continue to work unchanged. The new `--text` parameter provides an alternative input method.

You can also use the TTS module independently:

```python
from utils.opensource_tts import text_to_audio

# Generate audio file for later use
audio_path = text_to_audio(
    text="Your text here", 
    voice="female_1",
    language="en",
    output_path="my_audio.wav"
)

# Use with original inference script
# python inference.py --face video.mp4 --audio my_audio.wav --outfile result.mp4
```

## Model Options

### XTTS-v2 (Default)
- **Pros**: Voice cloning, 17 languages, high quality
- **Cons**: Large model (~1GB), requires speaker audio for cloning
- **Best for**: Voice cloning, multilingual content

### Simple TTS (Alternative)
- **Pros**: Lightweight, fast setup, no speaker audio needed
- **Cons**: English only, no voice cloning
- **Best for**: Quick testing, English-only content

```python
# Use simple TTS instead
from utils.simple_tts import simple_text_to_audio
audio_path = simple_text_to_audio("Hello world")
```