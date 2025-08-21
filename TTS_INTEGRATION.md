# Qwen TTS Integration for Video Retalking

This integration allows you to create lip-synced videos directly from text input using Qwen's Text-to-Speech (TTS) API, eliminating the need for pre-recorded audio files.

## Features

- **Text-to-Speech**: Convert any text to high-quality speech using Qwen TTS
- **Multiple Voices**: Choose from 7 different voices including Chinese dialects
- **Seamless Integration**: Works with existing video retalking pipeline
- **Bilingual Support**: Supports both Chinese and English text

## Available Voices

| Voice Name | Description |
|------------|-------------|
| Cherry     | Female, Chinese-English bilingual |
| Ethan      | Male, Chinese-English bilingual |
| Chelsie    | Female, Chinese-English bilingual |
| Serena     | Female, Chinese-English bilingual |
| Dylan      | Male, Beijing dialect |
| Jada       | Female, Shanghai dialect |
| Sunny      | Female, Sichuan dialect |

## Setup

### 1. Install Dependencies

```bash
pip install dashscope requests
```

### 2. Get Qwen API Key

1. Sign up at [DashScope](https://dashscope.aliyun.com/)
2. Get your API key from the console
3. Set the environment variable:

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

### 3. Download Model Checkpoints

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
    --tts_voice Dylan \
    --outfile output_with_tts.mp4
```

### With Different Voice

```bash
python inference_with_tts.py \
    --face person.mp4 \
    --text "你好，这是中文语音合成演示。" \
    --tts_voice Jada \
    --outfile chinese_output.mp4
```

### Run Example Script

```bash
python text_to_video_example.py
```

## API Reference

### QwenTTS Class

```python
from utils.qwen_tts import QwenTTS

# Initialize TTS
tts = QwenTTS(api_key="your-api-key")  # or use environment variable

# Generate speech
audio_path = tts.synthesize(
    text="Hello world",
    voice="Dylan",
    output_path="output.wav"
)

# List available voices
voices = tts.list_voices()
```

### Convenience Function

```python
from utils.qwen_tts import text_to_audio

# Simple text-to-audio conversion
audio_path = text_to_audio("Hello world", voice="Cherry")
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--text` | Text to convert to speech | Required if no `--audio` |
| `--audio` | Audio file path | Required if no `--text` |
| `--tts_voice` | TTS voice name | Dylan |
| `--face` | Input video/image path | Required |
| `--outfile` | Output video path | results/output_with_tts.mp4 |

All other arguments from the original inference script are supported.

## Pipeline Flow

1. **Text Input**: Provide text via `--text` parameter
2. **TTS Generation**: Text is converted to speech using Qwen TTS
3. **Audio Processing**: Generated audio is processed for lip sync
4. **Video Processing**: Original video retalking pipeline processes the video
5. **Output**: Final lip-synced video is saved

## Error Handling

Common issues and solutions:

### API Key Issues
```
Error: DASHSCOPE_API_KEY must be set
```
**Solution**: Set your API key as environment variable

### Voice Not Found
```
Voice 'InvalidVoice' not available
```
**Solution**: Use one of the supported voices listed above

### TTS API Failure
```
TTS API call failed: [error message]
```
**Solution**: Check internet connection and API key validity

## Performance Notes

- TTS generation adds ~2-5 seconds to processing time
- Generated audio files are temporarily stored and cleaned up automatically
- Network connection required for TTS API calls

## Examples

### English Text
```bash
python inference_with_tts.py \
    --face examples/face/1.mp4 \
    --text "Welcome to the future of video generation!" \
    --tts_voice Ethan \
    --outfile english_demo.mp4
```

### Chinese Text with Dialect
```bash
python inference_with_tts.py \
    --face examples/face/2.mp4 \
    --text "欢迎使用智能视频生成技术！" \
    --tts_voice Sunny \
    --outfile chinese_dialect_demo.mp4
```

### Bilingual Text
```bash
python inference_with_tts.py \
    --face examples/face/3.mp4 \
    --text "Hello everyone, 大家好！Welcome to our demonstration." \
    --tts_voice Cherry \
    --outfile bilingual_demo.mp4
```

## Troubleshooting

1. **ModuleNotFoundError: dashscope**: Install with `pip install dashscope`
2. **API rate limits**: Wait and retry, or check your API quota
3. **Audio quality issues**: Try different voices or add punctuation for better prosody
4. **Long processing times**: Reduce video resolution or length for faster processing

## Integration with Existing Workflows

The TTS integration is backward compatible. Existing scripts using `--audio` parameter will continue to work unchanged. The new `--text` parameter provides an alternative input method.

You can also use the TTS module independently:

```python
from utils.qwen_tts import text_to_audio

# Generate audio file for later use
audio_path = text_to_audio("Your text here", voice="Dylan", output_path="my_audio.wav")

# Use with original inference script
# python inference.py --face video.mp4 --audio my_audio.wav --outfile result.mp4
```