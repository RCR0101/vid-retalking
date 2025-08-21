import os
import requests
import tempfile
import dashscope
from typing import Optional


class QwenTTS:
    """
    Qwen Text-to-Speech integration for video retalking pipeline
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Qwen TTS with API key
        
        Args:
            api_key: DashScope API key. If None, will try to get from environment
        """
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY must be set as environment variable or passed as parameter")
        
        dashscope.api_key = self.api_key
        
        # Available voices
        self.voices = {
            'Cherry': 'Female, Chinese-English bilingual',
            'Ethan': 'Male, Chinese-English bilingual', 
            'Chelsie': 'Female, Chinese-English bilingual',
            'Serena': 'Female, Chinese-English bilingual',
            'Dylan': 'Male, Beijing dialect',
            'Jada': 'Female, Shanghai dialect',
            'Sunny': 'Female, Sichuan dialect'
        }
    
    def synthesize(self, text: str, voice: str = "Dylan", model: str = "qwen-tts-latest", 
                   output_path: Optional[str] = None) -> str:
        """
        Convert text to speech using Qwen TTS
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (Cherry, Ethan, Chelsie, Serena, Dylan, Jada, Sunny)
            model: TTS model to use
            output_path: Path to save audio file. If None, creates temporary file
            
        Returns:
            Path to the generated audio file
        """
        if voice not in self.voices:
            raise ValueError(f"Voice '{voice}' not available. Choose from: {list(self.voices.keys())}")
        
        try:
            response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
                model=model,
                text=text,
                voice=voice
            )
            
            if response.status_code == 200:
                audio_url = response.output.audio["url"]
                
                # Download the audio file
                audio_response = requests.get(audio_url)
                audio_response.raise_for_status()
                
                # Save to file
                if output_path is None:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    output_path = temp_file.name
                    temp_file.close()
                
                with open(output_path, 'wb') as f:
                    f.write(audio_response.content)
                
                return output_path
            else:
                raise Exception(f"TTS API call failed: {response.message}")
                
        except Exception as e:
            raise Exception(f"Error in TTS synthesis: {str(e)}")
    
    def list_voices(self) -> dict:
        """
        Get available voices and their descriptions
        
        Returns:
            Dictionary of voice names and descriptions
        """
        return self.voices.copy()


def text_to_audio(text: str, voice: str = "Dylan", output_path: Optional[str] = None) -> str:
    """
    Convenience function to convert text to audio using Qwen TTS
    
    Args:
        text: Text to convert to speech
        voice: Voice to use
        output_path: Output path for audio file
        
    Returns:
        Path to generated audio file
    """
    tts = QwenTTS()
    return tts.synthesize(text, voice, output_path=output_path)