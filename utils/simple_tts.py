import os
import tempfile
import torch
from typing import Optional
from TTS.api import TTS
import warnings
warnings.filterwarnings("ignore")


class SimpleTTS:
    """
    Simple TTS using lighter models for quick setup
    """
    
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC", device: Optional[str] = None):
        """
        Initialize Simple TTS with a lightweight model
        
        Args:
            model_name: TTS model to use (default: Tacotron2 for English)
            device: Device to run on (cuda/cpu). Auto-detected if None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        try:
            print(f"[TTS] Loading {model_name} on {self.device}...")
            self.tts = TTS(model_name).to(self.device)
            print("[TTS] Model loaded successfully!")
        except Exception as e:
            raise Exception(f"Failed to load TTS model: {str(e)}")
    
    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Convert text to speech using simple TTS
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file. If None, creates temporary file
            
        Returns:
            Path to the generated audio file
        """
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            output_path = temp_file.name
            temp_file.close()
        
        try:
            self.tts.tts_to_file(text=text, file_path=output_path)
            return output_path
            
        except Exception as e:
            raise Exception(f"Error in TTS synthesis: {str(e)}")


def simple_text_to_audio(text: str, output_path: Optional[str] = None) -> str:
    """
    Convenience function to convert text to audio using simple TTS
    
    Args:
        text: Text to convert to speech
        output_path: Output path for audio file
        
    Returns:
        Path to generated audio file
    """
    tts = SimpleTTS()
    return tts.synthesize(text, output_path)