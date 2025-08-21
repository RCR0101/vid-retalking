import os
import tempfile
import torch
from typing import Optional, Dict, List
from TTS.api import TTS
import warnings
warnings.filterwarnings("ignore")


class OpenSourceTTS:
    """
    Open-source Text-to-Speech integration using Coqui TTS for video retalking pipeline
    """
    
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", device: Optional[str] = None):
        """
        Initialize Open Source TTS with Coqui TTS
        
        Args:
            model_name: TTS model to use (default: XTTS-v2 for voice cloning)
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
        
        # Built-in voices for different models
        self.builtin_voices = self._get_builtin_voices()
        
    def _get_builtin_voices(self) -> Dict[str, str]:
        """Get available built-in voices based on model"""
        if "xtts" in self.model_name.lower():
            return {
                'female_1': 'Clear female voice',
                'male_1': 'Clear male voice',
                'female_2': 'Expressive female voice', 
                'male_2': 'Expressive male voice',
                'custom': 'Use custom speaker audio file'
            }
        else:
            # For other models, try to get speaker list
            try:
                speakers = getattr(self.tts, 'speakers', None)
                if speakers:
                    return {speaker: f"Speaker {speaker}" for speaker in speakers}
            except:
                pass
            
            return {
                'default': 'Default voice',
                'custom': 'Use custom speaker audio file'
            }
    
    def synthesize(self, text: str, voice: str = "female_1", speaker_wav: Optional[str] = None,
                   language: str = "en", output_path: Optional[str] = None) -> str:
        """
        Convert text to speech using Coqui TTS
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (for non-XTTS models) or preset voice for XTTS
            speaker_wav: Path to speaker audio file for voice cloning (XTTS only)
            language: Language code (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko)
            output_path: Path to save audio file. If None, creates temporary file
            
        Returns:
            Path to the generated audio file
        """
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            output_path = temp_file.name
            temp_file.close()
        
        try:
            if "xtts" in self.model_name.lower():
                # XTTS model supports voice cloning
                if speaker_wav and os.path.exists(speaker_wav):
                    # Use custom speaker file
                    self.tts.tts_to_file(
                        text=text,
                        speaker_wav=speaker_wav,
                        language=language,
                        file_path=output_path
                    )
                else:
                    # Use built-in voice samples
                    sample_audio = self._get_sample_audio(voice)
                    if sample_audio:
                        self.tts.tts_to_file(
                            text=text,
                            speaker_wav=sample_audio,
                            language=language,
                            file_path=output_path
                        )
                    else:
                        raise ValueError(f"Voice '{voice}' not available and no speaker_wav provided")
            else:
                # Other TTS models
                if hasattr(self.tts, 'speakers') and voice in self.tts.speakers:
                    self.tts.tts_to_file(text=text, speaker=voice, file_path=output_path)
                else:
                    self.tts.tts_to_file(text=text, file_path=output_path)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error in TTS synthesis: {str(e)}")
    
    def _get_sample_audio(self, voice: str) -> Optional[str]:
        """
        Get path to sample audio for built-in voices
        This is a placeholder - in practice, you'd have sample audio files
        """
        # For demo purposes, return None to indicate no sample available
        # In a real implementation, you'd have sample audio files for different voices
        sample_dir = os.path.join(os.path.dirname(__file__), '..', 'samples', 'voices')
        sample_path = os.path.join(sample_dir, f"{voice}.wav")
        
        if os.path.exists(sample_path):
            return sample_path
        
        return None
    
    def list_voices(self) -> Dict[str, str]:
        """
        Get available voices and their descriptions
        
        Returns:
            Dictionary of voice names and descriptions
        """
        return self.builtin_voices.copy()
    
    def list_languages(self) -> List[str]:
        """
        Get supported languages for XTTS model
        
        Returns:
            List of supported language codes
        """
        if "xtts" in self.model_name.lower():
            return ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]
        else:
            return ["en"]  # Most other models support English by default


def text_to_audio(text: str, voice: str = "female_1", speaker_wav: Optional[str] = None,
                  language: str = "en", output_path: Optional[str] = None,
                  model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2") -> str:
    """
    Convenience function to convert text to audio using open-source TTS
    
    Args:
        text: Text to convert to speech
        voice: Voice to use or preset name
        speaker_wav: Path to speaker audio file for voice cloning
        language: Language code
        output_path: Output path for audio file
        model_name: TTS model to use
        
    Returns:
        Path to generated audio file
    """
    tts = OpenSourceTTS(model_name=model_name)
    return tts.synthesize(text, voice, speaker_wav, language, output_path)