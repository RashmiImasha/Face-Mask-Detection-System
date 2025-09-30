import time
import threading
import numpy as np
import pygame
import pyttsx3


class AlertSystem:
    """
    Alert system: Plays beep sound and speaks warning message.
    Supports cooldown and can be safely restarted multiple times.
    """

    def __init__(self, alert_cooldown=5):
        self.alert_cooldown = alert_cooldown
        self.last_no_mask_count = 0
        self.last_alert_time = 0
        self.tts_running = False
        self._init_audio()

    def _init_audio(self):
        """Initialize audio once"""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
        except Exception as e:
            print(f"pygame init error: {e}")

        try:
            if not hasattr(self, 'tts_engine'):
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 1.0)
        except Exception as e:
            print(f"TTS init error: {e}")

    def play_beep_sound(self):
        """Play a short beep sound"""
        try:
            duration = 500  # ms
            frequency = 800
            sample_rate = 22050
            samples = int(sample_rate * duration / 1000)
            wave = np.sin(2 * np.pi * frequency * np.arange(samples) / sample_rate)
            wave = (wave * 32767).astype(np.int16)
            stereo_wave = np.column_stack((wave, wave))
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.play()
        except Exception as e:
            print(f"Beep sound error: {e}")

    def speak_warning(self, message):
        """Speak warning message in a separate thread"""
        if self.tts_running:
            return
        self.tts_running = True
        thread = threading.Thread(target=self._speak_thread, args=(message,))
        thread.daemon = True
        thread.start()

    def _speak_thread(self, message):
        try:
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS thread error: {e}")
        finally:
            self.tts_running = False

    def trigger_alert(self, no_mask_count):
        """Trigger alert only if count changed and cooldown passed"""
        current_time = time.time()
        if no_mask_count == 0:
            self.last_no_mask_count = 0
            return False

        if no_mask_count != self.last_no_mask_count and \
           current_time - self.last_alert_time >= self.alert_cooldown:

            self.play_beep_sound()
            message = (
                "Warning! One person without mask detected. Please wear your mask immediately."
                if no_mask_count == 1
                else f"Warning! {no_mask_count} persons without masks detected. Please wear your masks immediately."
            )
            time.sleep(0.3)
            self.speak_warning(message)

            self.last_no_mask_count = no_mask_count
            self.last_alert_time = current_time
            return True
        return False

    def cleanup(self):
        """Optional: cleanup audio only on full app exit"""
        try:
            pygame.mixer.quit()
            self.tts_engine.stop()
        except Exception as e:
            print(f"Cleanup error: {e}")
