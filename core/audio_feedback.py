# audio_feedback.py
"""
Audio Feedback System for PPE Verification — Prerecorded Audio Only
Plays .mp3 or .wav files from the configured audio folder.

FOLDER STRUCTURE (default: audio/):
  audio/
    compliant.mp3
    non_compliant.mp3
    missing_mask.mp3
    missing_haircap.mp3
    missing_gloves.mp3
    missing_boots.mp3
    missing_apron.mp3
    missing_long_sleeves.mp3

FILES can be .mp3 or .wav — the system checks for both automatically.

INSTALL DEPENDENCY (run once on your Pi):
  pip install pygame
"""

import os
import time
import threading
from queue import Queue, Empty

import pygame


class AudioFeedback:
    def __init__(self, config):
        """
        Initialize prerecorded audio feedback system.

        Args:
            config: Configuration module with audio settings
        """
        self.config = config
        self.enabled = config.ENABLE_AUDIO

        # Playback queue (non-blocking)
        self.audio_queue = Queue()
        self.is_playing = False

        # Cooldown tracking
        self.last_announcement = {}
        self.last_status = None
        self.last_periodic = 0

        # Interrupt flag — set True to make the worker stop waiting mid-playback
        self._interrupt = threading.Event()

        if not self.enabled:
            return

        # Initialize pygame mixer
        try:
            pygame.mixer.init()
            print("✓  Audio feedback initialized (prerecorded / pygame)")
        except Exception as e:
            print(f"✗  pygame mixer init failed: {e}")
            print("   Install with: pip install pygame")
            self.enabled = False
            return

        # Verify audio folder exists
        audio_dir = config.AUDIO_FILES_DIR
        if not os.path.exists(audio_dir):
            print(f"⚠  Audio folder not found: '{audio_dir}' — creating it")
            os.makedirs(audio_dir)

        # Scan and report which files are present / missing
        self._verify_audio_files()

        # Start background playback thread
        self._playback_thread = threading.Thread(
            target=self._playback_worker, daemon=True
        )
        self._playback_thread.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_audio_path(self, key):
        """
        Resolve a message key to an audio file path.
        Checks .mp3 first, then .wav. Returns None if neither exists.
        """
        base = os.path.join(self.config.AUDIO_FILES_DIR, key)
        for ext in (".mp3", ".wav"):
            path = base + ext
            if os.path.exists(path):
                return path
        return None

    def _verify_audio_files(self):
        """Print a checklist of expected audio files on startup."""
        expected_keys = list(self.config.AUDIO_MESSAGES.keys())
        print(f"\n  Audio file check ({self.config.AUDIO_FILES_DIR}/):")
        all_ok = True
        for key in expected_keys:
            path = self._get_audio_path(key)
            if path:
                print(f"    ✓  {os.path.basename(path)}")
            else:
                print(f"    ✗  {key}.mp3  ← FILE MISSING")
                all_ok = False
        if not all_ok:
            print(f"  ⚠  Missing files will be silently skipped during playback.\n")
        else:
            print(f"  All audio files present.\n")

    def _playback_worker(self):
        """
        Background thread — plays queued audio files one at a time.
        Stops immediately when interrupt() is called.
        """
        while True:
            try:
                file_path = self.audio_queue.get(timeout=1)
                if file_path:
                    self._interrupt.clear()   # Clear any previous interrupt
                    self.is_playing = True
                    try:
                        pygame.mixer.music.load(file_path)
                        # Apply current volume setting before every play
                        vol = max(0.0, min(1.0, getattr(self.config, 'AUDIO_VOLUME', 1.0)))
                        pygame.mixer.music.set_volume(vol)
                        pygame.mixer.music.play()
                        # Poll until done OR interrupted
                        while pygame.mixer.music.get_busy():
                            if self._interrupt.is_set():
                                pygame.mixer.music.stop()
                                break
                            time.sleep(0.05)
                    except Exception as e:
                        print(f"⚠  Audio playback error ({file_path}): {e}")
                    finally:
                        self.is_playing = False
            except Empty:
                continue
            except Exception as e:
                print(f"⚠  Playback worker error: {e}")
                self.is_playing = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def announce(self, message_key, force=False):
        """
        Queue an audio announcement by message key.

        Args:
            message_key: Key matching a file in AUDIO_FILES_DIR
                         e.g. 'compliant', 'missing_mask'
            force:       Skip cooldown check
        """
        if not self.enabled:
            return

        current_time = time.time()

        # Cooldown check
        if not force:
            last_time = self.last_announcement.get(message_key, 0)
            if current_time - last_time < self.config.AUDIO_COOLDOWN:
                return

        self.last_announcement[message_key] = current_time

        # Resolve file path
        file_path = self._get_audio_path(message_key)
        if file_path is None:
            print(f"⚠  No audio file for key: '{message_key}' (skipping)")
            return

        self.audio_queue.put(file_path)

    def interrupt(self):
        """
        Immediately stop the current audio and clear all pending announcements.

        Call this when advancing to the next PPE stage so leftover audio from
        the previous stage does not bleed into the new one.
        """
        if not self.enabled:
            return

        # 1. Drain the queue so nothing queued behind the current track plays
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break

        # 2. Signal the worker to stop waiting on the current track
        self._interrupt.set()

        # 3. Stop pygame directly as well (belt-and-suspenders)
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def stop(self):
        """Stop any active playback and disable the system."""
        self.enabled = False
        self.interrupt()

    def sync_volume(self):
        """
        Apply config.AUDIO_VOLUME to the pygame mixer immediately.
        Call this after reloading config so the new volume takes effect
        without waiting for the next announcement.
        """
        try:
            vol = max(0.0, min(1.0, getattr(self.config, 'AUDIO_VOLUME', 1.0)))
            pygame.mixer.music.set_volume(vol)
        except Exception:
            pass
