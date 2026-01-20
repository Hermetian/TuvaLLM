#!/usr/bin/env python3
"""
TuvaLLM Transcript Corrector
Interactive interface for correcting MMS transcriptions with audio playback.

Features:
- Side-by-side MMS vs Whisper comparison
- Audio playback for each segment
- Quality scoring (1-5 scale)
- Vocabulary building from confirmed words
- Progress tracking and save/resume
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Dict, List, Any

sys.path.insert(0, '/Users/discordwell/Library/Python/3.9/lib/python/site-packages')

# Base paths
PROJECT_ROOT = Path("/Users/discordwell/TuvaLLM")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
PROCESSED_DIR.mkdir(exist_ok=True)
(PROCESSED_DIR / "corrections").mkdir(exist_ok=True)


def load_transcripts() -> Dict[str, Any]:
    """Load both MMS and Whisper transcripts for comparison."""
    transcripts = {}

    # Load MMS Samoan transcript
    mms_file = TRANSCRIPTS_DIR / "mms_samoan_transcript.json"
    if mms_file.exists():
        with open(mms_file, "r", encoding="utf-8") as f:
            transcripts["mms"] = json.load(f)

    # Load Whisper transcript if available
    whisper_files = list(TRANSCRIPTS_DIR.glob("whisper_*.json"))
    if whisper_files:
        # Use most recent
        whisper_file = sorted(whisper_files)[-1]
        with open(whisper_file, "r", encoding="utf-8") as f:
            transcripts["whisper"] = json.load(f)

    return transcripts


def load_corrections() -> Dict[str, Any]:
    """Load existing corrections if any."""
    corrections_file = PROCESSED_DIR / "corrections" / "corrections.json"
    if corrections_file.exists():
        with open(corrections_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "segments": {},
        "vocabulary": {},
        "metadata": {
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "audio_file": "grn_tuvalu_01.mp3"
        }
    }


def save_corrections(corrections: Dict[str, Any]):
    """Save corrections to file."""
    corrections["metadata"]["last_modified"] = datetime.now().isoformat()
    corrections_file = PROCESSED_DIR / "corrections" / "corrections.json"
    with open(corrections_file, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)
    print(f"Saved to {corrections_file}")


def export_vocabulary(corrections: Dict[str, Any]):
    """Export confirmed vocabulary to separate file."""
    vocab = corrections.get("vocabulary", {})
    vocab_file = PROCESSED_DIR / "corrections" / "tuvaluan_vocabulary.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary exported to {vocab_file}")


def get_matching_whisper_segment(whisper_data: Dict, start: float, end: float) -> Optional[str]:
    """Find Whisper text that overlaps with given time range."""
    if not whisper_data:
        return None

    # Try to find in auto-detected results
    for key in whisper_data:
        if "segments" in whisper_data[key]:
            segments = whisper_data[key]["segments"]
            matching_text = []
            for seg in segments:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                # Check for overlap
                if seg_start < end and seg_end > start:
                    matching_text.append(seg.get("text", ""))
            if matching_text:
                return " ".join(matching_text)

    return None


class AudioPlayer:
    """Simple audio player for segment playback using pygame."""

    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        self.initialized = False
        self._waveform = None
        self._sample_rate = None

    def _load_audio(self):
        """Load audio file for segment extraction."""
        if self._waveform is not None:
            return

        try:
            import torchaudio
            self._waveform, self._sample_rate = torchaudio.load(self.audio_path)
            # Convert to mono if stereo
            if self._waveform.shape[0] > 1:
                self._waveform = self._waveform.mean(dim=0, keepdim=True)
            self._waveform = self._waveform.squeeze().numpy()
        except Exception as e:
            print(f"Error loading audio: {e}")

    def init_pygame(self):
        """Initialize pygame mixer."""
        if self.initialized:
            return True
        try:
            import pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=1)
            self.initialized = True
            return True
        except Exception as e:
            print(f"Could not initialize audio playback: {e}")
            return False

    def play_segment(self, start_time: float, end_time: float):
        """Play a specific segment of the audio."""
        self._load_audio()
        if self._waveform is None:
            print("Audio not loaded")
            return

        try:
            import pygame
            import numpy as np
            import tempfile
            import soundfile as sf

            if not self.init_pygame():
                return

            # Extract segment
            start_sample = int(start_time * self._sample_rate)
            end_sample = int(end_time * self._sample_rate)
            segment = self._waveform[start_sample:end_sample]

            # Save to temp file and play
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, segment, self._sample_rate)
                pygame.mixer.music.load(tmp.name)
                pygame.mixer.music.play()

                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)

                # Cleanup
                pygame.mixer.music.unload()
                os.unlink(tmp.name)

        except Exception as e:
            print(f"Playback error: {e}")

    def stop(self):
        """Stop current playback."""
        try:
            import pygame
            if self.initialized:
                pygame.mixer.music.stop()
        except:
            pass


def run_interactive_corrector():
    """Run the interactive correction interface."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, IntPrompt, Confirm
    from rich.text import Text

    console = Console()

    # Load data
    console.print("[bold blue]Loading transcripts...[/bold blue]")
    transcripts = load_transcripts()
    corrections = load_corrections()

    if "mms" not in transcripts:
        console.print("[bold red]Error: No MMS transcript found![/bold red]")
        return

    mms_data = transcripts["mms"]
    whisper_data = transcripts.get("whisper", {})
    segments = mms_data["segments"]

    # Initialize audio player
    audio_file = RAW_DIR / "grn_tuvalu_01.mp3"
    player = AudioPlayer(str(audio_file))

    console.print(Panel.fit(
        "[bold green]TuvaLLM Transcript Corrector[/bold green]\n\n"
        f"Total segments: {len(segments)}\n"
        f"Already corrected: {len(corrections['segments'])}\n"
        f"Vocabulary entries: {len(corrections['vocabulary'])}",
        title="Welcome"
    ))

    # Commands help
    console.print("\n[bold]Commands:[/bold]")
    console.print("  [cyan]p[/cyan] - Play audio segment")
    console.print("  [cyan]r[/cyan] - Replay audio")
    console.print("  [cyan]n[/cyan] - Next segment")
    console.print("  [cyan]b[/cyan] - Previous segment")
    console.print("  [cyan]g[/cyan] - Go to segment number")
    console.print("  [cyan]e[/cyan] - Edit/correct transcript")
    console.print("  [cyan]s[/cyan] - Set quality score (1-5)")
    console.print("  [cyan]v[/cyan] - Add word to vocabulary")
    console.print("  [cyan]w[/cyan] - Save and exit")
    console.print("  [cyan]q[/cyan] - Quit without saving")
    console.print("  [cyan]x[/cyan] - Export vocabulary")
    console.print()

    # Find first uncorrected segment or start at 0
    current_idx = 0
    for i, seg in enumerate(segments):
        seg_key = f"{seg['start']:.1f}-{seg['end']:.1f}"
        if seg_key not in corrections["segments"]:
            current_idx = i
            break

    while True:
        seg = segments[current_idx]
        seg_key = f"{seg['start']:.1f}-{seg['end']:.1f}"

        # Build display
        console.clear()
        console.print(f"[bold]Segment {current_idx + 1}/{len(segments)}[/bold]")
        console.print(f"Time: {seg['start']:.1f}s - {seg['end']:.1f}s\n")

        # Show MMS transcript
        mms_text = seg.get("text", "")
        console.print(Panel(mms_text, title="[blue]MMS (Samoan proxy)[/blue]", border_style="blue"))

        # Show Whisper transcript if available
        whisper_text = get_matching_whisper_segment(whisper_data, seg['start'], seg['end'])
        if whisper_text:
            console.print(Panel(whisper_text, title="[yellow]Whisper[/yellow]", border_style="yellow"))

        # Show correction if exists
        if seg_key in corrections["segments"]:
            corr = corrections["segments"][seg_key]
            console.print(Panel(
                f"[green]{corr.get('corrected_text', '')}[/green]\n"
                f"Quality: {corr.get('quality', '?')}/5",
                title="[green]Corrected[/green]",
                border_style="green"
            ))

        # Command prompt
        console.print()
        cmd = Prompt.ask("Command", default="p").lower().strip()

        if cmd == "p" or cmd == "r":
            console.print("[dim]Playing audio...[/dim]")
            player.play_segment(seg['start'], seg['end'])

        elif cmd == "n":
            if current_idx < len(segments) - 1:
                current_idx += 1
            else:
                console.print("[yellow]Already at last segment[/yellow]")
                input("Press Enter to continue...")

        elif cmd == "b":
            if current_idx > 0:
                current_idx -= 1
            else:
                console.print("[yellow]Already at first segment[/yellow]")
                input("Press Enter to continue...")

        elif cmd == "g":
            try:
                num = IntPrompt.ask("Go to segment number", default=current_idx + 1)
                if 1 <= num <= len(segments):
                    current_idx = num - 1
                else:
                    console.print("[red]Invalid segment number[/red]")
                    input("Press Enter to continue...")
            except:
                pass

        elif cmd == "e":
            # Get existing correction or MMS text as default
            existing = corrections["segments"].get(seg_key, {}).get("corrected_text", mms_text)
            console.print(f"\n[dim]Current: {existing}[/dim]")
            new_text = Prompt.ask("Corrected text", default=existing)

            # Initialize or update correction
            if seg_key not in corrections["segments"]:
                corrections["segments"][seg_key] = {
                    "original_mms": mms_text,
                    "whisper": whisper_text,
                    "start": seg['start'],
                    "end": seg['end']
                }
            corrections["segments"][seg_key]["corrected_text"] = new_text
            console.print("[green]Correction saved[/green]")
            input("Press Enter to continue...")

        elif cmd == "s":
            try:
                score = IntPrompt.ask("Quality score (1-5)", default=3)
                score = max(1, min(5, score))
                if seg_key not in corrections["segments"]:
                    corrections["segments"][seg_key] = {
                        "original_mms": mms_text,
                        "whisper": whisper_text,
                        "start": seg['start'],
                        "end": seg['end'],
                        "corrected_text": mms_text  # Default to MMS if not edited
                    }
                corrections["segments"][seg_key]["quality"] = score
                console.print(f"[green]Score set to {score}[/green]")
                input("Press Enter to continue...")
            except:
                pass

        elif cmd == "v":
            word = Prompt.ask("Word to add to vocabulary")
            if word:
                meaning = Prompt.ask("Meaning/notes (optional)", default="")
                pos = Prompt.ask("Part of speech (optional)", default="")
                corrections["vocabulary"][word.lower()] = {
                    "word": word,
                    "meaning": meaning,
                    "pos": pos,
                    "added": datetime.now().isoformat()
                }
                console.print(f"[green]Added '{word}' to vocabulary[/green]")
                input("Press Enter to continue...")

        elif cmd == "w":
            save_corrections(corrections)
            console.print("[bold green]Saved![/bold green]")
            break

        elif cmd == "x":
            export_vocabulary(corrections)
            input("Press Enter to continue...")

        elif cmd == "q":
            if Confirm.ask("Quit without saving?"):
                break
        else:
            console.print("[dim]Unknown command[/dim]")
            input("Press Enter to continue...")

    player.stop()
    console.print("\n[bold]Session ended.[/bold]")

    # Show summary
    corrected_count = len(corrections["segments"])
    high_quality = sum(1 for s in corrections["segments"].values() if s.get("quality", 0) >= 4)
    console.print(f"\nTotal corrected: {corrected_count}")
    console.print(f"High quality (4-5): {high_quality}")
    console.print(f"Vocabulary entries: {len(corrections['vocabulary'])}")


def batch_score_segments():
    """Quick batch scoring interface for triaging segments."""
    from rich.console import Console
    from rich.prompt import IntPrompt, Confirm

    console = Console()

    transcripts = load_transcripts()
    corrections = load_corrections()

    if "mms" not in transcripts:
        console.print("[bold red]No MMS transcript found![/bold red]")
        return

    segments = transcripts["mms"]["segments"]
    audio_file = RAW_DIR / "grn_tuvalu_01.mp3"
    player = AudioPlayer(str(audio_file))

    console.print("[bold]Batch Scoring Mode[/bold]")
    console.print("Listen to each segment and rate quality 1-5")
    console.print("Press 0 to skip, 6 to go back\n")

    i = 0
    while i < len(segments):
        seg = segments[i]
        seg_key = f"{seg['start']:.1f}-{seg['end']:.1f}"

        # Skip already scored
        if seg_key in corrections["segments"] and "quality" in corrections["segments"][seg_key]:
            i += 1
            continue

        console.print(f"\n[{i+1}/{len(segments)}] {seg['start']:.1f}s - {seg['end']:.1f}s")
        console.print(f"  MMS: {seg['text'][:100]}...")

        player.play_segment(seg['start'], seg['end'])

        try:
            score = IntPrompt.ask("Score (1-5, 0=skip, 6=back, 9=quit)", default=0)

            if score == 9:
                break
            elif score == 6:
                i = max(0, i - 1)
                continue
            elif score == 0:
                i += 1
                continue
            elif 1 <= score <= 5:
                if seg_key not in corrections["segments"]:
                    corrections["segments"][seg_key] = {
                        "original_mms": seg['text'],
                        "start": seg['start'],
                        "end": seg['end'],
                        "corrected_text": seg['text']
                    }
                corrections["segments"][seg_key]["quality"] = score
                i += 1
        except KeyboardInterrupt:
            break

    player.stop()

    if Confirm.ask("Save scores?"):
        save_corrections(corrections)


def show_statistics():
    """Show correction statistics."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    corrections = load_corrections()

    transcripts = load_transcripts()
    total_segments = len(transcripts.get("mms", {}).get("segments", []))

    # Count by quality
    quality_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "unrated": 0}
    for seg in corrections["segments"].values():
        q = seg.get("quality")
        if q in quality_counts:
            quality_counts[q] += 1
        else:
            quality_counts["unrated"] += 1

    table = Table(title="Correction Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total segments", str(total_segments))
    table.add_row("Corrected", str(len(corrections["segments"])))
    table.add_row("Remaining", str(total_segments - len(corrections["segments"])))
    table.add_row("---", "---")
    table.add_row("Quality 5 (excellent)", str(quality_counts[5]))
    table.add_row("Quality 4 (good)", str(quality_counts[4]))
    table.add_row("Quality 3 (fair)", str(quality_counts[3]))
    table.add_row("Quality 2 (poor)", str(quality_counts[2]))
    table.add_row("Quality 1 (bad)", str(quality_counts[1]))
    table.add_row("Unrated", str(quality_counts["unrated"]))
    table.add_row("---", "---")
    table.add_row("Vocabulary entries", str(len(corrections["vocabulary"])))

    console.print(table)

    # Show training-ready segments (quality >= 4)
    high_quality = [s for s in corrections["segments"].values() if s.get("quality", 0) >= 4]
    total_duration = sum(s["end"] - s["start"] for s in high_quality)
    console.print(f"\n[bold]Training-ready (quality >= 4):[/bold] {len(high_quality)} segments, {total_duration:.1f}s ({total_duration/60:.1f} min)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TuvaLLM Transcript Corrector")
    parser.add_argument("--mode", choices=["interactive", "batch", "stats"],
                       default="interactive", help="Correction mode")
    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive_corrector()
    elif args.mode == "batch":
        batch_score_segments()
    elif args.mode == "stats":
        show_statistics()


if __name__ == "__main__":
    main()
