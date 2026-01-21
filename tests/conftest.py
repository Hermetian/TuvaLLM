"""
TuvaLLM Test Configuration
Pytest fixtures for test suite.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np

# Add scripts directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def data_dir() -> Path:
    """Return the data directory."""
    return PROJECT_ROOT / "data"


@pytest.fixture
def sample_system4_text() -> str:
    """Sample text in System 4 (with macrons)."""
    return "E fā'nau mai a tino katoa i te saolotoga kae e 'pau telotolo tūlaga fakaaloalogina mo telotolo aiā."


@pytest.fixture
def sample_system6_text() -> str:
    """Sample text in System 6 (without macrons)."""
    return "E fa'nau mai a tino katoa i te saolotoga kae e 'pau telotolo tulaga fakaaloalogina mo telotolo aia."


@pytest.fixture
def sample_corrections() -> Dict[str, Any]:
    """Sample corrections data structure."""
    return {
        "segments": {
            "0.0-5.0": {
                "original_mms": "e fanau mai a tino katoa",
                "corrected_text": "E fā'nau mai a tino katoa",
                "start": 0.0,
                "end": 5.0,
                "quality": 4
            },
            "5.0-10.0": {
                "original_mms": "i te saolotoga",
                "corrected_text": "i te saolotoga kae",
                "start": 5.0,
                "end": 10.0,
                "quality": 5
            }
        },
        "vocabulary": {
            "fā'nau": {"word": "fā'nau", "meaning": "born", "pos": "verb"},
            "tūlaga": {"word": "tūlaga", "meaning": "status/position", "pos": "noun"}
        },
        "metadata": {
            "created": "2024-01-01T00:00:00",
            "last_modified": "2024-01-01T00:00:00",
            "audio_file": "test_audio.mp3"
        }
    }


@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # Generate 1 second of silence at 16kHz
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio = np.zeros(samples, dtype=np.float32)

        # Add some noise to make it non-empty
        audio += np.random.randn(samples).astype(np.float32) * 0.01

        sf.write(tmp.name, audio, sample_rate)
        yield tmp.name

    # Cleanup
    os.unlink(tmp.name)


@pytest.fixture
def temp_corrections_file(sample_corrections):
    """Create a temporary corrections file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as tmp:
        json.dump(sample_corrections, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        yield tmp.name

    os.unlink(tmp.name)


@pytest.fixture
def udhr_reference_texts() -> Dict[str, str]:
    """UDHR Article 1 reference texts in both orthographic systems."""
    return {
        "system4": "E fā'nau mai a tino katoa i te saolotoga kae e 'pau telotolo tūlaga fakaaloalogina mo telotolo aiā. Ne tuku atu ki a lātou a te mafaufau mo te loto lagona, tēlā lā, e 'tau o gā'lue fakatasi lātou e pēlā me ne taina.",
        "system6": "E fa'nau mai a tino katoa i te saolotoga kae e 'pau telotolo tulaga fakaaloalogina mo telotolo aia. Ne tuku atu ki a latou a te mafaufau mo te loto lagona, tela la, e 'tau o ga'lue fakatasi latou e pela me ne taina."
    }


@pytest.fixture
def word_dictionary() -> Dict[str, str]:
    """Sample word dictionary for System 6 to System 4 conversion."""
    return {
        "fanau": "fā'nau",
        "tulaga": "tūlaga",
        "latou": "lātou",
        "aia": "aiā",
        "tela": "tēlā",
        "galue": "gā'lue",
        "pela": "pēlā"
    }
