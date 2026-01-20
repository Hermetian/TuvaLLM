"""Tests for TuvaLLM core."""

from tuvallm import TuvaLLM


def test_init():
    llm = TuvaLLM()
    assert llm is not None
