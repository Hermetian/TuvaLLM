UDHR Tuvaluan Test Sample
==========================

This directory contains the ground truth benchmark for TuvaLLM ASR evaluation.

Contents:
- reference.txt    : Ground truth transcript (System 4 orthography)
- metadata.json    : Sample metadata including speaker and source info
- audio.mp3        : Audio file (MUST BE DOWNLOADED SEPARATELY)

To download the audio file:
1. Visit: https://www.omniglot.com/soundfiles/udhr/udhr_tuvaluan.mp3
2. Save the file as "audio.mp3" in this directory
3. Run: python scripts/benchmark_udhr.py

Or use curl/wget:
  curl -o data/test/udhr_sample/audio.mp3 https://www.omniglot.com/soundfiles/udhr/udhr_tuvaluan.mp3

Speaker: Tauina Lesaa
Source: Omniglot (https://omniglot.com)

Text (System 4 - with macrons):
E fa'nau mai a tino katoa i te saolotoga kae e 'pau telotolo tulaga
fakaaloalogina mo telotolo aia. Ne tuku atu ki a latou a te mafaufau
mo te loto lagona, tela la, e 'tau o ga'lue fakatasi latou e pela me ne taina.
