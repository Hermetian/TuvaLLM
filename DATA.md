# TuvaLLM Data Inventory

## Overview
This dataset contains **~47 hours** of Tuvaluan parliamentary session recordings with matching transcripts, plus extensive reference documents.

## Parliament Session Recordings

### December 2024 Session (Video)
Primary high-quality MP4 recordings from December 2024 parliament session.

| File | Size | Session |
|------|------|---------|
| Parliament Opening Session Day 1 - 09-12-24-004.mp4 | 2.4 GB | Day 1 Opening |
| Parliament Opening Session Day 2 - 10-12-24-003.mp4 | 3.4 GB | Day 2 Opening |
| Parliament Afternoon Session Day 2 - 10-12-24-002.mp4 | 3.4 GB | Day 2 Afternoon |
| Parliament Afternoon Session Day 3 - 11-12-24-007.mp4 | 3.4 GB | Day 3 Afternoon |
| Parliament Opening Session 1 Day 3 - 11-12-24.mp4 | 1.4 GB | Day 3 Opening 1 |
| Parliament Opening Session 2 Day 3 - 11-12-24.mp4 | 1.7 GB | Day 3 Opening 2 |
| Parliament Morning Session - Last Day 4 12-12-24.mp4 | 1.4 GB | Day 4 Morning |
| Parliament Afternoon Session Day 1 - 09-12-24.mp4 | 742 MB | Day 1 Afternoon |

### June 2024 Session (Audio)
MP3 recordings from June 2024 parliament session.

| File | Size | Session |
|------|------|---------|
| Parliament Day 1 Take 1 (24-06-24).mp3 | 223 MB | Day 1 |
| Parliament opening 2024 24-06-24.mp3 | 66 MB | Opening |
| Parliament Day 2 Take 1.mp3 | 189 MB | Day 2 |
| Parliament Day 2 Take 2.mp3 | 54 MB | Day 2 |
| Parliament Day 3 Take 1.mp3 | 153 MB | Day 3 |
| Parliament Day 3 Take 2.mp3 | 210 MB | Day 3 |
| Parliament Day 4 Take 1.mp3 | 103 MB | Day 4 |
| Parliament Day 4 Take 2.mp3 | 78 MB | Day 4 |
| Parliament Day 4 Take 3.mp3 | 159 MB | Day 4 |
| Parliament Day 5 Take 1.mp3 | 186 MB | Day 5 |
| Parliament Day 5 Take 2.mp3 | 67 MB | Day 5 |

## Transcripts (Training Data)
Official parliament transcripts in Tuvaluan - **critical for training**.

| File | Size | Content |
|------|------|---------|
| FINALPalamene June Parliament session 2024.docx | 504 KB | June 2024 session transcript |
| Lipoti Fono Palamene Tesema 2024.docx | 308 KB | December 2024 session transcript |

## Reference Documents

### Linguistic Resources
| File | Size | Content |
|------|------|---------|
| DICTIONARY Tuv_Palagi (2).PDF.pdf | 4.2 MB | **Tuvaluan-English Dictionary** |
| Tuvalu-All-Bible.pdf | 12 MB | Complete Tuvaluan Bible |
| 2993349-tuvalu-language.doc | 95 KB | Tuvaluan language reference |

### Government Documents (Tuvaluan text)
| File | Size | Content |
|------|------|---------|
| Tuv-Vers-Tuvalu-Citizen-Budget-Guide_2025-2026.pdf | 2.4 MB | Budget guide (Tuvaluan) |
| Eng-Vers-Tuvalu-Citizen-Budget-Guide_2025-2026.pdf | 2.4 MB | Budget guide (English parallel) |
| Final 47th Independence Address PM Teo.pdf | 264 KB | PM speech |
| Lauga ote 47 Tutokotasi 2025.pdf | 288 KB | Independence Day speech (Tuvaluan) |

### Historical Documents
| File | Size | Content |
|------|------|---------|
| Tuvalu - News Sheets Part One (1).pdf | 42 MB | Historical news in Tuvaluan |
| Tuvalu News Sheets Part 2 (1).pdf | 24 MB | Historical news in Tuvaluan |
| Tuvalu News Sheets 66-99 (1).pdf | 121 MB | Historical news in Tuvaluan |

## Data Location
```
data/
├── raw/
│   ├── parliament_video/     # December 2024 MP4s (12.6 GB)
│   └── research_materials/   # Extracted ZIP contents (7.4 GB extracted)
│       └── Research Materials/
│           └── Parliament Session Recordings/
│               ├── *.docx                    # Transcripts
│               └── Parliament Session Recording/
│                   ├── June Parliament Session/  # MP3s
│                   └── Parliament Session December 2024/  # MP4s
├── reference/                # PDFs and DOCs (3 files)
└── archive/grn/              # Old GRN data (archived)
```

## Training Data Quality

| Source | Audio | Transcript | Quality |
|--------|-------|------------|---------|
| December 2024 Parliament | MP4 video | DOCX official | ⭐⭐⭐⭐⭐ |
| June 2024 Parliament | MP3 audio | DOCX official | ⭐⭐⭐⭐⭐ |

**Total estimated audio**: ~47 hours
**Transcript availability**: Official government transcripts for both sessions
