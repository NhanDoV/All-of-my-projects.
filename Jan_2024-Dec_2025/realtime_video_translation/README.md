# Language-Learning Video Platform  
A streamlined system for processing videos, generating subtitles with AI, and enabling interactive language learning.

---

## System Overview

This platform processes uploaded videos using an AI-powered subtitle pipeline and provides users with an interactive player for language learning features such as dual subtitles, vocabulary lookups, and sentence-based navigation.

---

## Admin Workflow

### 1. Upload Video
Admins upload raw video files to cloud storage (e.g., S3, GCS).

### 2. AI Processing Pipeline
- **Audio Extraction** – Separate audio from video.  
- **Speech-to-Text** – Generate accurate subtitles with timestamps using AI (WhisperX, Google STT).  
- **Semantic Subtitle Chunking** – Split subtitles into meaningful sentences instead of fixed time windows.  
- **Translation** – Translate subtitles into multiple languages.  
- **Timestamp Sync** – Keep original subtitle timing across translations.

### 3. Manual Editing
Admins can refine:
- Subtitle content  
- Subtitle timing  
- Translations  
- Metadata (title, description, cover image, categories, difficulty level)

### 4. Publish
Finalized videos appear in the video library for users.

---

## User Workflow

### 1. Discover Videos
Users filter by:
- Original language  
- Category/genre  
- Difficulty level  
- Length  

### 2. Watch & Learn
- Select primary & secondary subtitle languages  
- View dual subtitles  
- Hover on subtitle words for:
  - Definitions  
  - Phonetics  
  - Save-to-vocabulary  

- Keyboard controls:
  - Previous/next subtitle sentence  
  - Play/pause  

- Adjust playback speed  
- View cultural/slang notes

### 3. Vocabulary Review
Saved words appear in the user's personal dictionary with:
- Flashcards  
- Quizzes  
- Review tools  

---

## Core Platform Features

### Video Player
- Built on Video.js, Plyr, or similar  
- Sentence-level seeking using subtitle cues  
- Dual subtitle display  
- IPA phonetic subtitles (optional)

### Subtitle & AI Tools
- Drag & drop video upload  
- AI subtitle segmentation  
- AI translations  
- Professional subtitle timeline editor  

### Learning Features
- Difficulty filtering (A1–C2)  
- Popup dictionary on word hover  
- Save vocabulary with one click  
- Future upgrades:
  - Spaced repetition flashcards  
  - AI-powered clip generator for speech shadowing  

---

## System Requirements

- Cloud storage service  
- AI speech-to-text models  
- Translation APIs  
- Backend for processing & user management  
- Database for users, subtitles, and vocabulary data