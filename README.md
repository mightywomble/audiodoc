# AudioDoc - Audio to Text Converter

A responsive web application that converts audio files to text using speech recognition technology. Works seamlessly on desktop, tablet, and mobile devices.

## Features

- 🎵 **Audio Upload**: Support for multiple audio formats (MP3, WAV, M4A, OGG, etc.)
- 🎤 **Speech Recognition**: Converts audio to text using OpenAI Whisper (offline)
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- 💾 **Word Export**: Save transcriptions as Word documents (.docx)
- 🔗 **Share Functionality**: Share text via native sharing or copy to clipboard
- ✏️ **Editable Results**: Edit transcribed text before saving or sharing
- 🎨 **Modern UI**: Beautiful gradient design with smooth animations

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /Users/david/code/audiodoc
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install system dependencies** (for audio processing):
   
   **On macOS**:
   ```bash
   brew install ffmpeg
   ```
   
   **On Ubuntu/Debian**:
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5001
   ```

3. **Upload an audio file** by:
   - Clicking the upload area and selecting a file
   - Dragging and dropping a file onto the upload area

4. **Wait for processing** - the app will convert your audio to text

5. **Review and edit** the transcribed text if needed

6. **Export or share**:
   - Click "Save as Word" to download a .docx file
   - Click "Share Text" to share via native sharing or copy to clipboard

## Supported Audio Formats

- MP3
- WAV
- M4A
- OGG
- FLAC
- And many other formats supported by FFmpeg

## Technical Details

### Backend
- **Flask**: Web framework
- **OpenAI Whisper**: Advanced speech recognition (offline)
- **python-docx**: Word document generation
- **PyTorch**: Machine learning framework for Whisper

### Frontend
- **Responsive HTML/CSS**: Works on all device sizes
- **Vanilla JavaScript**: No external dependencies
- **Modern UI**: CSS gradients and animations
- **Progressive Enhancement**: Fallbacks for older browsers

### File Processing
1. Audio files are temporarily saved and processed
2. Whisper directly processes various audio formats
3. AI-powered transcription converts audio to text
4. Temporary files are automatically cleaned up

## API Endpoints

- `GET /`: Main application page
- `POST /upload`: Upload and process audio file
- `POST /download_word`: Generate and download Word document
- `GET /health`: Health check endpoint

## Security Features

- File size limits (50MB maximum)
- File type validation
- Temporary file cleanup
- Secure file handling

## Error Handling

The application includes comprehensive error handling for:
- Invalid file types
- Network errors
- Speech recognition failures
- File processing errors

## Mobile Optimization

- Touch-friendly interface
- Optimized for various screen sizes
- Native mobile sharing support
- Responsive typography and spacing

## Browser Compatibility

- Modern browsers (Chrome, Firefox, Safari, Edge)
- Mobile browsers (iOS Safari, Chrome Mobile)
- Progressive enhancement for older browsers

## Troubleshooting

**Audio not recognized**:
- Ensure audio quality is good
- Check if there's clear speech in the audio
- Try with a different audio file

**Installation issues**:
- Make sure FFmpeg is installed
- Verify Python 3.8+ is being used
- Ensure sufficient disk space for Whisper models

**Performance**:
- First run may be slower as Whisper downloads models
- Larger files take longer to process
- No internet connection required after initial setup
- Consider breaking very long audio files into smaller chunks
