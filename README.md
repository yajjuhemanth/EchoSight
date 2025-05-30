# ECHOSIGHT

A multimodal Flask application that provides real-time speech recognition and image captioning capabilities using AI models.

## Features

### Hearing Mode
- Real-time speech recognition using Google's speech recognition API
- Continuous listening with adjustable parameters
- Transcript display and management
- User-friendly interface with status indicators

### Visual Mode
- Image captioning using BLIP (Bootstrapping Language-Image Pre-training) model
- Two operation modes:
  - Webcam feed with continuous or on-demand captioning
  - Image upload for single-image captioning
- Text-to-speech output of generated captions

## Installation

### Prerequisites
- Python 3.7+
- Webcam (for visual mode with webcam)
- Microphone (for hearing mode)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/echosight.git
   cd echosight
   ```

2. Create and activate a virtual environment (recommended):
   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Models:
   - The BLIP image captioning model (`Salesforce/blip-image-captioning-base`) will be automatically downloaded on first use of the visual mode
 

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Navigate to `http://localhost:5000` in your web browser

3. Choose between Hearing Mode and Visual Mode:
   - **Hearing Mode**: Transcribes speech in real-time
   - **Visual Mode**: Captions images from webcam or uploads

### First Run
- When you first use the Visual Mode, the application will download the BLIP model (~1GB)
- This download only happens once, after which the model is cached locally
- The download might take several minutes depending on your internet connection
- Progress will be displayed in the terminal

## Technical Details

- **Flask**: Web application framework
- **SpeechRecognition**: Library for performing speech recognition
- **PyTorch & Transformers**: For BLIP image captioning model
- **OpenCV**: For webcam integration and image processing
- **Pyttsx3**: For text-to-speech functionality

## License

MIT License - See LICENSE file for details

## Acknowledgments

- [BLIP Model](https://github.com/salesforce/BLIP) by Salesforce Research
- [SpeechRecognition library](https://github.com/Uberi/speech_recognition)
- [Flask framework](https://flask.palletsprojects.com/) 
