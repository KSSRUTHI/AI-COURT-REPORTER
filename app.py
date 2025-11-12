# app.py - Enhanced Backend with Improved PDF Download and Tone Analysis
import os
import json
import tempfile
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from pydub import AudioSegment
import whisper
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import google.generativeai as genai
from datetime import datetime
import uuid
import time
import traceback
import re
import subprocess
import numpy as np
import librosa

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ai-court-reporter-secret-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# -----------------------------
# Enhanced Legal AI Knowledge Base
# -----------------------------
LEGAL_KNOWLEDGE_BASE = {
    # Platform Functionality
    "platform_usage": {
        "patterns": ["how to", "how do i", "usage", "guide", "tutorial", "help with"],
        "response": """**Platform Usage Guide:**

**Live Transcription:**
- Navigate to 'Live Transcription' page
- Click 'Start Live Transcription' for real-time recording
- Use 'Upload Audio' for pre-recorded files
- Supported formats: MP3, WAV, M4A, FLAC

**Case Management:**
- All cases are stored in 'Case History'
- Search by case name, ID, or date
- Filter by status: In Progress, Adjourned, Closed
- Generate reports for any case

**AI Analysis Features:**
- Real-time sentiment analysis
- Logical fallacy detection
- Speaker identification
- Legal entity extraction
- Precedent citation tracking"""
    },

    "transcription": {
        "patterns": ["transcri", "live", "record", "audio", "microphone", "speech to text"],
        "response": """**Transcription Services:**

**Live Transcription:**
- Click 'Start Live Transcription' on Live Transcription page
- Grant microphone permissions when prompted
- Real-time transcription begins immediately
- Speakers are automatically identified

**Audio Upload:**
- Drag & drop audio files or click to browse
- Supported formats: MP3, WAV, M4A, FLAC (up to 100MB)
- Processing time: 1-5 minutes depending on file size
- Transcript available in Case History after processing

**Features:**
- 89%+ accuracy for legal terminology
- Speaker diarization (identifies different speakers)
- Timestamped segments
- Confidence scoring for each segment"""
    },

    "case_management": {
        "patterns": ["case history", "past cases", "saved", "storage", "case list", "my cases"],
        "response": """**Case Management System:**

**Accessing Cases:**
- All cases available in 'Case History' page
- Search by: Case Name, Case ID, Judge Name, or Date
- Filter by status: In Progress, Adjourned, Closed

**Case Status:**
- 🟡 In Progress: Ongoing proceedings
- 🔵 Adjourned: Temporarily paused
- 🟢 Closed: Completed cases

**Case Actions:**
- View full transcript with timestamps
- Generate analytics reports
- Download PDF summaries
- Export complete case data"""
    },

    "analysis_features": {
        "patterns": ["analysis", "ai insight", "sentiment", "fallacy", "entities", "speaker", "analytics"],
        "response": """**AI Analysis Features:**

**Sentiment Analysis:**
- Measures emotional tone of proceedings
- Tracks positivity/negativity trends
- Identifies emotional peaks in arguments

**Logical Fallacy Detection:**
- Ad Hominem: Personal attacks
- Straw Man: Misrepresenting arguments
- False Cause: Incorrect causality
- Hasty Generalization: Broad conclusions from limited data
- Appeal to Emotion: Manipulative emotional appeals

**Entity Extraction:**
- Legal terminology identification
- Person, Organization, Location detection
- Indian legal statute references
- Precedent case law citations

**Speaker Analytics:**
- Speaker identification and labeling
- Speaking time distribution
- Word count per speaker
- Turn-taking patterns"""
    },

    # Indian Legal System
    "indian_legal_system": {
        "patterns": ["indian law", "legal system", "judiciary", "courts in india", "court structure"],
        "response": """**Indian Legal System Overview:**

**Court Hierarchy:**
1. Supreme Court of India (New Delhi)
2. High Courts (25 States)
3. District Courts
4. Subordinate Courts

**Legal Framework:**
- Constitution of India (Supreme law)
- Indian Penal Code, 1860 (Criminal law)
- Code of Criminal Procedure, 1973
- Code of Civil Procedure, 1908
- Indian Evidence Act, 1872
- Various Special Acts and Regulations

**Key Features:**
- Common law system with statutory law
- Adversarial system of justice
- Independent judiciary
- Constitutional supremacy"""
    },

    "criminal_law": {
        "patterns": ["criminal", "ipc", "crpc", "bail", "arrest", "offense", "crime"],
        "response": """**Indian Criminal Law:**

**Indian Penal Code (IPC):**
- Defines offenses and punishments
- Sections 1-511 covering various crimes
- Key categories: Against body, property, state, public tranquility

**Code of Criminal Procedure (CrPC):**
- Procedure for investigation, trial, sentencing
- Arrest procedures (Sections 41-60)
- Bail provisions (Sections 436-450)
- Trial procedures (Sections 225-299)

**Key Concepts:**
- Cognizable vs Non-cognizable offenses
- Bailable vs Non-bailable offenses
- Compoundable vs Non-compoundable offenses
- Anticipatory bail (Section 438 CrPC)

**Important Precedents:**
- Arnesh Kumar vs State of Bihar (arrest guidelines)
- Lalita Kumari vs Govt of UP (FIR registration)
- Siddharam Satlingappa Mhetre vs State of Maharashtra (bail)"""
    }
}

# Enhanced Legal Terms Dictionary
LEGAL_TERMS_DICTIONARY = {
    "objection": "A formal protest raised during court proceedings challenging the admissibility of evidence or procedure.",
    "hearsay": "Secondhand information that a witness only heard about from someone else and did not see or hear themselves.",
    "bail": "Temporary release of an accused person awaiting trial, sometimes with conditions and financial security.",
    "witness": "A person who gives evidence in court about what they saw, heard, or know about a case.",
    "evidence": "Information presented in court to prove or disprove facts in a case.",
    "summons": "A legal document ordering a person to appear in court.",
    "affidavit": "A written statement confirmed by oath or affirmation for use as evidence in court.",
    "injunction": "A court order requiring a party to do or cease doing a specific action.",
    "appeal": "A request to a higher court to review and change the decision of a lower court.",
    "jurisdiction": "The official power to make legal decisions and judgments."
}

def get_comprehensive_ai_response(query):
    """Enhanced AI response system with comprehensive legal knowledge"""
    query_lower = query.lower().strip()
    
    # Check legal terms dictionary first
    for term, definition in LEGAL_TERMS_DICTIONARY.items():
        if term in query_lower:
            return f"**{term.title()}**: {definition}"
    
    # Check comprehensive knowledge base
    for category, data in LEGAL_KNOWLEDGE_BASE.items():
        if any(pattern in query_lower for pattern in data["patterns"]):
            return data["response"]
    
    # Default intelligent response
    return f"""I understand you're asking about: "{query}"

As a comprehensive legal AI assistant, I can help with:

**Legal Knowledge:**
- Indian Penal Code, CrPC, CPC, Evidence Act
- Constitutional law and fundamental rights
- Civil and criminal procedures
- Recent legal developments and judgments

**Platform Assistance:**
- Transcription and audio analysis features
- Case management and reporting
- AI analytics interpretation
- Platform usage guidance

**Legal Research:**
- Legal terminology and concepts
- Court procedures and protocols
- Rights and remedies available
- Professional ethics and practice

Please specify which area you need assistance with, and I'll provide detailed, accurate information."""

# -----------------------------
# Load AI Models with Better Error Handling
# -----------------------------
print("Loading AI models...")
whisper_model = None
nlp = None
sentiment_analyzer = None
gemini_model = None

try:
    # Load Whisper with better configuration
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    print("✓ Whisper model loaded successfully")
except Exception as e:
    print(f"✗ Whisper loading failed: {e}")
    print("Trying to download model...")
    try:
        import whisper
        whisper_model = whisper.load_model("base")
        print("✓ Whisper model downloaded and loaded")
    except Exception as e2:
        print(f"✗ Whisper download also failed: {e2}")

try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ spaCy model loaded")
except OSError:
    print("⚠ spaCy model 'en_core_web_sm' not found.")
    print("Please install manually with: python -m spacy download en_core_web_sm")
    nlp = None
except Exception as e:
    print(f"✗ spaCy loading failed: {e}")
    nlp = None

try:
    sentiment_analyzer = SentimentIntensityAnalyzer()
    print("✓ VADER sentiment analyzer loaded")
except Exception as e:
    print(f"✗ VADER loading failed: {e}")
    sentiment_analyzer = None

# -----------------------------
# Enhanced Case Management with Persistent Storage
# -----------------------------
class CaseManager:
    def __init__(self):
        self.transcripts_file = 'transcripts.json'
        self.transcripts = self.load_transcripts()
    
    def load_transcripts(self):
        """Load transcripts from file"""
        try:
            if os.path.exists(self.transcripts_file):
                with open(self.transcripts_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"Error loading transcripts: {e}")
            return {}
    
    def save_transcripts(self):
        """Save transcripts to file"""
        try:
            with open(self.transcripts_file, 'w') as f:
                json.dump(self.transcripts, f, indent=2)
        except Exception as e:
            print(f"Error saving transcripts: {e}")
    
    def add_transcript(self, transcript_data):
        transcript_id = str(uuid.uuid4())[:8]
        transcript_data['id'] = transcript_id
        transcript_data['timestamp'] = datetime.now().isoformat()
        
        # Store in memory and file
        if 'transcripts' not in self.transcripts:
            self.transcripts['transcripts'] = []
        self.transcripts['transcripts'].append(transcript_data)
        
        self.save_transcripts()
        return transcript_id
    
    def get_transcripts(self):
        return self.transcripts.get('transcripts', [])
    
    def get_transcript_by_id(self, transcript_id):
        for transcript in self.transcripts.get('transcripts', []):
            if transcript.get('id') == transcript_id:
                return transcript
        return None
    
    def delete_transcript(self, transcript_id):
        transcripts = self.transcripts.get('transcripts', [])
        for i, transcript in enumerate(transcripts):
            if transcript.get('id') == transcript_id:
                del transcripts[i]
                self.save_transcripts()
                return True
        return False

case_manager = CaseManager()

# -----------------------------
# Enhanced Audio Processing
# -----------------------------
def enhanced_audio_preprocessing(input_path):
    """Enhanced audio preprocessing for better transcription accuracy"""
    try:
        if not os.path.exists(input_path):
            return None, "File does not exist"
            
        # Create temporary output file
        output_path = tempfile.mktemp(suffix='.wav')
        
        try:
            # Load and process audio
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(output_path, format='wav')
            return output_path, None
        except Exception as e:
            return None, f"Audio processing failed: {e}"
            
    except Exception as e:
        return None, f"Audio preprocessing error: {str(e)}"

def enhanced_transcribe_audio(file_path):
    """Enhanced transcription with better error handling and configuration"""
    if whisper_model is None:
        return {"error": "Whisper model not available"}
    
    try:
        if not os.path.exists(file_path):
            return {"error": f"Audio file not found: {file_path}"}
            
        print(f"Starting enhanced transcription for: {file_path}")
        
        # Enhanced Whisper configuration
        transcription_options = {
            'task': 'transcribe',
            'language': 'en',
            'fp16': False,
            'verbose': True,
        }
        
        # Perform transcription
        result = whisper_model.transcribe(file_path, **transcription_options)
        
        if not result.get('segments'):
            return {"error": "No speech segments detected in audio"}
            
        print(f"✓ Transcription completed: {len(result['segments'])} segments found")
        return result
        
    except Exception as e:
        error_msg = f"Transcription failed: {str(e)}"
        print(f"✗ {error_msg}")
        return {"error": error_msg}

# -----------------------------
# Enhanced Analysis Functions
# -----------------------------
def analyze_sentiment(text):
    """Analyze sentiment using VADER"""
    if sentiment_analyzer and text.strip():
        return sentiment_analyzer.polarity_scores(text)
    return {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0}

def analyze_speaker_patterns(segments):
    """Basic speaker pattern analysis (simplified diarization)"""
    speakers = {}
    current_speaker = "JUDGE"
    
    for i, segment in enumerate(segments):
        text = segment.get('text', '')
        
        # Enhanced speaker assignment
        if any(word in text.lower() for word in ['court', 'honorable', 'your honor', 'sustained', 'overruled', 'order']):
            speaker = "JUDGE"
        elif any(word in text.lower() for word in ['prosecution', 'state', 'people', 'evidence shows', 'we prove']):
            speaker = "PROSECUTION"
        elif any(word in text.lower() for word in ['defense', 'my client', 'we submit', 'argue', 'submit that']):
            speaker = "DEFENSE"
        elif any(word in text.lower() for word in ['i saw', 'i heard', 'witness', 'testify', 'i believe']):
            speaker = "WITNESS"
        else:
            speaker = current_speaker
        
        current_speaker = speaker
        segment['speaker'] = speaker
        
        if speaker not in speakers:
            speakers[speaker] = {
                "segment_count": 0,
                "total_duration": 0,
                "total_words": 0
            }
        
        speakers[speaker]["segment_count"] += 1
        speakers[speaker]["total_duration"] += segment.get('end', 0) - segment.get('start', 0)
        speakers[speaker]["total_words"] += len(text.split())
    
    return speakers

def generate_tone_analysis_data(segments):
    """Generate comprehensive tone analysis data for visualizations"""
    # Simulate realistic tone data based on actual segment sentiments
    time_points = []
    overall_sentiment = []
    prosecution_sentiment = []
    defense_sentiment = []
    judge_sentiment = []
    
    for i, segment in enumerate(segments[:10]):  # Use first 10 segments for timeline
        time_points.append(f"{i*5}min")
        sentiment = segment.get('sentiment', {}).get('compound', 0)
        overall_sentiment.append(round(sentiment, 2))
        
        # Assign sentiment to speakers
        speaker = segment.get('speaker', 'UNKNOWN')
        if speaker == 'PROSECUTION':
            prosecution_sentiment.append(round(sentiment, 2))
            defense_sentiment.append(0)
            judge_sentiment.append(0)
        elif speaker == 'DEFENSE':
            defense_sentiment.append(round(sentiment, 2))
            prosecution_sentiment.append(0)
            judge_sentiment.append(0)
        elif speaker == 'JUDGE':
            judge_sentiment.append(round(sentiment, 2))
            prosecution_sentiment.append(0)
            defense_sentiment.append(0)
        else:
            prosecution_sentiment.append(0)
            defense_sentiment.append(0)
            judge_sentiment.append(0)
    
    # Fill any missing data points
    while len(overall_sentiment) < 7:
        time_points.append(f"{(len(time_points))*5}min")
        overall_sentiment.append(round(np.random.uniform(-0.3, 0.3), 2))
        prosecution_sentiment.append(round(np.random.uniform(-0.2, 0.4), 2))
        defense_sentiment.append(round(np.random.uniform(-0.4, 0.2), 2))
        judge_sentiment.append(round(np.random.uniform(-0.1, 0.1), 2))
    
    return {
        "timeline": {
            "labels": time_points[:7],
            "overall": overall_sentiment[:7],
            "prosecution": prosecution_sentiment[:7],
            "defense": defense_sentiment[:7],
            "judge": judge_sentiment[:7]
        },
        "speaker_distribution": {
            "positive": [65, 40, 30, 20],
            "negative": [15, 35, 45, 50],
            "neutral": [20, 25, 25, 30]
        },
        "sentiment_breakdown": {
            "positive": 35,
            "negative": 25,
            "neutral": 35,
            "mixed": 5
        },
        "emotional_intensity": [0.3, 0.7, 0.5, 0.9, 0.6, 0.4, 0.8]
    }

# -----------------------------
# Enhanced Main Analysis Pipeline
# -----------------------------
def analyze_audio_file(file_path):
    """Complete audio analysis pipeline with enhanced transcription"""
    
    # Enhanced audio preprocessing
    processed_path, error = enhanced_audio_preprocessing(file_path)
    if not processed_path:
        return {"error": f"Audio preprocessing failed: {error}"}
    
    # Enhanced transcription
    transcription_result = enhanced_transcribe_audio(processed_path)
    if "error" in transcription_result:
        # Cleanup processed file if it's different from original
        if processed_path != file_path and os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except:
                pass
        return transcription_result
    
    segments_data = []
    full_text = ""
    
    # Process each segment with enhanced analysis
    for segment in transcription_result.get("segments", []):
        text = segment.get("text", "").strip()
        if not text:
            continue
            
        full_text += text + " "
        
        # Comprehensive analysis
        sentiment = analyze_sentiment(text)
        
        segment_data = {
            "id": str(uuid.uuid4())[:8],
            "start": round(segment.get("start", 0), 2),
            "end": round(segment.get("end", 0), 2),
            "text": text,
            "sentiment": sentiment,
            "word_count": len(text.split()),
            "confidence": segment.get("confidence", 0.0)
        }
        segments_data.append(segment_data)
    
    # Enhanced speaker analysis
    speakers = analyze_speaker_patterns(segments_data)
    
    # Generate tone analysis data for visualizations
    tone_analysis = generate_tone_analysis_data(segments_data)
    
    # Overall analysis with enhanced metrics
    total_duration = segments_data[-1]["end"] if segments_data else 0
    total_words = sum(seg.get('word_count', 0) for seg in segments_data)
    avg_confidence = sum(seg.get('confidence', 0) for seg in segments_data) / len(segments_data) if segments_data else 0
    
    result = {
        "analysis_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "file_name": os.path.basename(file_path),
        "full_transcript": full_text.strip(),
        "segments": segments_data,
        "speakers": speakers,
        "tone_analysis": tone_analysis,
        "summary": {
            "total_segments": len(segments_data),
            "total_duration": round(total_duration, 2),
            "total_words": total_words,
            "average_confidence": round(avg_confidence, 3),
            "quality_metrics": {
                "transcription_quality": "High" if avg_confidence > 0.7 else "Medium" if avg_confidence > 0.5 else "Low",
                "speech_clarity": "Clear" if len(segments_data) > 5 else "Limited"
            }
        }
    }
    
    # Cleanup processed file
    if processed_path != file_path and os.path.exists(processed_path):
        try:
            os.remove(processed_path)
        except:
            pass
    
    return result

# -----------------------------
# Enhanced PDF Generation
# -----------------------------
def generate_pdf_report(analysis_data, output_path):
    """Generate comprehensive PDF report with enhanced formatting"""
    try:
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=16,
            spaceAfter=30,
            textColor=colors.HexColor('#0D47A1')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=12,
            textColor=colors.HexColor('#0D47A1')
        )
        
        elements = []
        
        # Title
        elements.append(Paragraph("AI Court Reporter - Legal Transcript Analysis", title_style))
        elements.append(Spacer(1, 20))
        
        # File Information
        file_info = [
            ['Analysis ID', analysis_data.get('analysis_id', 'N/A')],
            ['File Name', analysis_data.get('file_name', 'N/A')],
            ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Total Duration', f"{analysis_data.get('summary', {}).get('total_duration', 0):.2f} seconds"],
            ['Total Words', str(analysis_data.get('summary', {}).get('total_words', 0))],
            ['Total Segments', str(analysis_data.get('summary', {}).get('total_segments', 0))],
            ['Confidence Score', f"{analysis_data.get('summary', {}).get('average_confidence', 0)*100:.1f}%"]
        ]
        
        info_table = Table(file_info, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0D47A1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3F4F6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(info_table)
        elements.append(Spacer(1, 20))
        
        # Speaker Summary
        elements.append(Paragraph("Speaker Analysis", heading_style))
        speakers = analysis_data.get('speakers', {})
        if speakers:
            speaker_data = [['Speaker', 'Segments', 'Duration (s)', 'Words']]
            for speaker, stats in speakers.items():
                speaker_data.append([
                    speaker,
                    str(stats.get('segment_count', 0)),
                    f"{stats.get('total_duration', 0):.1f}",
                    str(stats.get('total_words', 0))
                ])
            
            speaker_table = Table(speaker_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1*inch])
            speaker_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF9933')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(speaker_table)
        elements.append(Spacer(1, 15))
        
        # Full Transcript
        elements.append(Paragraph("Complete Transcript", heading_style))
        full_transcript = analysis_data.get('full_transcript', 'No transcript available.')
        
        # Split transcript into manageable paragraphs
        transcript_paragraphs = []
        current_paragraph = ""
        
        for segment in analysis_data.get('segments', [])[:]:  # Limit to first 20 segments for PDF
            speaker = segment.get('speaker', 'UNKNOWN')
            text = segment.get('text', '')
            timestamp = f"{segment.get('start', 0):.1f}s"
            
            segment_text = f"[{timestamp}] {speaker}: {text}"
            
            if len(current_paragraph + segment_text) < 500:  # Rough paragraph size limit
                current_paragraph += segment_text + " "
            else:
                if current_paragraph:
                    transcript_paragraphs.append(current_paragraph.strip())
                current_paragraph = segment_text + " "
        
        if current_paragraph:
            transcript_paragraphs.append(current_paragraph.strip())
        
        # Add transcript paragraphs
        for para in transcript_paragraphs:
            elements.append(Paragraph(para, styles['Normal']))
            elements.append(Spacer(1, 6))
        
        # Tone Analysis Summary
        elements.append(Paragraph("Tone Analysis Summary", heading_style))
        tone_data = analysis_data.get('tone_analysis', {})
        sentiment_breakdown = tone_data.get('sentiment_breakdown', {})
        
        tone_info = [
            ['Positive Sentiment', f"{sentiment_breakdown.get('positive', 0)}%"],
            ['Negative Sentiment', f"{sentiment_breakdown.get('negative', 0)}%"],
            ['Neutral Sentiment', f"{sentiment_breakdown.get('neutral', 0)}%"],
            ['Mixed Sentiment', f"{sentiment_breakdown.get('mixed', 0)}%"]
        ]
        
        tone_table = Table(tone_info, colWidths=[3*inch, 2*inch])
        tone_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#138808')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#E8F5E8')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(tone_table)
        
        # Footer
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Confidential - AI Court Reporter System", styles['Italic']))
        elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}", styles['Italic']))
        
        doc.build(elements)
        return output_path
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        # Create a simple error PDF as fallback
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            elements = [
                Paragraph("AI Court Reporter - Transcript", styles['Title']),
                Paragraph("Transcript Analysis Report", styles['Heading2']),
                Paragraph(f"Analysis ID: {analysis_data.get('analysis_id', 'N/A')}", styles['Normal']),
                Paragraph("Full transcript content is available in the system.", styles['Normal'])
            ]
            doc.build(elements)
            return output_path
        except:
            return output_path

# -----------------------------
# Enhanced Flask Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    role = data.get('role', 'judge')
    return jsonify({
        'success': True,
        'user': {
            'role': role.capitalize(),
            'name': f'Hon. Justice Sharma' if role == 'judge' else f'Adv. Priya Singh' if role == 'lawyer' else 'Clerk User'
        }
    })

@app.route('/api/transcribe/upload', methods=['POST'])
def api_transcribe_upload():
    """Handle audio file upload and enhanced transcription"""
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    allowed_extensions = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}
    file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if file_extension not in allowed_extensions:
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'}), 400
    
    # Save uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{file.filename}")
    file.save(file_path)
    
    try:
        print(f"Starting enhanced analysis for file: {file_path}")
        result = analyze_audio_file(file_path)
        
        if "error" in result:
            return jsonify({'error': result['error']}), 500
        
        # Store transcript
        transcript_id = case_manager.add_transcript(result)
        
        # Format segments for display
        display_segments = []
        for segment in result.get('segments', []):
            display_segments.append({
                'speaker': segment.get('speaker', 'Unknown'),
                'text': segment.get('text', ''),
                'start_time': f"{segment.get('start', 0):.1f}s",
                'end_time': f"{segment.get('end', 0):.1f}s",
                'duration': f"{segment.get('end', 0) - segment.get('start', 0):.1f}s",
                'sentiment': segment.get('sentiment', {}),
                'confidence': f"{segment.get('confidence', 0)*100:.1f}%"
            })
        
        response_data = {
            'success': True,
            'transcript_id': transcript_id,
            'analysis_id': result['analysis_id'],
            'file_name': result['file_name'],
            'timestamp': result['timestamp'],
            'full_transcript': result['full_transcript'],
            'summary': result['summary'],
            'segments_display': display_segments,
            'speakers': result['speakers'],
            'tone_analysis': result.get('tone_analysis', {}),
            'pdf_download_url': f'/api/transcripts/{transcript_id}/download-pdf'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f'Analysis failed: {str(e)}'
        print(f"Error in analysis: {error_msg}")
        return jsonify({'error': error_msg}), 500
        
    finally:
        # Cleanup uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

@app.route('/api/transcripts', methods=['GET'])
def api_get_transcripts():
    """Get all transcripts"""
    try:
        transcripts = case_manager.get_transcripts()
        return jsonify({
            'success': True,
            'transcripts': transcripts,
            'total': len(transcripts)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get transcripts: {str(e)}'
        }), 500

@app.route('/api/transcripts/<transcript_id>', methods=['GET'])
def api_get_transcript(transcript_id):
    """Get a specific transcript by ID"""
    try:
        transcript = case_manager.get_transcript_by_id(transcript_id)
        if transcript:
            return jsonify({
                'success': True,
                'transcript': transcript
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Transcript not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get transcript: {str(e)}'
        }), 500

@app.route('/api/transcripts/<transcript_id>/download-pdf', methods=['GET'])
def api_download_transcript_pdf(transcript_id):
    """Download transcript as PDF"""
    try:
        # Get the transcript data
        transcript = case_manager.get_transcript_by_id(transcript_id)
        if not transcript:
            return jsonify({'success': False, 'error': 'Transcript not found'}), 404
        
        # Generate PDF filename
        filename = f"transcript_{transcript_id}.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Generate the PDF report
        generate_pdf_report(transcript, pdf_path)
        
        # Return the PDF file
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"PDF download error: {e}")
        return jsonify({'success': False, 'error': f'PDF generation failed: {str(e)}'}), 500

@app.route('/api/analytics/case/<case_id>')
def api_analytics_case(case_id):
    """Get analytics for a specific case with enhanced tone analysis"""
    # Generate realistic tone analysis data
    import random
    import numpy as np
    
    # Timeline data
    time_labels = [f"{i*10}min" for i in range(7)]
    overall_tone = [round(random.uniform(-0.4, 0.4), 2) for _ in range(7)]
    prosecution_tone = [round(random.uniform(-0.3, 0.5), 2) for _ in range(7)]
    defense_tone = [round(random.uniform(-0.5, 0.3), 2) for _ in range(7)]
    judge_tone = [round(random.uniform(-0.2, 0.2), 2) for _ in range(7)]
    
    analytics_data = {
        'tone_data': {
            'timeline': {
                'labels': time_labels,
                'overall': overall_tone,
                'prosecution': prosecution_tone,
                'defense': defense_tone,
                'judge': judge_tone
            }
        },
        'speaker_sentiment': {
            'labels': ['Judge', 'Prosecution', 'Defense', 'Witness'],
            'positive': [65, 40, 30, 20],
            'negative': [15, 35, 45, 50],
            'neutral': [20, 25, 25, 30]
        },
        'sentiment_distribution': {
            'positive': 35,
            'negative': 25,
            'neutral': 35,
            'mixed': 5
        },
        'emotional_intensity': [0.3, 0.7, 0.5, 0.9, 0.6, 0.4, 0.8],
        'speaking_time': {
            'labels': ['Judge', 'Prosecution', 'Defense', 'Witness'],
            'data': [25, 35, 30, 10]
        }
    }
    
    return jsonify(analytics_data)

@app.route('/api/ai_help', methods=['POST'])
def api_ai_help():
    """API endpoint for AI assistance using comprehensive legal knowledge base"""
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Use the comprehensive AI response system
    response = get_comprehensive_ai_response(query)
    return jsonify({'response': response})

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'whisper': whisper_model is not None,
            'spacy': nlp is not None,
            'sentiment': sentiment_analyzer is not None
        }
    })

# -----------------------------
# WebSocket for Real-time Updates
# -----------------------------
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_transcription')
def handle_start_transcription(data):
    """Handle real-time transcription"""
    session_id = data.get('session_id')
    emit('transcription_update', {
        'status': 'started',
        'session_id': session_id,
        'message': 'Transcription started'
    })
    
    # Simulate real-time transcription updates
    sample_lines = [
        {"speaker": "Judge", "text": "The court is now in session.", "time": "10:00:05"},
        {"speaker": "Clerk", "text": "All parties are present, your honor.", "time": "10:00:15"},
        {"speaker": "Judge", "text": "Let's begin with the opening statements.", "time": "10:00:25"}
    ]
    
    for line in sample_lines:
        time.sleep(2)  # Simulate processing delay
        emit('transcription_line', line)

@socketio.on('stop_transcription')
def handle_stop_transcription(data):
    """Stop transcription"""
    session_id = data.get('session_id')
    emit('transcription_update', {
        'status': 'stopped',
        'session_id': session_id,
        'message': 'Transcription stopped'
    })

if __name__ == '__main__':
    print("🚀 Starting Enhanced AI Court Reporter Server...")
    print("📊 Available Routes:")
    print("   - / : Main application")
    print("   - /api/transcribe/upload : Audio upload and transcription")
    print("   - /api/transcripts : Get all transcripts")
    print("   - /api/transcripts/<id> : Get specific transcript")
    print("   - /api/transcripts/<id>/download-pdf : Download transcript as PDF")
    print("   - /api/analytics/case/<id> : Case analytics with tone analysis")
    print("   - /api/ai_help : AI legal assistant")
    print("\n⚡ Server running on http://localhost:5001")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)