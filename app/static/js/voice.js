/**
 * VoiceController
 * Handles Speech-to-Text (Microphone input) and Text-to-Speech (Audio output)
 * with full Indian Regional Language support (Tamil, Hindi, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, English).
 */
class VoiceController {
    constructor() {
        this.recognition = null;
        this.isListening = false;
        this.autoSpeak = true;
        this.currentAudio = null;
        this.currentLanguage = "en";
        this.speechLocale = "en-IN";
        this.preferredVoice = "en-IN-NeerjaNeural";

        // Auto-detect mode: browsers' own speech recognition must be told the language in advance,
        // so instead we record the audio and the server (Whisper) works out which language was spoken.
        this.autoDetect = false;
        this.recorder = null;
        this.recordingCleanup = null;

        this.onSpeechResult = null;      // (text, detectedLanguageOrNull)
        this.onSpeechInterim = null;
        this.onListeningChange = null;
        this.onTranscribing = null;      // (busy) while the recording is being transcribed
        this.onVoiceError = null;        // (message)

        this.initRecognition();
    }

    initRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.warn("Web Speech API (SpeechRecognition) is not supported in this browser.");
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.continuous = false;
        this.recognition.interimResults = true;
        this.recognition.lang = this.speechLocale;

        this.recognition.onstart = () => {
            this.isListening = true;
            if (this.onListeningChange) this.onListeningChange(true);
        };

        this.recognition.onresult = (event) => {
            let interimTranscript = "";
            let finalTranscript = "";

            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }

            if (interimTranscript && this.onSpeechInterim) {
                this.onSpeechInterim(interimTranscript);
            }

            if (finalTranscript && this.onSpeechResult) {
                this.onSpeechResult(finalTranscript.trim());
            }
        };

        this.recognition.onerror = (event) => {
            console.warn("Speech recognition error:", event.error);
            this.isListening = false;
            if (this.onListeningChange) this.onListeningChange(false);
        };

        this.recognition.onend = () => {
            this.isListening = false;
            if (this.onListeningChange) this.onListeningChange(false);
        };
    }

    setLanguage(langCode, speechLocale, defaultVoice) {
        this.currentLanguage = langCode || "en";
        this.speechLocale = speechLocale || "en-IN";
        if (defaultVoice) {
            this.preferredVoice = defaultVoice;
        }
        if (this.recognition) {
            this.recognition.lang = this.speechLocale;
            console.log(`Speech recognition language updated to: ${this.speechLocale}`);
        }
    }

    startListening() {
        if (this.autoDetect) {
            this.startRecording();
            return;
        }
        if (!this.recognition) {
            alert("Speech recognition is not supported in this browser. Please use Chrome, Edge, Safari, or Brave.");
            return;
        }
        this.stopSpeaking();

        // Make sure recognition language matches current locale
        if (this.recognition.lang !== this.speechLocale) {
            this.recognition.lang = this.speechLocale;
        }

        try {
            this.recognition.start();
        } catch (e) {
            console.debug("Recognition already active or restarting:", e);
        }
    }

    stopListening() {
        if (this.recorder && this.recorder.state === "recording") {
            this.recorder.stop();
            return;
        }
        if (this.recognition && this.isListening) {
            this.recognition.stop();
        }
    }

    /**
     * Record from the microphone until the speaker pauses, then send the audio to the server,
     * which returns the transcript and the language it was spoken in.
     */
    async startRecording() {
        if (!navigator.mediaDevices || !window.MediaRecorder) {
            this._voiceError("Voice recording isn't supported in this browser. Please use Chrome, Edge or Safari.");
            return;
        }
        this.stopSpeaking();

        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (e) {
            this._voiceError("Microphone access was blocked. Allow the microphone for this site and try again.");
            return;
        }

        const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"]
            .find((t) => MediaRecorder.isTypeSupported(t)) || "";
        const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
        const chunks = [];
        let heardSpeech = false;

        // Simple voice-activity detection: stop ~1.5s after the speaker goes quiet
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 2048;
        audioCtx.createMediaStreamSource(stream).connect(analyser);
        const samples = new Float32Array(analyser.fftSize);
        const startedAt = Date.now();
        let lastVoiceAt = Date.now();

        const monitor = setInterval(() => {
            analyser.getFloatTimeDomainData(samples);
            let sum = 0;
            for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
            const rms = Math.sqrt(sum / samples.length);
            const now = Date.now();
            if (rms > 0.02) {
                heardSpeech = true;
                lastVoiceAt = now;
            }
            const silentFor = now - lastVoiceAt;
            if ((heardSpeech && silentFor > 1500) || (!heardSpeech && now - startedAt > 7000) || now - startedAt > 15000) {
                if (recorder.state === "recording") recorder.stop();
            }
        }, 100);

        this.recordingCleanup = () => {
            clearInterval(monitor);
            stream.getTracks().forEach((t) => t.stop());
            audioCtx.close().catch(() => {});
            this.recordingCleanup = null;
        };

        recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
        recorder.onstop = async () => {
            if (this.recordingCleanup) this.recordingCleanup();
            this.recorder = null;
            this.isListening = false;
            if (this.onListeningChange) this.onListeningChange(false);

            if (!heardSpeech || chunks.length === 0) {
                this._voiceError("I didn't hear anything. Tap the mic and speak again.");
                return;
            }
            const blob = new Blob(chunks, { type: recorder.mimeType || mimeType || "audio/webm" });
            if (this.onTranscribing) this.onTranscribing(true);
            try {
                const resp = await fetch("/api/voice/transcribe", {
                    method: "POST",
                    headers: { "Content-Type": blob.type || "audio/webm" },
                    body: blob,
                });
                const data = await resp.json();
                if (this.onTranscribing) this.onTranscribing(false);
                if (data.text) {
                    if (this.onSpeechResult) this.onSpeechResult(data.text, data.language || null);
                } else {
                    this._voiceError(data.error || data.detail || "Sorry, I couldn't understand that. Please try again.");
                }
            } catch (err) {
                if (this.onTranscribing) this.onTranscribing(false);
                this._voiceError("Couldn't reach the server to transcribe your voice.");
            }
        };

        this.recorder = recorder;
        recorder.start();
        this.isListening = true;
        if (this.onListeningChange) this.onListeningChange(true);
    }

    _voiceError(message) {
        console.warn(message);
        if (this.onVoiceError) this.onVoiceError(message);
    }

    toggleListening() {
        if (this.isListening) {
            this.stopListening();
        } else {
            this.startListening();
        }
    }

    /**
     * Play spoken text via server neural audio URL or browser SpeechSynthesis
     */
    async speak(text, audioUrl = null, locale = null) {
        if (!this.autoSpeak) return;

        this.stopSpeaking();

        // 1. If backend provided high quality neural audio URL, play it
        if (audioUrl) {
            try {
                this.currentAudio = new Audio(audioUrl);
                await this.currentAudio.play();
                return;
            } catch (err) {
                console.warn("Could not play neural audio URL, falling back to browser synthesis:", err);
            }
        }

        // 2. Fallback to Browser SpeechSynthesis with regional locale
        if ("speechSynthesis" in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.95;
            utterance.pitch = 1.0;
            utterance.lang = locale || this.speechLocale || "en-IN";

            // Try to find a matching voice in the browser for this language
            const voices = window.speechSynthesis.getVoices();
            if (voices && voices.length > 0) {
                const targetPrefix = (locale || this.speechLocale).split("-")[0];
                const match = voices.find(v => v.lang.startsWith(targetPrefix));
                if (match) {
                    utterance.voice = match;
                }
            }

            window.speechSynthesis.speak(utterance);
        }
    }

    stopSpeaking() {
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            this.currentAudio = null;
        }
        if ("speechSynthesis" in window) {
            window.speechSynthesis.cancel();
        }
    }
}

window.voiceController = new VoiceController();
