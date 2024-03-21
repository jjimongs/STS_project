import os
import io
import time
import openai
import pyaudio
import threading
import configparser
from pydub import AudioSegment
from pydub.playback import play
from google.cloud import texttospeech
from google.cloud import speech_v1 as speech

# ffmpeg와 ffprobe 경로 직접 설정
os.environ["PATH"] += os.pathsep + "C:\\Users\\wwsgb\\ffmpeg\\ffmpeg-6.1.1-full_build\\bin"

# 설정파일 로드
config = configparser.ConfigParser()
config.read('config.ini')
openai.api_key = config.get('api_key', 'openai_key')

class Transcriber:
    def __init__(self):
        # STT API 인증 정보 설정
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\wwsgb\\Json_key\\stt_json_key\\buoyant-mason-412215-845850c6db6c.json"
        self.client = speech.SpeechClient()
        self.transcript = ""
        self.last_transcript_time = time.time()
        self.history = []                # 대화 이력 저장
        self.silence_threshold = 5 * 60  # 5분간 대화 없을 시 리셋
        self.running = True
        self.is_speaking = False         # GPT 말하고 있는지에 대한 여부
        self.setup_audio_stream()
    
    def setup_audio_stream(self):   #pyaudio 활용해 새로운 오디오 입력 스트림 설정
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024  #처리 프레임(오디오 샘플 수)
        )
    
    def restart_audio_stream(self):     #기존의 오디오 입력 스트림 중지 -> 스트림 닫고 인스턴스 종료
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.pyaudio_instance.terminate()


    def transcribe_streaming(self):   # 실시간 사용자 음성 텍스트로 변환하는 함수
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",
            metadata=speech.RecognitionMetadata(
                interaction_type=speech.RecognitionMetadata.InteractionType.VOICE_SEARCH, 
                microphone_distance=speech.RecognitionMetadata.MicrophoneDistance.NEARFIELD,
                original_media_type=speech.RecognitionMetadata.OriginalMediaType.AUDIO,
                recording_device_type=speech.RecognitionMetadata.RecordingDeviceType.SMARTPHONE,
            )
        )

        #interim_results=True => 중간 결과도 반환하겠다.
        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

        def generate_requests():  #오디오 스트림에서 읽은 데이터를 StreamingRecognizeRequest 객체에 담아 STT API에 반환
            while self.running:
                data = stream.read(1024, exception_on_overflow=False)
                yield speech.StreamingRecognizeRequest(audio_content=data)    #STT에 오디어 데이터 전송하기 위한 요청 객체(제너레이터 함수(하나씩))

        requests = generate_requests()
        responses = self.client.streaming_recognize(streaming_config, requests) #STT로 전송해 음성인식 수행 (streaming_recognize 메소드로부터 반환된 스트리밍 응답 객체)

        for response in responses:
            if not response.results:
                continue

            result = response.results[-1] #가장 최근 결과 가져옴

            if result.is_final:
                transcript = result.alternatives[0].transcript
                
                if "작동 그만해" in transcript:
                    self.running = False
                    print("프로그램 중단")
                    break
                
                if not self.is_speaking: #GPT가 말하고 있지 않으면 인식된 텍스트 처리
                    self.transcript = transcript
                    self.last_transcript_time = time.time()  #마지막 시간 현재로 업데이트
                    self.process_response()
             
        stream.stop_stream()
        stream.close()
        p.terminate()


    def process_response(self):
        print("사용자 질문: {}".format(self.transcript))
        gpt_response = self.get_gpt_response(self.transcript)
        print("GPT 응답: {}".format(gpt_response))
        self.history.append((self.transcript, gpt_response))  # 대화 이력에 추가(저장소)
        self.text_to_speech(gpt_response)


    def text_to_speech(self, text):  # GPT로부터 받은 응답 음성 변환 함수, 마이크 제어 로직 포함
        self.is_speaking = True
        self.restart_audio_stream()  # 응답 시작 전 마이크 중지 (응답할때 오디오 스트림 재시작을 위한 마이크 중지)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\wwsgb\\Json_key\\tts_json_key\\buoyant-mason-412215-1df2f278305e.json"
        tts_client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        audio = io.BytesIO(response.audio_content)
        song = AudioSegment.from_mp3(audio)
        play(song)
        time.sleep(0.1)  # GPT 응답 후 0.1초 대기
        self.is_speaking = False
        self.setup_audio_stream()  # 응답 완료 후 오디오스트림 재시작


    def get_gpt_response(self, text): # 이전 질문과 답변 GPT 모델에 포함시키는 함수
        messages = [{"role": "system", "content": "일상 대화"}]
        for question, answer in self.history:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": text})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            return f"An error occurred: {str(e)}"


    def monitor_silence(self):
        while self.running:
            if time.time() - self.last_transcript_time > self.silence_threshold:
                if not self.is_speaking:
                    self.history.clear()  # 5분 이상 대화 없을 경우 입력 초기화
                    self.last_transcript_time = time.time()
            time.sleep(0.25)


transcriber = Transcriber()
transcription_thread = threading.Thread(target=transcriber.transcribe_streaming)
monitor_thread = threading.Thread(target=transcriber.monitor_silence)

transcription_thread.start()
monitor_thread.start()

transcription_thread.join()
monitor_thread.join()
