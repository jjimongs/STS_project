import os
import io
import time
import pickle
import openai
import threading
import pyaudio
import numpy as np
import configparser
import tensorflow as tf
from single_Local import FirebaseLocal
from konlpy.tag import Komoran
from pydub import AudioSegment
from pydub.playback import play
from google.cloud import texttospeech
from google.cloud import speech_v1 as speech

os.chdir('C:/Users/user/ElderlyCareRobot')
# ffmpeg와 ffprobe 경로 직접 설정 (환경변수에 추가했는데 잘 되지 않아 직접 설정)
os.environ["PATH"] += os.pathsep + "C:\\Users\\user\\ffmpeg-6.1.1-full_build\\bin"

local_instance = FirebaseLocal()

# 설정파일 로드(openai key)
config = configparser.ConfigParser()
config.read(r'C:\Users\user\Json_key\config.ini')  # 파일의 전체 경로 지정
openai_api_key = config.get('openai', 'api_key')

#감정분석
class DetectEmotion:
    def __init__(self, model_path, label_encoder_path):
        # 모델과 라벨 인코더를 로드합니다.
        self.model = tf.keras.models.load_model(model_path)
        with open(label_encoder_path, 'rb') as file:
            self.label_encoder = pickle.load(file)

        # 모든 연산을 float16으로 설정하여 메모리 사용량 감소
        tf.keras.backend.set_floatx('float16')

    def predict(self, texts):
        # 주어진 텍스트 리스트에 대해 감정을 예측합니다.
        #이 함수는 벡터화된 텍스트를 모델에 입력으로 제공하고, 예측된 감정 레이블을 반환합니다.

        vectorized_texts = self.model.layers[0](texts)
        predictions = self.model.predict(vectorized_texts, batch_size=1)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_emotions = self.label_encoder.inverse_transform(predicted_classes)
        return predicted_emotions

# 오디오 스트림 설정, 주변 소음 측정, STT 기능 클래스
class SpeechToText: #STT 함수
    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\user\\Json_key\\STT_key\\sigma-kayak-421806-2be227c3a876.json"
        self.client = speech.SpeechClient()
        self.tts = TextToSpeech()
        self.transcript = ""             # 곰돌아 빠진 음성 텍스트로 저장
        self.prompt = ""                 # 인식된 음성 감정에 맞게 프롬프트 텍스트 저장
        self.come_transcript = ""        # 인식된 음성 텍스트로 저장
        self.stream_use = False          # 오디오 스트림 사용 유무(기본 설정 = false)
        self.noise_level = 0             # 주변 소음 측정 (소음에 따라 마이크 음성 입력 강도 달라짐)
        self.setup_audio_stream()

    def setup_audio_stream(self):        # pyaudio 활용해 새로운 오디오 입력 스트림 설정
        start_time = time.time()         # 시간 측정 시작
        if not self.stream_use:          # 오디오가 들어오지 않고 있으면 오디오 스트림 켜기
            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )
        end_time = time.time()           # 시간 측정 종료
        self.stream_use = True
        print(f"오디오 스트림 설정 완료 시간: {end_time - start_time} 초")

    def restart_audio_stream(self):       # 기존의 오디오 입력 스트림 중지 -> 스트림 닫고 인스턴스 종료
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.pyaudio_instance.terminate()
        self.stream_use = False           # 스트림 중지 시 플래그 업데이트
        print("오디오 스트림 중지")

    def measure_ambient_noise(self, duration=1.0):  #주변 소음에 따라 음성 인식의 감도 조절
        if self.stream.is_active():
            frames = []
            for _ in range(int(self.stream._rate / self.stream._frames_per_buffer * duration)):
                data = self.stream.read(self.stream._frames_per_buffer)
                frames.append(np.frombuffer(data, dtype=np.int16))
            frame = np.concatenate(frames)
            return np.max(np.abs(frame)) / 32767
        else:
            return 0

    def adjust_recognition_sensitivity(self, noise_level):
        if noise_level < 0.1:
            speech_contexts = [speech.SpeechContext(phrases=["주변이 조용합니다."])]
        elif noise_level < 0.3:
            speech_contexts = [speech.SpeechContext(phrases=["주변이 약간 시끄럽습니다."])]
        else:
            speech_contexts = [speech.SpeechContext(phrases=["주변이 시끄럽습니다."])]
        return speech_contexts

    def process_emergency(self, word):          #응급 상황
        if word == '살리':
            print('살려줘(긴급) 관련 동작')
        elif '아프' in word:
            print('아파(긴급) 관련 동작')
        elif '이상' in word:
            print('이상해(긴급) 관련 동작')

    def process_schedule(self, word, nouns):    # 스케줄(일정) 관련
        if word == '맞추':
            if '일정' in nouns:
                print('일정 맞춤 관련동작 실행')
        elif word == '알리':
            if '일정' in nouns:
                print('일정 알림 관련동작 실행')

    def process_alarm(self, word, nouns):       # 알람 관련
        if word == '맞추':
            if '알람' in nouns:
                print('알람 맞춤 관련동작 실행')
        elif word == '알리':
            if '알람' in nouns:
                print('알람 알림 관련동작 실행')

    def process_weather(self, word, nouns):     #날씨 관련
        if word == '알리':
            if '날씨' in nouns:
                if '오늘' in nouns:
                    print('오늘 날씨 출력')
                    self.local.fetchWeatherData()
                    self.local.printTodayWeather()
                elif '내일' in nouns:
                    print('내일 날씨 출력')
                else:
                    print('오늘 날씨 출력')
                    self.local.fetchWeatherData()
                    self.local.printTodayWeather()
        elif '날씨' in nouns:
            if '내일' in nouns:
                print('내일 날씨 출력')
            elif '오늘' in nouns:
                print('오늘 날씨 출력')
                self.local.fetchWeatherData()
                self.local.printTodayWeather()
            else:
                print('오늘 날씨 출력')
                self.local.fetchWeatherData()
                self.local.printTodayWeather()

    def process_command(self, transcript):          #로컬 부분 처리 함수
        local = FirebaseLocal()
        pos_result = Komoran.pos(transcript)

        nouns = [word for word, pos in pos_result if pos in ('NNG', 'NNP')]
        verbs = [word for word, pos in pos_result if pos in ('VV', 'VA')]

        verb_indices = [i for i, (word, pos) in enumerate(pos_result) if pos == 'VV' and word not in ['있']]
        last_verb_index = verb_indices[-1] if verb_indices else -1

        for i, (word, pos) in enumerate(pos_result):
            if i == last_verb_index:
                self.process_emergency(word)
                self.process_schedule(word, nouns)
                self.process_alarm(word, nouns)
                self.process_weather(word, nouns)
                break
        else:
            self.process_emergency(verbs)
            self.process_weather(None, nouns)

    def transcribe_streaming(self):
       while True:
            print("음성 인식을 시작합니다. (텍스트로 변환 중)")
            if not self.stream_use:     #오디오 스트림 안켜져 있으면 스트림 켜기
                self.setup_audio_stream()

            if self.stream is not None and self.stream.is_active() and not self.tts.is_speaking:     # 스트림 존재하며 켜져 있을 때
                noise_level = self.measure_ambient_noise()              # 소음 축정 후 노이즈 레벨에 맞춘 음성 감도 조절
                speech_contexts = self.adjust_recognition_sensitivity(noise_level)

                if noise_level < 0.1:
                    self.stream.input_volume_float = 1.0
                elif noise_level < 0.3:
                    self.stream.input_volume_float = 0.7
                else:
                    self.stream.input_volume_float = 0.4

                # Speech-to-Text API를 사용하기 위한 설정
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="ko-KR",
                    metadata=speech.RecognitionMetadata(
                        interaction_type=speech.RecognitionMetadata.InteractionType.VOICE_SEARCH,
                        microphone_distance=speech.RecognitionMetadata.MicrophoneDistance.NEARFIELD,
                        original_media_type=speech.RecognitionMetadata.OriginalMediaType.AUDIO,
                        recording_device_type=speech.RecognitionMetadata.RecordingDeviceType.SMARTPHONE,
                    ),
                    speech_contexts=speech_contexts
                )

                streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

                def generate_requests():        #오디오 스트림에서 읽은 데이터를 StreamingRecognizeRequest 객체에 담아 STT API에 반환
                    start_time = time.time()
                    while time.time() - start_time < 300:  #5분 이하로만 실행
                        data = self.stream.read(1024, exception_on_overflow=False)
                        yield speech.StreamingRecognizeRequest(audio_content=data)

                requests = generate_requests()
                # STT로 전송해 음성인식 수행 (streaming_recognize 메소드로부터 반환된 스트리밍 응답 객체)
                # 음성 인식 결과는 responses 변수에 저장
                responses = self.client.streaming_recognize(streaming_config, requests)

                for response in responses:
                    if not response.results: #음성 인식 결과 없으면 건너뜀
                        continue

                    # 가장 최근 결과 가져옴 (마지막 결과)
                    result = response.results[-1]

                    if result.is_final: #결과가 최종 결과인 경우
                        transcript = result.alternatives[0].transcript

                        # 로컬 처리 부분 불러와 확인하기
                        is_processed = self.process_command(self.transcript)

                        if is_processed:
                            # process_command에서 처리된 경우 반환
                            return

                        # 로컬에서 처리 안된 경우 일상대화인지 확인
                        elif "곰돌아" in transcript or "공돌아" in transcript or "곤도라" in transcript or "곰도라" in transcript or "곰돌 아" in transcript:
                            self.come_transcript = transcript
                            print("들어온 질문: {}".format(self.come_transcript))
                            #곰돌아 뺴고 전달
                            clean_transcript = transcript.replace("곰돌아", "").replace("공돌아", "").replace("곤도라", "").replace("곰도라", "").replace("곰돌 아", "").strip()
                            self.transcript = clean_transcript
                            return self.transcript

                        else:
                            print("무시된 음성 입력:", transcript)

                self.stream.stop_stream()
                self.stream.close()
                self.pyaudio_instance.terminate()
                time.sleep(0.15)
                self.setup_audio_stream()

            if not self.stream.is_active() or self.tts.is_speaking:
                print("음성 인식이 중단되었습니다.")
                time.sleep(0.15)
class GPTResponse:
    def __init__(self):
        self.history = []                               # 대화 이력 저장
        self.local = FirebaseLocal()                    # local 처리 위한 인스턴스 생성
        self.komoran = Komoran(userdic='./word.txt')    # 형태소 분석기 인스턴스 생성
        self.last_transcript_time = time.time()         # 마지막 음성 시간 저장 변수
        self.alone_last_transcript_time = time.time()   # 혼자 뭐하세요?를 위한 마지막 음성 시간 저장 변수
        self.silence_threshold = 5 * 60                 # 5분 설정 (대화 이력 저장은 5분 이상 말 없으면 삭제)
        self.alone = 120 * 60                           # 혼자 뭐하세요? 의 2시간 설정
        self.response_timer = None
        self.silence_timer = None
        # 감정분석 클래스 인스턴스
        self.de = DetectEmotion('C:\\Users\\wwsgb\\OneDrive\\emotion\\light_emotion_model', 'C:\\Users\\wwsgb\\OneDrive\\emotion\\label_encoder.pkl')

    def get_gpt_response(self, text):   # STT로 받아온 텍스트에 대한 GPT 응답 가져오는 역할
        messages = [{"role": "system", "content": "일상 대화"}]
        for question, answer in self.history:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": text})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.7,
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def emotion_detection(self, transcript):    # 주어진 텍스트에 대한 감정 감지
        emotions = self.de.predict([transcript])
        return emotions[0]

    def process_response(self, transcript):     # 음성인식 결과 처리. GPT 응답 생성
        emotion = self.emotion_detection(transcript)
        print(f"감지된 감정: {emotion}")

        # 각 감정에 따른 프롬프트 작성
        if 'angry' in emotion or '분노' in emotion:
            prompt = f"사용자가 화났습니다. 사용자의 질문: {transcript}\n 에 대한 사용자의 마음을 진정시키고 합리적인 답변을 해주세요."
        elif 'disgust' in emotion:
            prompt = f"사용자가 역겨움을 느끼고 있습니다. 사용자의 질문: {transcript}\n이에 대한 이해하고 수용하는 태도로 답변을 해주세요."
        elif 'fear' in emotion or '불안' in emotion:
            prompt = f"사용자가 불안함 느끼고 있습니다. 사용자의 질문: {transcript}\n이에 대한 안정적이고 위로가 되는 답변을 해주세요."
        elif 'happiness' in emotion or '기쁨' in emotion:
            prompt = f"사용자가 기쁨을 느끼고 있습니다. 사용자의 질문: {transcript}\n이에 대한 밝고 긍정적인 답변을 해주세요."
        elif 'neutral' in emotion or '무감정' in emotion:
            prompt = f"사용자의 질문: {transcript}\n이에 대한  명확하고 객관적인 답변을 해주세요."
        elif 'sadness' in emotion or '슬픔' in emotion or '상처' in emotion:
            prompt = f"사용자가 상처받고, 슬퍼하고 있습니다. 사용자의 질문: {transcript}\n이에 대한 위로의 답변을 해주세요."
        elif 'surprise' in emotion or '당황' in emotion:
            prompt = f"사용자가 당황하고 있습니다. 사용자의 질문: {transcript}\n이에 대한 안정감을 주고 상황을 명확히 해주는 답변을 해주세요."
        else:
            prompt = f"사용자의 질문: {transcript}\n 이에 대한 답변을 해주세요."
        print(f"프롬프트: {prompt} ")

        # 프롬프트 작성 후 GPT 응답 받아오기
        gpt_response = self.get_gpt_response(prompt)
        print("GPT 응답: {}".format(gpt_response))
        self.history.append((transcript, gpt_response))
        self.last_transcript_time = time.time()
        self.alone_last_transcript_time = time.time()
        return gpt_response


    def start_response_timer(self):
        if self.response_timer is not None:
            self.response_timer.cancel()
        self.response_timer = threading.Timer(self.alone, self.check_alone)
        self.response_timer.start()

    def start_silence_timer(self):
        if self.silence_timer is not None:
            self.silence_timer.cancel()
        self.silence_timer = threading.Timer(self.silence_threshold, self.reset_memory)
        self.silence_timer.start()

    def reset_memory(self):
        if time.time() - self.last_transcript_time > self.silence_threshold:
            print(f"{self.silence_threshold}초 이상 대화 없어 메모리 초기화합니다.")
            self.history.clear()
            self.last_transcript_time = time.time()
            self.alone_last_transcript_time = time.time()

    def check_alone(self):
        if time.time() - self.alone_last_transcript_time > self.alone:
            print(" 혼자 뭐하세요?")
            self.alone_last_transcript_time = time.time()
            self.start_response_timer()

class TextToSpeech:
    def __init__(self):
        self.is_speaking = False

    def text_to_speech(self, text):
        self.is_speaking = True
        stt = SpeechToText()
        stt.restart_audio_stream()
        while stt.stream_use:
            time.sleep(0.1)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\user\\Json_key\\TTS_key\\sigma-kayak-421806-c28bf22ca0f0.json"
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
        time.sleep(0.2)
        stt.setup_audio_stream()
        self.is_speaking = False

def main():
    stt = SpeechToText()
    gpt = GPTResponse()
    tts = TextToSpeech()

    while True:
        transcript = stt.transcribe_streaming()
        if transcript:
            gpt_response = gpt.process_response(transcript)
            gpt.start_silence_timer()
            gpt.start_response_timer()
            tts.text_to_speech(gpt_response)

            # audio_listener에서 제거한 부분을 여기에 추가합니다.
            local_instance.fm.audio_listener(None)

if __name__ == "__main__":
    main()
