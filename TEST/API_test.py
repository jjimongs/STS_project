from google.cloud import speech_v1p1beta1 as speech
import os

# 인증 파일 경로 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\wwsgb\\tts_json_key\\buoyant-mason-412215-39cbac402c06.json"

def transcribe_audio():
    client = speech.SpeechClient()

    with open("test.wav", "rb") as audio_file:
        input_audio = audio_file.read()

    audio = speech.RecognitionAudio(content=input_audio)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,  # 샘플링 빈도를 16000으로 변경
        language_code="ko-KR",
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        print("Transcript: {}".format(result.alternatives[0].transcript))

# 함수 실행
transcribe_audio()
