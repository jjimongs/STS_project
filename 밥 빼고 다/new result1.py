#밥, 혼자 계시나요?, 날씨, 알람 넣는데 왜 streaming 다시 실행 안하는지,
# 힙 손상 -> tts X => 멀티쓰레딩 문제일지도
import os
import io
import time
import pickle
import openai
import threading
import pyaudio
import numpy as np
import tensorflow as tf
from konlpy import jvm
from pydub import AudioSegment
from pydub.playback import play
from google.cloud import texttospeech
from google.cloud import speech_v1 as speech
from TaskManager import TaskManager
import asyncio

from singleton import SingletonMeta

# os.chdir('C:/Users/user/ElderlyCareRobot')
os.environ["PATH"] += os.pathsep + "C:\\Users\\user\\ffmpeg-6.1.1-full_build\\bin"

# os.environ["PATH"] += os.pathsep + "C:\\Users\\grpug\\ffmpeg-6.1.1-full_build\\bin"

# os.environ["PATH"] += os.pathsep +  "C:\\Users\\grpug\\ffmpeg\\bin"

class DetectEmotion(metaclass=SingletonMeta):
    def __init__(self, model_path, label_encoder_path):
        model_path = 'C:\\Users\\user\\OneDrive\\emtion\\light_emotion_model'
        label_encoder_path = 'C:\\Users\\user\\OneDrive\\emtion\\label_encoder.pkl'
        self.model = tf.keras.models.load_model(model_path)
        with open(label_encoder_path, 'rb') as file:
            self.label_encoder = pickle.load(file)
        tf.keras.backend.set_floatx('float16')

    def predict(self, texts):
        vectorized_texts = self.model.layers[0](texts)
        predictions = self.model.predict(vectorized_texts, batch_size=1)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_emotions = self.label_encoder.inverse_transform(predicted_classes)
        return predicted_emotions


class SpeechToText(metaclass=SingletonMeta):
    def __init__(self):
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\grpug\\Json_key\\stt_json_key\\careful-hangar-423707-k6-43fb48a2baff.json"
        os.environ[
            "GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\user\\Json_key\\STT_key\\careful-hangar-423707-k6-43fb48a2baff.json"
        self.client = speech.SpeechClient()
        self.tm = TaskManager().get_instance()
        self.tts = TextToSpeech().get_instance()
        # self.gpt = GPTResponse().get_instance()
        self.transcript = ""  # 곰돌아 빠진 음성 텍스트로 저장
        self.prompt = ""  # 인식된 음성 감정에 맞게 프롬프트 텍스트 저장
        self.come_transcript = ""  # 인식된 음성 텍스트로 저장
        self.stream_use = False  # 오디오 스트림 사용 유무(기본 설정 = false)
        self.noise_level = 0  # 주변 소음 측정 (소음에 따라 마이크 음성 입력 강도 달라짐)
        self.setup_audio_stream()
        self.sleeping = False      # 자는지 아닌지
        self.daily = False         # 로컬인지 아닌지
        self.current_question = None   # 밥 상태, 약 상태 업로드 하는 부분
        self.is_quiz_answer = False    #퀴즈 답이 왓는지
        self.weather_data = ""       #날씨가 왔는지
        self.emergency = False        # 응급 상황 구별을 위함
        self.quiz_answer = ""         # 퀴즈 답 저장 부분

    def setup_audio_stream(self):
        start_time = time.time()
        if not self.stream_use:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )
        end_time = time.time()
        self.stream_use = True
        print(f"오디오 스트림 설정 완료 시간: {end_time - start_time} 초")

    def restart_audio_stream(self):
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.pyaudio_instance.terminate()
        self.stream_use = False
        print("오디오 스트림 중지")

    def measure_ambient_noise(self, duration=1.0):
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

    def process_emergency(self, transcript):  # 응급 상황
        from Local import Local
        local = Local().get_instance()

        print('응급상황 발생')
        local.set_emergency()
        local.set_not_emergency()
        emergency = "응급상황 동작을 보호자에게 보냈습니다"
        return {'type': 'emergency', 'data': emergency }


    def process_weather(self, transcript):  # 날씨 관련
        from Local import Local
        local = Local().get_instance()


        print('오늘 날씨 출력')
        self.weather_data = local.fetchWeatherData()
        print(f"'날씨 데이터': {self.weather_data}")


        return {'type': 'weather', 'data': self.weather_data}


    async def transcribe_answer(self):   # 퀴즈 처리를 위한 스트리밍 부분 (비동기)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR"
        )

        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

        def generate_requests():
            while True:
                data = self.stream.read(1024, exception_on_overflow=False)
                yield speech.StreamingRecognizeRequest(audio_content=data)

        requests = generate_requests()
        responses = self.client.streaming_recognize(streaming_config, requests)

        for response in responses:
            if not response.results:
                continue

            result = response.results[-1]

            if result.is_final:
                return result.alternatives[0].transcript

    def process_quiz(self, transcript):  # 퀴즈 풀래 ( 퀴즈 불러오는 함수 )
        from Local import Local
        local = Local().get_instance()

        selected_quizzes = local.select_random_quizzes()
        local.set_quiz()

        for quiz in selected_quizzes:
            question = quiz[0]
            answer = quiz[1]
            return {'type': 'quiz', 'question': question, 'answer': answer} #타입은 퀴즈, 문제 & 답안 저장


    def process_schedule(self,transcript):   # 스케줄(일정) 저장을 위한 함수
        from Local import Local
        local = Local().get_instance()

        self.promise_transcript = transcript
        print(f"promise_transcript : {self.promise_transcript}")
        local.add_alarm(self.promise_transcript)
        promise = f"{self.promise_transcript} 에 대한 일정이 저장되었습니다."
        return {'type': 'schedule', 'data': promise}


    def process_meal_medi(self, transcript):
        from Local import Local
        local = Local().get_instance()
        # 밥을 먹었다 -> setmeal()만 호출하면 됨
        if self.current_question == "meal" and "먹었어" in transcript:
            if "안" in transcript or "못" in transcript or "않" in transcript:
                self.daily = True
                print("사용자가 밥을 먹지 않았다고 응답했습니다.")
                self.current_question = None
            else:
                self.daily = True
                self.restart_audio_stream()
                print("사용자가 밥을 먹었다고 응답했습니다.")
                local.set_meal()
                self.setup_audio_stream()
                self.current_question = None

        elif self.current_question == "medi" and "먹었어" in transcript:
            if "안" in transcript or "못" in transcript or "않" in transcript:
                self.daily = True
                print("사용자가 약을 먹지 않았다고 응답했습니다.")
                self.current_question = None
            else:
                self.daily = True
                self.restart_audio_stream()
                print("사용자가 약을 먹었다고 응답했습니다.")
                local.set_medi()
                self.setup_audio_stream()
                self.current_question = None

    """def check_alone1(self):
        from TaskManager import TaskManager
        self.task_manager = TaskManager().get_instance()
        self.alone_Flag = True
        self.task_manager.add_task(lambda: self.tts.text_to_speech("지금 잘 계시나요?"))
        """

    async def Audio_speak(self, transcript):
        from Local import Local
        local = Local().get_instance()
        if "녹음" in transcript and "다시" in transcript:
            self.daily = True
            print("녹음 다시 듣기 local로 이동합니다")
            chosen_file = await local.RandomChoiceAudio()
            return {'type': 'Audio', 'data': chosen_file}
        return None

    def is_daily(self, transcript):
        daliy_list = ['살려','아파','날씨','알려 줘', '알려줘','퀴즈', '일정', '약속']
        for daliy_word in daliy_list:
            if daliy_word in transcript:
                self.daily = True
                break

    def process_command(self, transcript):  # 로컬 부분 처리 함수
        # 응급 상황 처리
        if '살려' in transcript or "아파" in transcript :
            self.emergency = True
            emergency_result = self.process_emergency(transcript)
            return emergency_result

        # 날씨 데이터 처리
        elif "날씨" in transcript and "알려 줘" in transcript:
            weather_result = self.process_weather(transcript)
            weather_dict = {
                    'type': weather_result['type'],
                    'data': weather_result['data']  # 문자열 데이터를 딕셔너리의 data 키에 할당
            }
            return weather_dict


        # 퀴즈 처리
        elif "퀴즈" in transcript:
            quiz_result = self.process_quiz(transcript)
            quiz_dict = {
                'type': quiz_result['type'],
                'question': quiz_result['question'], # 문자열 데이터를 딕셔너리의 question 키에 할당
                'answer': quiz_result['answer']  # 문자열 데이터를 딕셔너리의 question 키에 할당
            }
            return quiz_dict

        # 일정 처리
        elif "일정" in transcript or "약속" in transcript:
            schedule_result = self.process_schedule(transcript)
            return schedule_result

        # 모든 조건에 맞지 않을 경우
        else:
            print("해당하는 처리 명령이 없습니다.")
            return None

    async def transcribe_streaming(self):
        from Local import Local
        from TaskManager import TaskManager
        local = Local().get_instance()
        task_manager = TaskManager.get_instance()

        while True:

            if not task_manager.task_in_progress:
                print("음성 인식을 시작합니다. (텍스트로 변환 중)")

                if not self.stream_use:
                    self.setup_audio_stream()

                if self.stream is not None and self.stream.is_active() and not self.tts.is_speaking:
                    noise_level = self.measure_ambient_noise()
                    speech_contexts = self.adjust_recognition_sensitivity(noise_level)

                    if noise_level < 0.1:
                        self.stream.input_volume_float = 1.0
                    elif noise_level < 0.3:
                        self.stream.input_volume_float = 0.7
                    else:
                        self.stream.input_volume_float = 0.4

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

                    def generate_requests():
                        start_time = time.time()
                        while time.time() - start_time < 300:
                            data = self.stream.read(1024, exception_on_overflow=False)
                            yield speech.StreamingRecognizeRequest(audio_content=data)

                    requests = generate_requests()
                    responses = self.client.streaming_recognize(streaming_config, requests)

                    for response in responses:
                        print("response")

                        if not response.results:
                            continue

                        result = response.results[-1]

                        if result.is_final:
                            transcript = result.alternatives[0].transcript


                            self.is_daily(transcript)

                            if "잘게" in transcript or "외출 다녀올게" in transcript:
                                self.sleeping = True
                                print("모드 비활성화")
                                local.state_ref.update({
                                    'currentState': 4
                                })
                                task_manager.set_user_status(4)
                                return None

                            elif "일어났어" in transcript or "다녀왔어" in transcript:
                                self.sleeping = False
                                print("모드 활성화, 다시 정상작동 시작합니다.")
                                local.state_ref.update({
                                    'currentState': 1
                                })
                                task_manager.set_user_status(1)
                                local.ProcessRecentAudio()
                                return None

                            if self.sleeping:
                                print("모드 비활성화 중입니다. GPT 요청을 건너뜁니다.")
                                return None

                            elif self.daily and self.emergency:
                                # 응급상황 처리
                                print(f"로컬 처리 결과(응급): {transcript}")
                                self.daily = False
                                self.emergency = False
                                print("응급! false 완료.")

                            else:
                                if self.daily:
                                    if "곰돌아" in transcript or "공돌아" in transcript  or "공도아" in transcript or "곤도라" in transcript or "곰도라" in transcript or "곰돌 아" in transcript or "구글아" in transcript or "구글 아" in transcript:
                                        self.come_transcript = transcript
                                        print(f"로컬 처리 결과(나머지): {self.come_transcript}")
                                        command_result = self.process_command(self.come_transcript)
                                        print(f"Command Result1: {command_result}")  # 로그 추가
                                        audio_result = await self.Audio_speak(self.come_transcript)

                                        response = command_result or audio_result
                                        print(f"response : {response}")
                                        print()
                                        final_response = await self.handle_response(response)
                                        print(f"final_response: {final_response} ")
                                        print()
                                        await self.Audio_speak(final_response)
                                        return final_response

                                    else:
                                        print(f"무시된 음성 입력(나머지): {transcript}")
                                        from TaskManager import TaskManager
                                        self.task_manager = TaskManager().get_instance()
                                        if not self.task_manager.task_queue.empty():
                                            print("Processing background tasks...")
                                            await task_manager.process_tasks()

                                else:#GPT 처리 부분
                                    if "곰돌아" in transcript or "공돌아" in transcript or "곤도라" in transcript or "곰도라" in transcript or "곰돌 아" in transcript or "구글아" in transcript:
                                        self.come_transcript = transcript
                                        print(f"들어온 질문(gpt): {self.come_transcript}")
                                        clean_transcript = transcript.replace("곰돌아", "").replace("공돌아", "").replace("곤도라",
                                                                                                                    "").replace(
                                            "곰도라", "").replace("곰돌 아", "").strip()
                                        self.transcript = clean_transcript
                                        return self.transcript,"GPT"

                                    else:
                                        print(f"무시된 음성 입력: {transcript}")
                                        from TaskManager import TaskManager
                                        self.task_manager = TaskManager().get_instance()
                                        if not self.task_manager.task_queue.empty() :
                                            print("Processing background tasks...")
                                            await task_manager.process_tasks()

                        # 비동기적으로 잠시 대기하여 다른 작업 처리 기회
                        await asyncio.sleep(0.1)

                    if time.time() - self.last_input_time > 300:
                        print(f"5분 동안 말이 없어 재시작합니다. 경과 시간: {time.time() - self.last_input_time}초")
                        self.restart_audio_stream()
                        time.sleep(0.25)
                        self.pyaudio_instance.terminate()
                        time.sleep(0.15)
                        self.setup_audio_stream()
                        self.last_input_time = time.time()
                        print(f"오디오 스트림 재시작 시간: {self.last_input_time}")

            if not self.stream.is_active() or self.tts.is_speaking:
                print("음성 인식이 중단되었습니다.")
                await asyncio.sleep(0.15)

    async def handle_response(self, response):
        print("핸들 응답 : ",response)
        print("핸들 응답 타입 : ", type(response))

        if isinstance(response, dict):
            if response['type'] == 'emergency':
                return f"응급 상황 발생: {response['data']}","없음"
            elif response['type'] == 'weather':
                return f"현재 날씨는 다음과 같습니다: {response['data']}","없음"
            elif response['type'] == 'quiz':
                return f"퀴즈 질문: {response['question']}",response['answer']
            elif response['type'] == 'schedule':
                return f"약속: {response['data']}","없음"
            elif response['type'] == 'Audio':
                return response['data'],"녹음"
            else:
                return "명령을 인식할 수 없습니다.","없음"


        else:
            print("handle_response 함수는 딕셔너리 타입의 입력을 기대합니다.")
            return "입력 형식 오류","없음"


class GPTResponse(metaclass=SingletonMeta):
    def __init__(self):
        self.history = []
        self.last_transcript_time = time.time()
        self.alone_last_transcript_time = time.time()
        self.silence_threshold = 5 * 60
        self.alone = 120 * 60
        self.response_timer = None
        self.silence_timer = None
        self.alone_Flag = False
        self.de = DetectEmotion('light_emotion_model210', 'label_encoder.pkl')
        self.tts = TextToSpeech().get_instance()
        self.stt = SpeechToText().get_instance()

    def get_gpt_response(self, text):
        messages = [{"role": "system", "content": "일상 대화"}]
        for question, answer in self.history:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": text})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.7,
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def emotion_detection(self, transcript):
        emotions = self.de.predict([transcript])
        return emotions[0]

    def process_response(self, transcript):
        from Local import Local
        local = Local().get_instance()
        emotion = self.emotion_detection(transcript)
        print(f"감지된 감정: {emotion}")

        if 'angry' in emotion or '분노' in emotion:
            prompt = f"사용자가 화났습니다. 사용자의 질문: {transcript}\n 에 대한 사용자의 마음을 진정시키고 합리적인 답변을 해주세요. 근데 감정에 몰두하지 말고 사용자의 질문에 대해 더욱 집중하여 질문에 맞는 응답을 하되, 감정에 대한 답변을 살짝 추가해주세요. 또한 언제나 존댓말로 출력해주세요. 추가로 답변에 이모티콘 사용은 하지 않았으면 좋겠어요"
            local.setMood("나쁨")
        elif 'disgust' in emotion:
            prompt = f"사용자가 역겨움을 느끼고 있습니다. 사용자의 질문: {transcript}\n이에 대한 이해하고 수용하는 태도로 답변을 해주세요.근데 감정에 몰두하지 말고 사용자의 질문에 대해 더욱 집중하여 질문에 맞는 응답을 하되, 감정에 대한 답변을 살짝 추가해주세요. 또한 언제나 존댓말로 출력해주세요. 추가로 답변에 이모티콘 사용은 하지 않았으면 좋겠어요"
            local.setMood("나쁨")
        elif 'fear' in emotion or '불안' in emotion:
            prompt = f"사용자가 불안함 느끼고 있습니다. 사용자의 질문: {transcript}\n이에 대한 안정적이고 위로가 되는 답변을 해주세요.근데 감정에 몰두하지 말고 사용자의 질문에 대해 더욱 집중하여 질문에 맞는 응답을 하되, 감정에 대한 답변을 살짝 추가해주세요. 또한 언제나 존댓말로 출력해주세요. 추가로 답변에 이모티콘 사용은 하지 않았으면 좋겠어요."
            local.setMood("나쁨")
        elif 'happiness' in emotion or '기쁨' in emotion:
            prompt = f"사용자가 기쁨을 느끼고 있습니다. 사용자의 질문: {transcript}\n이에 대한 밝고 긍정적인 답변을 해주세요.근데 감정에 몰두하지 말고 사용자의 질문에 대해 더욱 집중하여 질문에 맞는 응답을 하되, 감정에 대한 답변을 살짝 추가해주세요. 또한 언제나 존댓말로 출력해주세요. 추가로 답변에 이모티콘 사용은 하지 않았으면 좋겠어요."
            local.setMood("좋음")
        elif 'neutral' in emotion or '무감정' in emotion:
            prompt = f"사용자의 질문: {transcript}\n이에 대한  명확하고 객관적인 답변을 해주세요."
            local.setMood("보통")
        elif 'sadness' in emotion or '슬픔' in emotion or '상처' in emotion:
            prompt = f"사용자가 상처받고, 슬퍼하고 있습니다. 사용자의 질문: {transcript}\n이에 대한 위로의 답변을 해주세요.근데 감정에 몰두하지 말고 사용자의 질문에 대해 더욱 집중하여 질문에 맞는 응답을 하되, 감정에 대한 답변을 살짝 추가해주세요. 또한 언제나 존댓말로 출력해주세요. 추가로 답변에 이모티콘 사용은 하지 않았으면 좋겠어요."
            local.setMood("나쁨")
        elif 'surprise' in emotion or '당황' in emotion:
            prompt = f"사용자가 당황하고 있습니다. 사용자의 질문: {transcript}\n이에 대한 안정감을 주고 상황을 명확히 해주는 답변을 해주세요.근데 감정에 몰두하지 말고 사용자의 질문에 대해 더욱 집중하여 질문에 맞는 응답을 하되, 감정에 대한 답변을 살짝 추가해주세요. 또한 언제나 존댓말로 출력해주세요. 추가로 답변에 이모티콘 사용은 하지 않았으면 좋겠어요. "
            local.setMood("나쁨")
        else:
            prompt = f"사용자의 질문: {transcript}\n 이에 대한 답변을 해주세요."
            local.setMood("보통")
        print(f"프롬프트: {prompt} ")

        gpt_response = self.get_gpt_response(prompt)
        print("GPT 응답: {}".format(gpt_response))
        self.history.append((transcript, gpt_response))
        self.last_transcript_time = time.time()
        self.alone_last_transcript_time = time.time()
        # print("alone_last_time 업데이트 (gpt 부분)")
        return gpt_response

    def start_response_timer(self):
        if self.response_timer is not None:
            self.response_timer.cancel()

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


class TextToSpeech(metaclass=SingletonMeta):
    def __init__(self):
        self.is_speaking = False

    def text_to_speech(self, text):

        try:
            self.is_speaking = True
            stt = SpeechToText().get_instance()
            gpt = GPTResponse().get_instance()
            stt.restart_audio_stream()
            print(1)
            print("111111",text)
            print(type(text))
            while stt.stream_use:
                time.sleep(0.1)
            # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\wwsgb\\Json_key\\tts_json_key\\sigma-kayak-421806-c28bf22ca0f0.json"
            # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] ="C:\\Users\\grpug\\Json_key\\tts_json_key\\careful-hangar-423707-k6-f3a7f26801af.json"
            os.environ[
                "GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\user\\Json_key\\TTS_key\\careful-hangar-423707-k6-f3a7f26801af.json"
            tts_client = texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="ko-KR",
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)


            print(2)

            response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            print(3)
            print("Response audio content type:", type(response.audio_content))
            print("Length of audio content:", len(response.audio_content))
            audio_data = io.BytesIO(response.audio_content)
            audio_segment = AudioSegment.from_file(audio_data, format="mp3")
            print("Audio data is valid.")
            audio = io.BytesIO(response.audio_content)
            print(audio)
            print(4)
            try:
                song = AudioSegment.from_mp3(audio)
                print(5)
                play(song)
                print(6)
            except Exception as e:
                print("Error converting audio to song:", str(e))
            time.sleep(0.2)
            stt.setup_audio_stream()
            gpt.alone_last_transcript_time = time.time()
            gpt.last_transcript_time = time.time()
            stt.last_response_time = time.time()
            # print("alone_last_time 업데이트 (tts 부분)")
            self.is_speaking = False
            # await asyncio.sleep(1)  # 비동기적 대기를 시뮬레이션
        except Exception as e:
            print(f"An error occurred: {str(e)}")  # 오류 메시지 출력
            raise

    def text_to_speech1(self, text):
        try:
            self.is_speaking = True
            stt = SpeechToText().get_instance()
            stt.restart_audio_stream()
            print(1)
            print("22222222222222",text)
            print(type(text))
            while stt.stream_use:
                time.sleep(0.1)
            # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\wwsgb\\Json_key\\tts_json_key\\sigma-kayak-421806-c28bf22ca0f0.json"
            # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] ="C:\\Users\\grpug\\Json_key\\tts_json_key\\careful-hangar-423707-k6-f3a7f26801af.json"
            os.environ[
                "GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\user\\Json_key\\TTS_key\\careful-hangar-423707-k6-f3a7f26801af.json"
            tts_client = texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="ko-KR",
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            print(2)

            response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            print(3)
            print("Response audio content type:", type(response.audio_content))
            print("Length of audio content:", len(response.audio_content))
            audio_data = io.BytesIO(response.audio_content)
            audio_segment = AudioSegment.from_file(audio_data, format="mp3")
            print("Audio data is valid.")
            audio = io.BytesIO(response.audio_content)
            print(audio)
            print(4)
            try:
                song = AudioSegment.from_mp3(audio)
                print(5)
                play(song)
                print(6)
            except Exception as e:
                print("Error converting audio to song:", str(e))
            time.sleep(0.2)
            stt.setup_audio_stream()
            stt.last_response_time = time.time()
            # print("alone_last_time 업데이트 (tts 부분)")
            self.is_speaking = False
            # await asyncio.sleep(1)  # 비동기적 대기를 시뮬레이션
        except Exception as e:
            print(f"An error occurred: {str(e)}")  # 오류 메시지 출력
            raise


def main():
    stt = SpeechToText().get_instance()
    gpt = GPTResponse().get_instance()
    tts = TextToSpeech().get_instance()

    while True:
        transcript = stt.transcribe_streaming()
        if transcript:
            gpt_response = gpt.process_response(transcript)
            gpt.start_silence_timer()
            gpt.start_response_timer()
            tts.text_to_speech(gpt_response)
        time.sleep(0.25)


def test():
    tts = TextToSpeech().get_instance()
    tts.text_to_speech("안녕하세요")


if __name__ == "__main__":
    main()
