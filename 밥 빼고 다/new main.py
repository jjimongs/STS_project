import asyncio
import configparser

import pygame

from Local import Local
import openai

from pydub import AudioSegment
from pydub.playback import play
from TaskManager import TaskManager
from STS.result1 import SpeechToText, GPTResponse, TextToSpeech
import os
import os

async def play_audio_file(file_path):
    try:
        song = AudioSegment.from_file(file_path)
        play(song)
    except Exception as e:
        print(f"Error playing the audio file: {e}")

# 비동기적으로 음성 인식을 처리하는 함수
async def run_transcription_and_response():
    stt = SpeechToText().get_instance()
    gpt = GPTResponse().get_instance()
    tts = TextToSpeech().get_instance()
    task_manager = TaskManager.get_instance()
    awaiting_answer = False  # 퀴즈 답변 대기 플래그
    correct_answer = None  # 정답 저장 변수

    while True:
        n = 0
        print(f"task_manager.task_queue.empty(): {task_manager.task_queue.empty()}")
        print(f"task_manager.task_in_progress: {task_manager.task_in_progress}")

        #if task_manager.task_queue.empty() and not task_manager.task_in_progress:
        data,answer = await stt.transcribe_streaming()  # 수정된 부분: transcribe_streaming으로부터 직접 응답 받기
        print("데이터가 뭔데 안들어가?", data)
        if data:
            if answer == "없음":
                response = data  # handle_response 호출하여 응답 처리
                print("불린횟수 :", n)
                if response:
                    print("response에 알맞는 음성 출력합니다.")

                    tts.text_to_speech(response)  # 응답을 TTS로 출력
                    print("정말 출력했나요?")

            elif answer == "GPT":
                    print("1.1.1.1")
                    task_manager.set_task_in_progress(True)
                    gpt_response = gpt.process_response(data)
                    gpt.start_silence_timer()
                    gpt.start_response_timer()
                    tts.text_to_speech(gpt_response)  # 비동기 호출
                    task_manager.set_task_in_progress(False)

            elif answer == "녹음":
                    print("오디오 재생합니다")
                    await play_audio_file(data)  # 오디오 파일 재생
            else:
                correct_answer = answer
                tts.text_to_speech(data)  # 응답을 TTS로 출력
                # 사용자 응답을 기다리는 경우
                user_response = await stt.transcribe_answer()
                print("사용자의 응답", user_response)
                if user_response:
                    if correct_answer.lower() in user_response.lower():
                        tts.text_to_speech("정답입니다!")
                    else:
                        tts.text_to_speech(f"틀렸습니다. 정답은 {correct_answer}입니다.")
        else:
            print("데이터 xxxxxxxxxxxxxxxxxxxx")

    """else:
        print("Processing background tasks...")
        await task_manager.process_tasks()"""

    #await asyncio.sleep(1)

    """
    while True:
        if not task_manager.task_in_progress:
            task_manager.set_speech_recognition_status(True)
            transcript = await stt.transcribe_streaming()  # 비동기 호출
            task_manager.set_speech_recognition_status(False)

            if transcript:
                task_manager.set_user_status(1)

                if task_manager.user_status == 1:
                    task_manager.set_task_in_progress(True)
                    gpt_response = gpt.process_response(transcript)
                    gpt.start_silence_timer()
                    gpt.start_response_timer()
                    await tts.text_to_speech(gpt_response)  # 비동기 호출
                    task_manager.set_task_in_progress(False)
            await asyncio.sleep(0.1)  # 비동기 대기
    """

# 비동기적으로 태스크 큐를 처리하는 함수
async def run_task_processing():
    task_manager = TaskManager.get_instance()
    print("run_task_processing started")  # 함수 시작 확인
    while True:
        print(f"task_manager.task_queue.empty() : {task_manager.task_queue.empty()}")
        print(f"task_manager.is_quiet_time() : {task_manager.is_quiet_time()}")
        print(f"task_manager.task_in_progress : {task_manager.task_in_progress}")
        print(f"task_manager.task_queue.empty() and not task_manager.task_in_progress and not task_manager.is_quiet_time(): : {task_manager.task_in_progress}")

        #if not task_manager.task_queue.empty() and not task_manager.task_in_progress and not task_manager.is_quiet_time():
        if not task_manager.task_queue.empty() :
            await task_manager.process_tasks()
        await asyncio.sleep(1)


"""async def main():
    local = Local().get_instance()
    local.initialize_firebase_app()

    tts = TextToSpeech().get_instance()
    # 프로그램이 시작될 때 TTS로 초기 메시지를 출력

    pygame.mixer.init()

    # 파일 재생
    pygame.mixer.music.load('cyberpunk.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    await tts.text_to_speech("안녕하세요 저랑 재미있는 대화를 해봐요")
    # 비동기 작업 생성 및 실행
    await asyncio.gather(
        run_transcription_and_response(),
        run_task_processing()
    )"""

# 음성 인식과 태스크 처리를 병렬로 실행하는 메인 함수
async def main():
    local = Local().get_instance()
    local.initialize_firebase_app()

    tts = TextToSpeech().get_instance()
    # 프로그램이 시작될 때 TTS로 초기 메시지를 출력

    pygame.mixer.init()

    # 파일 재생
    pygame.mixer.music.load('cyberpunk.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    tts.text_to_speech("안녕하세요 저랑 재미있는 대화를 해봐요")
    # 비동기 작업 생성 및 실행
    await run_transcription_and_response()

if __name__ == "__main__":


    #os.environ["PATH"] += os.pathsep + "C:\\Users\\grpug\\ffmpeg\\ffmpeg-6.1.1-full_build\\bin"

    # 설정파일 로드
    config = configparser.ConfigParser()
    config.read('STS/config.ini')
    openai.api_key = config.get('openai', 'api_key')
    #main()
    asyncio.run(main())

"""if __name__ == '__main__':
    local = Local()
    local.loadData()
    local.updateStatus(True)
    local.fetchWeatherData()"""
""" # 상태 변경을 감지하기 위한 키워드
                if "다녀왔어" or "다녀 왔어" in transcript:
                    task_manager.set_user_status(1)
                    print("사용자의 상태가 1로 변경되었습니다.")
                elif "다녀올게" or "다녀 올게" in transcript:
                    task_manager.set_user_status(0)
                    print("사용자의 상태가 0으로 변경되었습니다.")"""
