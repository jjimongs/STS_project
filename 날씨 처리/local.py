import os
import re
import firebase_admin
import numpy as np
from firebase_admin import credentials, db
import random
from datetime import datetime, timedelta
from google.cloud import storage
from google.oauth2 import service_account
import asyncio
import pandas as pd
import requests
import pytz
import json
import pygame
import time

from LocalScheduler import LocalScheduler
from ScheduleStore import ScheduleStore
from TaskManager import TaskManager
import threading
import subprocess
from singleton import SingletonMeta

class Local(metaclass=SingletonMeta):
    def __init__(self):
        self.googlecred = service_account.Credentials.from_service_account_file("firebase-adminsdk.json")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "firebase-adminsdk.json"

        # 파이어베이스 앱 초기화 확인 및 설정
        self.initialize_firebase_app()

        # 파이어베이스 앱 초기화
        self.cred = credentials.Certificate("firebase-adminsdk.json")

        # Google Cloud Storage 클라이언트 초기화
        storage_client = storage.Client(credentials=self.googlecred, project=self.googlecred.project_id)
        self.save_file = 'saved_data.json'
        self.sn = 'qwer'  # 시리얼 넘버
        self.storagepath = 'projact01-82fe5.appspot.com'
        self.excelpath = 'weather/weather.xlsx'
        self.uid = ""  # 임의로 지정
        self.audio_folder = 'audio'


        # 기상청
        self.APIKEY = 'wPjxtRD78GNtmOnTPl/PRjST3uog3CItQ+bdVaSxK9aeTPgtCGsMD5585FHl098w4xAkscTei7UjhnoY1+8wcQ=='
        self.BASEURL = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'

        self.YOUTUBEAPIKEY = 'AIzaSyAmOWV0BS1q0yVufEmbdeXObmYbh03iK9Q'

        self.address = None
        self.weatherdata = None

        self.todayweather = {
            'date': '',
            'sky': '4',
            'precip': '1',
            'lowtemp': '8.0',
            'hightemp': '13.0'
        }

        self.params = {
            'serviceKey': 'wPjxtRD78GNtmOnTPl/PRjST3uog3CItQ+bdVaSxK9aeTPgtCGsMD5585FHl098w4xAkscTei7UjhnoY1+8wcQ==',
            'numOfRows': '160',
            'dataType': 'JSON',
            'base_date': '',
            'base_time': '0200',
            'nx': '',
            'ny': ''
        }

        self.STATUS_OF_SKY = {
            '1': '맑음️',
            '3': '구름많음',
            '4': '흐림',
        }

        self.STATUS_OF_PRECIPITATION = {
            '0': '없음',
            '1': '비',
            '2': '비나 눈',
            '3': '눈',
            '4': '소나기'
        }

        self.data = {}
        self.loadData()

        self.task_manager = TaskManager.get_instance()
        self.state_ref = db.reference(f'UserAccount/{self.uid}')
        self.audio_files_ref = db.reference(f'AudioFiles/{self.uid}')
        self.internal_schedule = {}
        self.request_interval = 1  # 요청 간 1초 대기

        self.set_schedule_listener()
        self.set_status_listener()
        self.set_audio_listener()

        self.meal_cnt = 0
        self.medi_cnt = 0

        self.data_store = ScheduleStore().get_instance()
        self.localScheduler = LocalScheduler(self.uid).get_instance()



        # 자정에 카운터를 초기화하는 스케줄러 설정
        self.schedule_midnight_reset()




    def initialize_firebase_app(self):
        try:
            if not firebase_admin._apps:
                self.cred = credentials.Certificate("firebase-adminsdk.json")
                firebase_admin.initialize_app(self.cred, {
                    'databaseURL': 'https://projact01-82fe5-default-rtdb.firebaseio.com/'
                })
                print("파이어베이스 초기화")
            else:
                self.cred = firebase_admin.get_app().credential
                print("파이어베이스 초기화 안함")

            # google-auth 라이브러리를 사용하여 credentials 설정
            self.googlecred = service_account.Credentials.from_service_account_file("firebase-adminsdk.json")

        except Exception as e:
            print(f"파이어베이스 초기화 오류: {e}")

    @staticmethod
    def int64_to_int(obj):
        """JSON 직렬화를 위해 int64 타입을 int로 변환합니다."""
        if isinstance(obj, np.int64):
            return int(obj)
        return obj

    def loadData(self):
        try:
            if os.path.exists(self.save_file):
                with open(self.save_file, 'r') as file:
                    settings = json.load(file)
                    self.params = settings.get('params', self.params)
                    self.uid = settings.get('uid', self.uid)
                    saved_date = settings.get('todayweather', {}).get('date')
                    self.todayweather['date'] = saved_date if saved_date is not None else datetime.now().strftime('%Y%m%d')
                    print("Settings loaded from file.")
            else:
                print("No save file found. Getting new data.")
                self.uid = self.getUid(self.sn)
                region_codes = self.getRegionCode()
                if region_codes is not None:
                    self.params = region_codes
                self.saveData()
        except Exception as e:
            print(f"loadData error occurred: {e}")

    def saveData(self):
        try:
            settings = {
                'params': self.params,
                'uid': self.uid
            }
            with open(self.save_file, 'w') as file:
                json.dump(settings, file, default=Local.int64_to_int)
                print("Settings saved to file.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def is_quiet_time(self):
        current_time = datetime.now().time()
        current_state = self.checkCurrentState()
        return current_time >= datetime.strptime("22:00", "%H:%M").time() or current_time <= datetime.strptime("06:00", "%H:%M").time() or current_state == 4

    def process_sound(self, input_sound):
        if self.is_quiet_time():
            if input_sound in ['살려줘', '도와줘']:
                self.set_emergency()
            else:
                print("조용한 시간입니다. 소리를 내지 않습니다.")
        else:
            self.play_sound_through_speaker(input_sound)
            return input_sound

    def play_sound_through_speaker(self, sound):
        from STS.result1 import TextToSpeech
        self.sts = TextToSpeech().get_instance()
        self.sts.text_to_speech(sound)
        print(f"스피커에서 소리 출력: ")

    def printFileInfo(self, filename):
        date_str, time_str = filename.split('.')[0].split('_')
        date_time_str = f'{date_str} {time_str}'
        date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H-%M-%S')
        formatted_str = date_time_obj.strftime('%Y년 %m월 %d일 %p %I시 %M분 음성 녹음입니다')
        print(formatted_str)
        #self.task_manager.add_task(lambda: self.play_sound_through_speaker(formatted_str))


    def convert_mp3_ffmpeg(self, input_path, output_path):
        try:
            command = ['ffmpeg', '-i', input_path, output_path]
            subprocess.run(command, check=True)
            print(f"Converted {input_path} to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting audio with ffmpeg: {e}")

    def playAudio(self, audio):
        try:
            last_time_ref = db.reference(f'UserAccount/{self.uid}/lastTime')
            input_path = f"audio/{audio}"
            output_path = f"audio/converted_{audio}"

            # 변환된 파일이 없으면 변환 수행
            if not os.path.exists(output_path):
                self.convert_mp3_ffmpeg(input_path, output_path)

            pygame.mixer.init()

            # 파일이 존재하는지 확인
            if not os.path.exists(output_path):
                print(f"Audio file {output_path} does not exist.")
                return
            # 파일이 유효한지 확인
            if os.path.getsize(output_path) == 0:
                print(f"Audio file {output_path} is empty.")
                return

            # 파일 재생
            pygame.mixer.music.load(output_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            # 날짜 정보 추출 수정
            # 'converted_' 접두어와 파일 확장자를 제거합니다.
            filename_without_prefix = audio.replace("converted_", "")
            new_last_time = filename_without_prefix.split('.')[0]

            last_time_ref.set(new_last_time)  # 변경: update 대신 set 사용
        except pygame.error as e:
            print(f"Error playing audio: {e}")
    def SendRecentAudio(self):
        last_time_ref = db.reference(f'UserAccount/{self.uid}/lastTime')
        last_time_str = last_time_ref.get()
        print(f"Last time from DB: {last_time_str}")

        try:
            last_time = datetime.strptime(last_time_str, '%Y-%m-%d_%H-%M-%S')
        except ValueError:
            print(f"Invalid date format for 'lastTime': {last_time_str}")
            last_time = datetime.min  # 또는 적절한 기본값 설정

        new_files = []
        if os.path.exists(self.audio_folder):
            for file_name in os.listdir(self.audio_folder):
                full_path = os.path.join(self.audio_folder, file_name)
                if os.path.isfile(full_path):
                    # 파일 이름에서 'converted_' 접두사가 있으면 제거
                    normalized_file_name = file_name.replace('converted_', '') if file_name.startswith(
                        'converted_') else file_name
                    file_date_str = normalized_file_name.split('.')[0]
                    try:
                        file_date = datetime.strptime(file_date_str, '%Y-%m-%d_%H-%M-%S')
                        if file_date > last_time:
                            new_files.append(file_name)
                    except ValueError:
                        print(f"Error parsing date from filename {file_name}")

        print(f"New files found: {new_files}")
        return new_files

    def ProcessRecentAudio(self):  #외출 복귀나 일어난 후에 불러야하는 함수
        RecentAudioList = self.SendRecentAudio()
        if RecentAudioList:
                RecentAudioList.sort()
                for RecentAudio in RecentAudioList:
                    self.task_manager.add_task(lambda:self.playAudio(RecentAudio))



    async def RandomChoiceAudio(self):
        try:
            files = [os.path.join(self.audio_folder, f) for f in os.listdir(self.audio_folder) if
                os.path.isfile(os.path.join(self.audio_folder, f))]
            if files:
                chosen_file = random.choice(files)
                return chosen_file  # 선택된 파일의 경로 반환
            else:
                print("No audio files available to play.")
        except Exception as e:
            print(f"Error in checkLastestAudio: {e}")

    def fetchWeatherData(self):
        seoul_timezone = pytz.timezone('Asia/Seoul')
        curdate = datetime.now(seoul_timezone).strftime('%Y%m%d')

        print(curdate)
        print(self.todayweather)

        max_retries = 3
        attempt = 0
        while attempt < max_retries:
            try:
                print("날짜가 다르거나 데이터가 없습니다. 날씨 정보 갱신을 시도합니다.")
                self.params['base_date'] = curdate
                self.todayweather['date'] = curdate

                # API 키와 요청 매개변수 출력
                print(f"API Key: {self.params.get('serviceKey')}")
                print(f"Request Params: {self.params}")

                response = requests.get(self.BASEURL, params=self.params)
                response.raise_for_status()

                # 응답 텍스트 출력
                print("API 응답 텍스트:", response.text)

                # 응답이 비어있지 않은지 확인
                if response.text.strip() == "":
                    raise ValueError("API 응답이 비어 있습니다.")

                data = response.json()
                print("API 응답 데이터:", data)

                if 'response' in data and 'body' in data['response']:
                    items = data['response']['body']['items']['item']
                    print(f'item : {items}')
                    self.updateTodayWeather(items, curdate)
                    print("날씨 정보가 성공적으로 업데이트되었습니다.")
                    self.printTodayWeather()
                    break
                else:
                    print("API 응답 형식이 예상과 다릅니다. 응답 데이터를 확인하세요.")
                    print(data)
                    attempt += 1
                    time.sleep(self.request_interval)

            except ValueError as ve:
                print(f"오류 발생, 잠시 후 다시 시도합니다: {ve}")
                attempt += 1
                time.sleep(self.request_interval)
            except requests.exceptions.RequestException as re:
                print(f"오류 발생, 잠시 후 다시 시도합니다: {re}")
                attempt += 1
                time.sleep(self.request_interval)
            except Exception as e:
                print(f"오류 발생, 잠시 후 다시 시도합니다: {e}")
                attempt += 1
                time.sleep(self.request_interval)

        if attempt == max_retries:
            print("날씨 정보를 갱신할 수 없습니다. 나중에 다시 시도해주세요.")

    def updateTodayWeather(self, items, curdate):
        self.todayweather['date'] = curdate
        for item in items:
            category = item['category']
            if category == 'SKY':
                self.todayweather['sky'] = item['fcstValue']
            elif category == 'PTY':
                self.todayweather['precip'] = item['fcstValue']
            elif category == 'TMN':
                self.todayweather['lowtemp'] = item['fcstValue']
            elif category == 'TMX':
                self.todayweather['hightemp'] = item['fcstValue']

    def divisionWeather(self, category, fcst_time):
        for item in self.weatherdata:
            if item['category'] == category and item['fcstTime'] == fcst_time:
                return item['fcstValue']
        return None

    def printTodayWeather(self):
        from STS.result1 import TextToSpeech
        self.tts = TextToSpeech().get_instance()
        if None in self.todayweather.values():
            print("날씨 정보를 가져오지 못했습니다. 다시 시도해주세요.")
        else:
            weather_msg = f"현재 날씨는 {self.STATUS_OF_SKY[self.todayweather['sky']]}이고, 강수는 {self.STATUS_OF_PRECIPITATION[self.todayweather['precip']]}입니다."
            weather_msg += f"\n최고 기온은 {self.todayweather['hightemp']}도 이고, 최저 기온은 {self.todayweather['lowtemp']}도 입니다."
            print(weather_msg)
            return (weather_msg)

    def printAllForecastTimesForCategory(self, category):
        try:
            seoul_timezone = pytz.timezone('Asia/Seoul')
            curtime = datetime.now(seoul_timezone)
            self.params['base_date'] = curtime.strftime('%Y%m%d')
            self.params['pageNo'] = '1'
            response = requests.get(self.BASEURL, params=self.params)
            response.raise_for_status()
            data = response.json()
            items = data['response']['body']['items']['item']
            filtered_items = list(filter(lambda x: x['category'] == category, items))
            for item in filtered_items:
                print(f"Time: {item['fcstTime']}, Value: {item['fcstValue']}")
        except requests.exceptions.RequestException as err:
            print(f"Error occurred: {err}")

    def weather(self):
        self.fetchWeatherData()
        sky = self.divisionWeather('SKY', '0500')
        precip = self.divisionWeather('PTY', '0500')
        lowtemp = self.divisionWeather('TMN', '0600')
        hightemp = self.divisionWeather('TMX', '1500')
        if sky is None or precip is None or lowtemp is None or hightemp is None:
            print("날씨 정보를 가져오지 못했습니다. 다시 말씀해주세요")
        else:
            weather_msg = f"현재 날씨는 {self.STATUS_OF_SKY[sky]}이고, 강수는 {self.STATUS_OF_PRECIPITATION[precip]}입니다."
            weather_msg += f"\n최고 기온은 {hightemp}도 이고, 최저 기온은 {lowtemp}도 입니다."
            print(weather_msg)

    def checkCurrentState(self):
        user_data = self.state_ref.get()
        current_state = user_data.get('currentState')
        return current_state

    def checkStateQuestion(self):
        self.state_ref.update({
            'currentState': 1
        })

    def checkSTateQuestion1(self):
        current_state = self.checkCurrentState()
        if 1 <= current_state < 3:
            self.state_ref.update({
                'currentState': current_state + 1
            })

    async def checkLastCommunication(self):
        while True:
            response_time_ref = db.reference(f'UserAccount/{self.uid}/responseTime')
            response_time_str = response_time_ref.get()
            response_time = datetime.strptime(response_time_str, '%Y-%m-%d_%H-%M-%S')
            if datetime.now() - response_time > timedelta(hours=2):
                self.checkStateQuestion()
            await asyncio.sleep(60 * 30)

    # FirebaseManager의 메서드들
    def set_schedule_listener(self):
        ref = db.reference(f'UserSchedules/{self.uid}')
        ref.listen(self.schedule_listener)

    def set_status_listener(self):
        ref = db.reference(f'UserAccount/{self.uid}/currentState')
        ref.listen(self.status_listener)

    def set_audio_listener(self):
        ref = db.reference(f'AudioFiles/{self.uid}')
        ref.listen(self.audio_listener)

    def schedule_listener(self, event):
        print(f"Schedule update: {event.event_type} {event.path} {event.data}")
        if event.data is not None:
            pass
            #self.data_store.update_data(event.data)
            #self.localScheduler.check_schedules()
        else:
            print("Received None data, might be a deletion or error.")

    def status_listener(self, event):
        if event.data == 1:
            print("Status is 1, checking today's schedules.")

    def downloadAudio(self, filename):
        try:
            # google-auth credentials 사용하여 storage client 초기화
            storageclient = storage.Client(credentials=self.googlecred)
            bucket = storageclient.bucket(self.storagepath)
            if not os.path.exists(self.audio_folder):
                os.makedirs(self.audio_folder)
            blob = bucket.blob(f'audio/{filename}')
            localpath = os.path.join(self.audio_folder, filename)
            if not os.path.exists(localpath):
                blob.download_to_filename(localpath)
                print(f"Downloaded {blob.name} to {localpath}.")
                return localpath
            else:
                print(f"{filename} already exists in {self.audio_folder}.")
                return localpath
        except Exception as e:
            print(f"Error in downloadAudio: {e}")
            return None

    def audio_listener(self, event):
        try:
            print("audio_listener")
            file_info_dict = event.data
            if file_info_dict:
                print('file_info_dict : ', file_info_dict)

                # 단일 파일 정보만 처리
                if 'fileName' in file_info_dict and 'fileUrl' in file_info_dict:
                    file_name = file_info_dict.get('fileName', '')
                    print('file_name : ', file_name)
                    if file_name and not os.path.exists(f'audio/{file_name}'):
                        local_path = self.downloadAudio(file_name)
                        match = re.search(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})', file_name)
                        if match:
                            hour = int(match.group(4))
                            period = "오전" if hour < 12 else "오후"
                            hour = hour if hour < 12 else hour - 12
                            date_time_str = f"{match.group(2)}월 {match.group(3)}일 {period}{hour}시 {match.group(5)}분 녹음입니다."
                            print(date_time_str)

                            self.task_manager.add_task(lambda: self.play_sound_through_speaker(date_time_str))
                            self.task_manager.add_task(lambda: self.ProcessRecentAudio())  # 그래서 이건 뭐로 바뀐걸까?
                            #self.task_manager.add_task(lambda: self.checkLastestAudio())
                    else:
                        print("file_name already exists or is invalid.")
                else:
                    print("Ignoring multiple file info.")
            else:
                print("file_info_dict is None")
        except Exception as e:
            print(f"Error in audio_listener: {e}")

    def load_initial_data(self):
        ref = db.reference(f'UserSchedules/{self.uid}')
        schedules = ref.get()
        if schedules:
            self.data_store.update_data(schedules)

    def getUid(self, sn):
        try:
            userref = db.reference('UserAccount')
            query = userref.order_by_child('serialNumber').equal_to(sn).get()
            for uid, userinfo in query.items():
                if userinfo['serialNumber'] == sn:
                    return uid
            return None
        except Exception as e:
            print(f"loadData error occurred: {e}")
            return None

    def addMood(self, mood):
        curdate = datetime.now().strftime('%Y-%m-%d')
        moodref = db.reference(f'Mood/{self.uid}').push()
        moodref.set({
            'mood': mood,
            'date': curdate
        })

    def setMood(self, mood):
        curdate = datetime.now().strftime('%Y-%m-%d')
        moodref = db.reference(f'Mood/{self.uid}')
        moods = moodref.order_by_child('date').equal_to(curdate).get()
        if moods:
            for key, value in moods.items():
                moodref.child(key).update({'mood': mood})
        else:
            self.addMood(mood)

    def schedule_midnight_reset(self):
        now = datetime.now()
        midnight = datetime.combine(now.date(), datetime.min.time()) + timedelta(days=1)
        time_until_midnight = (midnight - now).total_seconds()
        threading.Timer(time_until_midnight, self.reset_counters).start()

    def reset_counters(self):
        base_path = f'UserAccount/{self.uid}/todoList'

        # 리셋할 경로와 값을 정의합니다.
        reset_values = {
            'meal': {
                'meal_evening': 0,
                'meal_morning': 0,
                'meal_noon': 0
            },
            'medicine': {
                'medicine_evening': 0,
                'medicine_morning': 0,
                'medicine_noon': 0
            },
            'quiz': {
                'quiz_completed': 0
            }
        }

        # Firebase 경로를 업데이트합니다.
        for category, values in reset_values.items():
            for subcategory, value in values.items():
                ref_path = f'{base_path}/{category}/{subcategory}'
                db.reference(ref_path).set(value)
                print(f'Set {ref_path} to {value}')

        self.meal_cnt = 0
        self.medi_cnt = 0
        self.schedule_midnight_reset()
        print("Counters reset to 0")

    def set_meal(self):
        try:
            if self.meal_cnt == 0:
                meal_ref = db.reference(f'UserAccount/{self.uid}/todoList/meal/morning')
            elif self.meal_cnt == 1:
                meal_ref = db.reference(f'UserAccount/{self.uid}/todoList/meal/noon')
            elif self.meal_cnt == 2:
                meal_ref = db.reference(f'UserAccount/{self.uid}/todoList/meal/evening')

            meal_ref.set(1)
            self.meal_cnt = (self.meal_cnt + 1) % 3  # 카운터를 증가시키고 0, 1, 2로 순환
            print(f"Meal status for {meal_ref.key} set to 1")

            # 설정된 값을 읽어오기
            new_meal_value = meal_ref.get()
            print(f"Updated Meal status: {new_meal_value}")
        except Exception as e:
            print(f"Failed to set meal status: {e}")

    def set_medi(self):
        try:
            if self.medi_cnt == 0:
                medi_ref = db.reference(f'UserAccount/{self.uid}/todoList/medicine/morning')
            elif self.medi_cnt == 1:
                medi_ref = db.reference(f'UserAccount/{self.uid}/todoList/medicine/noon')
            elif self.medi_cnt == 2:
                medi_ref = db.reference(f'UserAccount/{self.uid}/todoList/medicine/evening')

            medi_ref.set(1)
            self.medi_cnt = (self.medi_cnt + 1) % 3  # 카운터를 증가시키고 0, 1, 2로 순환
            print(f"Medicine status for {medi_ref.key} set to 1")

            # 설정된 값을 읽어오기
            new_medi_value = medi_ref.get()
            print(f"Updated Medicine status: {new_medi_value}")
        except Exception as e:
            print(f"Failed to set medicine status: {e}")

    def set_quiz(self):
        try:
            quiz_ref = db.reference(f'UserAccount/{self.uid}/todoList/quiz')
            quiz_ref.set(1)
            print("Quiz status set to 1")
        except Exception as e:
            print(f"Failed to set quiz status: {e}")

    def set_emergency(self):
        user_ref = db.reference(f'UserAccount/{self.uid}')
        user_ref.update({'emergency': 1})

    def add_alarm(self,text):
        date_pattern = re.compile(r'(\d{1,2})월\s*(\d{1,2})일')
        time_pattern = re.compile(r'(\d{1,2})시\s*(\d{1,2})?\s*분?')

        # 날짜와 시간 추출
        date_match = date_pattern.search(text)
        time_match = time_pattern.search(text)

        if date_match and time_match:
            month = int(date_match.group(1))
            day = int(date_match.group(2))
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0

            # 현재 연도를 사용하여 datetime 객체 생성
            current_year = datetime.now().year
            date = datetime(current_year, month, day, hour, minute)

            # '에' 뒤의 텍스트 추출
            remaining_text = text.split('에', 1)[-1].strip()
            schedule = remaining_text.split('약속')[0].strip()

            schedule_date = date.strftime('%Y-%m-%d')
            schedule_time = date.strftime('%H:%M')
            schedule_detail = schedule

            user_ref = db.reference(f'UserSchedules/{self.uid}')
            new_schedule_ref = user_ref.push()
            schedule_uid = new_schedule_ref.key

            schedule_data = {
                'date': schedule_date,
                'time': schedule_time,
                'text': schedule_detail,
                'id': schedule_uid
            }

            new_schedule_ref.set(schedule_data)


    def getMood(self, sn):
        ref = db.reference(f'Mood/{sn}')
        moods = ref.get()
        for moodid, moodinfo in moods.items():
            print(f"Date: {moodinfo['date']}, Mood: {moodinfo['mood']}")

    def getAddress(self):
        userref = db.reference(f'UserAccount/{self.uid}')
        userdata = userref.get()
        if userdata:
            province = userdata.get('province', '')
            city = userdata.get('city', '')
            district = userdata.get('district', '')
            self.address = f"{province} {city} {district}"
            print(self.address)
        else:
            self.address = None
    def select_random_quizzes(self,num_quizzes=2):
            quiz_ref = db.reference('quiz/quiz')
            quiz_data = quiz_ref.get()
            keys = list(quiz_data.keys())
            selected_keys = random.sample(keys, num_quizzes)
            selected_quizzes = [[quiz_data[key]['question'], quiz_data[key]['answer']] for key in selected_keys]
            return selected_quizzes

    def getRegionCode(self):
        try:
            if self.address is None:
                self.getAddress()
            addressparts = self.address.split()
            df = pd.read_excel(self.excelpath)
            matchingrows = df[
                (df['1단계'] == addressparts[0]) &
                (df['2단계'] == addressparts[1]) &
                (df['3단계'] == addressparts[2])
            ]
            if not matchingrows.empty:
                return {
                    'nx': matchingrows['격자 X'].values[0],
                    'ny': matchingrows['격자 Y'].values[0]
                }
        except Exception as e:
            print(f"getRegionCode error: {e}")
        return None

"""if __name__ == '__main__':
    local = Local().get_instance()
    local.fetchWeatherData()
"""
