import os
import logging
import pathlib
from logging.handlers import TimedRotatingFileHandler

# 목적
# log파일 위치와 로그 이름을 설정한다.
file_path = pathlib.Path(__file__).parent.parent.absolute() / 'train_log' / 'crypto_bot.log'
os.makedirs(file_path.parents[0], exist_ok=True)  # 로그 폴더가 존재하는지 확인 후 없으면 생성

# logger instance 생성
logger = logging.getLogger(__name__)

# logger instance로 로그찍기
# 로그레벨 순서 debug > info > warning > error > critical
# setLevel 에서 logger. + 대문자 DEBUG, INFO, WANRING, ERROR, CRITICAL 옵션 설정
# print 를 logger.debug로 모두 대체 해서 사용하면 logger 출력 위치를 알 수 있기 때문에 도움이 됨
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
file_handler = TimedRotatingFileHandler(file_path, when="midnight", encoding='utf-8')

# formmater 생성
formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
file_handler.suffix = "%Y%m%d"

# logger instance에 handler 설정
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
