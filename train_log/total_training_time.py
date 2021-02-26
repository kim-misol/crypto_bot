'''
모델학습에 걸린 총 시간
'''

from pathlib import Path
import time
import statistics


cwd = Path.cwd()
print(cwd)
files = list(cwd.glob('*'))
print(files)

fname_list = []
acc_list = []
# 특정 종목의 모델 학습 변수 별 예측률 비교

for file in files:
    fname = str(file)
    if '학습 로그' in fname:
        text = file.read_text()
        progress_bar = "[==============================] - "
        line_list = text.split(progress_bar)[1:]

        i, s = 0, 0
        for line in line_list:
            s += int(line[:line.index(" ") - 1])
            i += 1

        print(fname, f"\t{i} epochs\t {s} seconds")
        if s >= 86400:
            d = int(s / 86400)
            total_time = f"{time.strftime('%H:%M:%S', time.gmtime(s))} + {d} day(s)"
            print(total_time)
        else:
            total_time = time.strftime('%H:%M:%S', time.gmtime(s))
            print(total_time)


        # # 마지막
        # last_line = line_list[-1]
        # acc_str = last_line[last_line.index('val_accuracy: ') + len('val_accuracy: '):]
        # acc = float(acc_str.split("\n")[0]) * 100

        acc_text = "val_accuracy:"
        line_list = text.split(acc_text)[1:]

        i, acc_list = 0, []
        for line in line_list:
            acc_list.append(float(line.split("\n")[0]) * 100)
        print(statistics.mean(acc_list), "\n")

