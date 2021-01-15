from trading_calendars import get_calendar

kr_calendar = get_calendar('XKRX')
print(kr_calendar)

# 장이 열린 일자 
open_days = kr_calendar.all_sessions
print(str(open_days[-1])[:10])
# 장이 열린 일자의 분 데이터
all_minutes = kr_calendar.all_minutes
