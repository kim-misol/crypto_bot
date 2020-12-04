print("\n*************************************ex1*************************************")
# Dictionary는 "키(Key) - 값(Value)" 쌍을 요소로 갖는 컬렉션이다
company = {'삼성전자': '005930', 'LG전자': '066570'}

print(f"company: {company} , type: {type(company)}")
# 특정 key값의 value 출력
print("삼성전자 : ", company['삼성전자'])

# key, value 추가
company['SK텔레콤'] = '017670'
print(f"company: {company} , type: {type(company)}")

print("\n*************************************ex2*************************************")
# for문에 활용
for key in company:
    val = company[key]
    print(f"key : {key}, val : {val}")

print("\n*************************************ex3*************************************")
# key 값 가져오기
keys = company.keys()
for k in keys:
    print(k)

# values 값 가져오기
values = company.values()
for v in values:
    print(v)

print("\n*************************************ex4*************************************")
# DataFrame ? => 행렬을 저장
# pandas라는 모듈의 DataFrame 클래스를 import
from pandas import DataFrame

company = {
    'code': ('005930', '066570', '017670'),
    'code_name': ('삼성전자', 'LG전자', 'SK텔레콤'),
    'close': ('55200', '71000', '234000')
}
print(company['code'])

df_company = DataFrame(company)

print(f"type(df_company): {type(df_company)} ")
print(f"len: {len(df_company)}")

print("\n*************************************ex5*************************************")
'''
iloc : 행, 열 단위로 DataFrame 안에 있는 데이터에 접근
'''
# 아래는 0행 0열과 0행 1열을 출력
print(f"df_company.iloc[0, 0] :{df_company.iloc[0, 0]}, df_company.iloc[0, 1] : {df_company.iloc[0, 1]}")

print("\n*************************************ex6*************************************")
'''
loc:  iloc와 동일하게 DataFrame에 행, 열로 접근하는데 열의 경우 라벨명으로 접근.
'''
# 아래는 0행 'code_name' 열과 0행 'code'열을 출력

print(
    f"df_company.loc[0, 'code'] :{df_company.loc[0, 'code']}, df_company.loc[0, 'code_name'] : {df_company.loc[0, 'code_name']}")

# 위와 같이 코드가 길어지면 아래 처럼 수정 가능
print(f'''df_company.loc[0, 'code'] :{df_company.loc[0, 'code']}, 
df_company.loc[0, 'code_name'] : {df_company.loc[0, 'code_name']}''')

print("\n*************************************ex7*************************************")
# 칼럼 순서 변경 가능
company2 = {
    'code': ('005930', '066570', '017670'),
    'code_name': ('삼성전자', 'LG전자', 'SK텔레콤'),
    'close': ('55200', '71000', '234000')
}
df_company2 = DataFrame(company2, columns=['close', 'code', 'code_name'])
print(f"df_company2 : {df_company2}")
