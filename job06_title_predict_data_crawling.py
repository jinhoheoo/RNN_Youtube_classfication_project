from selenium import webdriver as wb
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs
import re
import random
import time
import pandas as pd
import datetime

# f-string
keywords = ['music','음악','game','게임','sports','스포츠','cook','요리','pets','애완동물','nature','자연']
key_count = 0
df_titles = pd.DataFrame()

titleList = []

for keyword in keywords:

    yt_url = f'https://www.youtube.com/results?search_query={keyword}10'
    driver = wb.Chrome()
    try:
        driver.get(yt_url)
    except:
        print('drivet.get', keyword)

    # 브라우저 로드가 완료되기 위한 시간
    time.sleep(2)


    # selenium을 이용해서 HTML문서를 변환한 후에는 반드시 브라우저를 종료해야 한다!
    html = bs(driver.page_source, 'html.parser')
    #print(html)

    driver.close()

    for content in html.select('a#video-title'):
        title = content.get('title')
        title = re.compile('[^가-힣a-zA-Z]').sub(' ', title)


    # 유튜브 내용을 저장할 딕셔너리 생성
    key_label_num =2*int(key_count/2)

    if key_count % 2 == 1:


        df_section_title = pd.DataFrame(titleList, columns=['titles'])
        df_section_title['category'] = keywords[key_label_num]
        df_titles = pd.concat([df_titles, df_section_title], axis='rows', ignore_index=True)
        titleList=[]
    key_count += 1

df_titles.to_csv('./data/data_exam_{}.csv'.format( datetime.datetime.now().strftime('%Y%m%d')), index=False)
