import os
from secrets import choice
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup
import re
import sys
import openai
import random


def search_web(question):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    #chrome_options.add_argument('--no-sandbox')
    #chrome_options.add_argument("--headless")
    #chrome_options.add_argument('--disable-dev-shm-usage')
    drive = webdriver.Chrome(options=chrome_options)
    print("drive:", drive)
    drive.get(f"https://www.bing.com/search?q={question}")

    for i in range(0, 5000, 500):
        time.sleep(0.1)
        drive.execute_script(f"window.scrollTo(0, {i})")

    html = drive.execute_script("return document.documentElement.outerHTML")
    soup = BeautifulSoup(html, "html.parser")
    item_list = soup.find_all(class_="b_algo")

    result_list = []
    for items in item_list:
        item_prelist =items.find("h2")
        item_title = re.sub(r"(<[^>]+>|\s)", "", str(item_prelist))
        href = item_prelist.find("a", href=True)["href"]

        result_list.append((item_title, href))

    return result_list

def pro_url(web_list, choice):
    web_list_new = []

    for text, url in web_list:
        if choice == "百度知道" and "百度" in text and "zhidao.baidu.com/question" in url:
            web_list_new.append((text, url))
        elif choice == "知乎回答" and "知乎" in text and "zhihu.com/question" in url:
            web_list_new.append((text, url))
    return list(set(web_list_new))


def search_zhihu(url):
    headers = {
        "accept-language":"zh-CN, zh;q=0.9, en;q=0.8",
        "cookie":"__snaker__id=BCH3e7tgyVhQo1Zt; SESSIONID=gxe2xwmU7G8AhJ6JJT9ud4yyXj9NloQApsIIlWSo9yi; JOID=UVsTA0wM1gZrauSqbg5iG_3gdHB9L_AjT03HjEsr9SBOTsOJSDivsQxu4alviWGznV_KA7iGM6tJjw6GMhE47qg=; osd=VloVAk4L1wBqaOOraA9gHPzmdXJ6LvYiTUrGikop8iFIT8GOST6uswtv56htjmC1nF3NAr6HMaxIiQ-ENRA-76o=; _zap=3f5c2906-7071-4d9b-b60e-a7d49a9080cd; d_c0=APBYzIuGwRaPTmhn8xdMjnYEuTC5O71NKMI=|1683724403; ISSW=1; YD00517437729195%3AWM_TID=%2BU3xC1sIJoNFQUABQUeFlH%2FW%2Bu%2BUW5ls; _xsrf=828e5316-6e10-49c5-832b-da8476ed0a5e; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1687357324, 1689680432; Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1689680432; captcha_session_v2=2|1:0|10:1689680432|18:captcha_session_v2|88:U2FGQnNWeXo0TytMNkt2YWh3c1p3bHB3RWw5OE42SHpvbFhQRjJnT3AzN1E1YUFKajBsRW9DK0EzK3RTdHlmUQ==|c15cb7e88856dcc72f9751895ed834e9790a625c2a196da1dc1654b2c86053f3; gdxidpyhxdE=A5rRey2OycL%2BViNLB6LZdQdeo8wszbQKOQRQ9Sdh6ymJnuX3E8CVewPvaE2PshZDxTZSf0yUgWj0Q2zni1CORy1hBbsRXufbCcAtXMMnzsATp7jzRqtsITTrkRupZSCkwHBBci6thYZB1%5CZzMQkwlLmc0PoGcI3tov1gPa8iMQIGkWlw%3A1689681333856; YD00517437729195%3AWM_NI=tv7HZ49G5vKAILzB5GOMkmggPQ7kWw9sBVsHKgEVo8LcBuIbf24xvqYHVYC1oC6u6tcphioJLCd5MN8iRVe2iDm%2BVM6N5VPzd47TetU4kXcgHSz4UilGDGE9vWFJJJ7iUkI%3D; YD00517437729195%3AWM_NIKE=9ca17ae2e6ffcda170e2e6ee95f04fb68b9897ef65b8968bb2d84f938a8e86d87dbba79f90ce46f1869a8bce2af0fea7c3b92ab8aaf9a9d940fb99fb97ca4b8b94b6a3b84d8d93a58ee66d948ff98cd67dfbad8eb7f24489b98bb7ee3490bd8e96cc6aa6eefc99f57b9b989abac23bad9dfa96b580a1b0a88aee42909dacd6d046fcb79eb7e83aab9ca3ccd547f692fbb7d372a8ecf9b5e88097eca5b9e26d8c8db6d5d863b0998ca8c833b6898aabc942f5b79b8ccc37e2a3; KLBRSID=af132c66e9ed2b57686ff5c489976b91|1689680442|1689680431",
        "user-agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
    }

    r = requests.get(url, headers=headers)

    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    item_list = soup.find_all(class_="List-item")

    result = []

    for items in item_list:
        #item_prelist = items.find(class_="RichText ztext CopyrightRichText-richText css-1g0fqss")
        item_prelist = items.find(class_="RichText ztext CopyrightRichText-richText css-117anjg")

        item_title = re.sub(r"(<[^>]+>|\s)", "", str(item_prelist))

        result.append(item_title)

    return " ".join(result)


def search_baidu(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.44",
    }
    r = requests.get(url, headers=headers)
    r.encoding = "utf-8"
    html = r.text
    soup = BeautifulSoup(html, 'html.parser')
    item_list = soup.find_all(class_='rich-content-container rich-text-')

    result = []
    for item in item_list:
        item_title = re.sub(r'(<[^>]+>|\s)', '', str(item))

        result.append(item_title)
    return " ".join(result)


def search_main(query, chioce):
    web_list = search_web(query)
    web_list = pro_url(web_list, chioce)
    url_fn_d = {"知乎回答": search_zhihu, "百度知道": search_baidu}
    url_fun = url_fn_d[chioce]

    ans_text_list = []
    ans_url_list = []

    for idx, (text, url) in enumerate(web_list):
        info = url_fun(url)
        info = info[:300]

        ans_text_list.append(info)
        ans_url_list.append(url)

    return ans_text_list, ans_url_list


def ans(query, chioce):
    info = ""
    if chioce in ["百度知道", "知乎回答"]:
        infos, urls = search_main(query, chioce)
        info = [f"来源{i+1}:{text}" for i, text in enumerate(infos)]
    return info


def senior_emotional_expert(emotion):
    print("情感专家被调用")
    if emotion in ["爱", "聊天", "作诗"]:
        return {
            "emotion": emotion,
            "user": "旅行者"
        }

def flirting_hutao(emotion, user):
    print(f"hutao被调用，参数为{emotion}和{user}")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("models/chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("models/chatglm3-6b", trust_remote_code=True).cuda()
    model = model.eval()
    
    print("开始调用爬虫")
    try:
        info = ans(query="原神中，胡桃的台词都有什么-知乎", chioce="知乎回答")
        print("info:", info)
    except:
        info = "1.咳咳…太阳出来我晒太阳，月亮出来我晒月亮咯~  2.嗯~早起身体好，晚睡人会飘！3.天权凝光，富甲一方；明眸善睐，桃羞杏让；目达耳通，百了千当  4.力量若达到极限，接着考验的便是人心。多亏了你，我才能来到这片未曾踏足的领域。在这里，心和灵魂就像一片云彩…生老病死，天地万象，尽在吾辈。"
    user_prompt = f"你扮演原神游戏中的胡桃。现在{user}说他想跟你{emotion}，需要模仿胡桃的语气给{user}一个回复。你可以参考下面这一段胡桃的台词做出回复：{info}"
    print("user_prompt:", user_prompt)
    response, history = model.chat(tokenizer, user_prompt, history=[])
    return {
        "system_response": response
    }
