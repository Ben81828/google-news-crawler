import urllib.request
from bs4 import BeautifulSoup as Soup
import datetime
from dateutil import tz
from time import sleep
from typing import List
import pandas as pd

### CLASSEs
class GoogleNews:

    def __init__(self, account:str, password:str, proxy_list=List[str],lang="en", period="", encode="utf-8", region=None):
        self.account = account
        self.password = password
        self.proxy_list = proxy_list
        self.user_agent = 'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:64.0) Gecko/20100101 Firefox/64.0'
        self.__lang = lang
        # 若user提供region，則headers加入accept-language，否則不加
        if region:
            self.accept_language= lang + '-' + region + ',' + lang + ';q=0.9'
            self.headers = {'User-Agent': self.user_agent, 'Accept-Language': self.accept_language}
        else:
            self.headers = {'User-Agent': self.user_agent}
        self.__period = period
        self.__encode = encode
        self.proxy = self.proxy_check(proxy_list=self.proxy_list)
        self.__results = self.result_df_init()

    def response_builder(self, url:str, proxy:str, is_google_news_search:bool=False):
        """
        url: 要連線的網址
        proxy: 要使用的proxy
        將proxy設置於opener，並回傳req
        is_google_news_search: 搜尋goolge news時要設為True
        """
        if is_google_news_search:
            self.req = urllib.request.Request(self.url.replace("search?","search?hl="+self.__lang+"&gl="+self.__lang+"&"), headers=self.headers)
        else:
            self.req = urllib.request.Request(url, headers=self.headers)

        self.proxy = proxy
        self.proxy_support = urllib.request.ProxyHandler(
                            {
                                'http':f'http://{self.account}:{self.password}@{proxy}', 
                                'https':f'https://{self.account}:{self.password}@{proxy}',
                            })

        opener = urllib.request.build_opener(self.proxy_support)
        urllib.request.install_opener(opener)
        self.response = urllib.request.urlopen(self.req, timeout=3)

        return self.response

    def proxy_check(self, proxy_list:List[str], url:str="", retries:int=3):
        """
        proxy_list: 一連串可能可用的proxy串列
        url:測試並建立連線的網址
        有時公司個別proxy會失效
        此函式讀取所有可能可用的proxy，一一嘗試連線，並回傳第一個可連線的proxy

        """
        test_url = "https://www.google.com/?hl=zh_TW"
        if not url:
            url = test_url
        self.work_proxies = []
         # 每個proxy_list中的proxy嘗試次數
        print('proxy check')
        for proxy in proxy_list:
            print('proxy= ', proxy, end='')
            for attempt in range(retries):
                if self.work_proxies==[]:
                    try:
                        self.response = self.response_builder(url=url, proxy=proxy, is_google_news_search=False)
                        self.work_proxies.append(proxy)
                        print(" ok")
                        # 成功連線
                        # self.response.close()
                    except Exception as e:
                        print(f' failed (Exception: {e}')
                        sleep(3)

        if not self.work_proxies: 
            print("目前並無有效proxy")
        else:
            self.proxy = self.work_proxies[0]
            return self.proxy
        
    def url_content_getter(self, url):

        try:
            self.response = self.response_builder(url=self.url, proxy=self.proxy, is_google_news_search=True)
        except Exception as e:
            print(e)
            self.response = None
            self.proxy = self.proxy_check(proxy_list=self.proxy_list, url=url)
        
        if self.response is not None:
            self.page = self.response.read()
            self.content = Soup(self.page, "html.parser")
            self.response.close()
            return self.content
        else:
            print(f"搜尋時連線錯誤")
            return None

    def utc_to_taipei(self, date:str):
        """
        date: utc格式時間字串，如"2023-02-04T23:10:10Z"
        將utc時間轉換為台北時間
        """
        try:
            utc = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')
            from_zone = tz.gettz('UTC')
            to_zone = tz.gettz('Asia/Taipei')
            utc = utc.replace(tzinfo=from_zone)
            taipei = utc.astimezone(to_zone)
            taipei_strtime = taipei.strftime("%Y-%m-%d %H:%M:%S")
            return taipei_strtime
        except Exception as e:
            print(e)
            return None
        
    def result_df_init(self):
        results_col = ['title', 'desc', 'date', 'datetime', 'link', 'img', 'media', 'site', 'keyword', 'category']
        self.__results = pd.DataFrame(columns=results_col)
        return self.__results

    def get_news(self, key:str="", key_catogory:str="", deamplify:bool=False):
        """
        key: 搜尋關鍵字
        key_catogory: 搜尋關鍵字的類別
        deamplify: 是否要解除google news的網址縮寫
        依據關鍵字去google news搜尋，並解析搜尋結果，彙整至__results
        """
        my_key = key

        if self.__encode != "":
            key = urllib.request.quote(key.encode(self.__encode))

        if key == "":
            self.url = 'https://news.google.com/?hl={}'.format(self.__lang)
        else:
            key = "+".join(key.split(" ")) 
            self.url = 'https://news.google.com/search?q={}+when:{}&hl={}'.format( key, self.__period, self.__lang.lower())

    
        # retries = 3 # 嘗試連線次數
        # for attempt in range(retries):
        #     try:
        #         self.response = self.response_builder(url=self.url, proxy=self.proxy, is_google_news_search=True)
        #         self.page = self.response.read()
        #         self.content = Soup(self.page, "html.parser")
        #         self.response.close()    
        #         break
        #     except Exception as e:  # 連線錯誤，articles尚未迭帶
        #         print(e)
        #         print(f"搜尋時連線錯誤: 關鍵字-{my_key}(等待10秒後重試)")
        #         sleep(10)
        #         self.response = None
        #         self.proxy = self.proxy_check(proxy_list=self.proxy_list)

        # if self.response is None:
        #     print(f"搜尋時連線錯誤: 停止搜關鍵字-{my_key}")
        #     return

        content = self.url_content_getter(url=self.url)
        if content is None:
            return
        
        else:           
            # 開始解析文章
            articles = self.content.select('div[class="NiLAwe y6IFtc R7GTQ keNKEd j7vNaf nID9nc"]')
            for article in articles:
                try:

                    # title
                    try:
                        title=article.find('h3').text
                    except:
                        title=None

                    # description
                    try:
                        desc=article.find('span').text
                    except:
                        desc=None

                    # date
                    try:
                        date = article.find("time").text
                    except:
                        date = None

                    # datetime
                    try:
                        datetime=article.find('time').get('datetime')
                        datetime_chars=self.utc_to_taipei(datetime)
                    except:
                        datetime_chars=None 

                    # link
                    if not deamplify:
                        link = 'news.google.com/' + article.find("h3").find("a").get("href")
                    else:
                        try:
                            link = 'news.google.com/' + article.find("h3").find("a").get("href")
                        except Exception as deamp_e:
                            print(deamp_e)
                            link = article.find("article").get("jslog").split('2:')[1].split(';')[0]

                    if link.startswith('https://www.youtube.com/watch?v='):
                        desc = 'video'

                    # image
                    try:
                        img = article.find("img").get("src")
                    except:
                        img = None

                    # site
                    try:
                        site=article.select('a[class*="wEwyrc"]')[0].text
                    except:
                        site=None

                    # collection
                    data = [title, desc, date, datetime_chars, link, img, None, site, my_key, key_catogory]
                    self.__results.loc[len(self.__results)] = data
                
                    
                # 解析錯誤
                except Exception as e_article:                   
                    print(e_article, f"{my_key}一篇文章解析錯誤")
                    return None

            return self.__results
        
    def results( self, sort:bool=False) -> pd.DataFrame:
        """
        呼叫後Return __results.
        回傳的__results會依照datetime欄位排序
        sort: True為ascending, False為descending
        """
        results=self.__results
        if sort:
            try:
                results=results.sort_values(by='datetime',ascending=False)
            except:
                print("No datetime column")
        return results

