### MODULES
import warnings
warnings.filterwarnings("ignore")
import urllib.request
from bs4 import BeautifulSoup as Soup, ResultSet
import datetime
from dateutil import tz
from time import sleep
import pandas as pd
import hanzidentifier
import pyshorteners
import re
import requests
# 進度條
from tqdm import tqdm

### 分詞用 ###
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

### 模型pipeline內物件 ###
# TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

# LSA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# LSI
from gensim import corpora, models

# Model
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

### 讀取模型用 ###
import dill

# LSA 降維(TruncatedSVD、正規化)
class LSA:
    def __init__(self, n_components=300):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=self.n_components)
        self.normalizer = Normalizer(copy=False)
    
    def fit(self, X, y=None):
        self.X_lsa = make_pipeline(self.svd, self.normalizer).fit(X)
        return self
        
    def transform(self, X, y=None):
        return self.X_lsa.transform(X)

# LSI 找潛在議題，並用議題得分當特徵
class LSI:
    def __init__(self, num_topics=20):
        self.num_topics = num_topics
    
    def fit(self, X, y=None):
        X = X.str.split()
        self.dictionary = corpora.Dictionary(X)
        
        # 轉為 BOW，然後字元長度不超過1的刪掉
        self.corpus = [self.dictionary.doc2bow(token) for token in X if len(token) > 1]
        
        # 建模，num_topics=x，即x個議題
        self.lsi = models.LsiModel(self.corpus, id2word=self.dictionary, num_topics=self.num_topics)

        return self

    def transform(self, X, y=None):

        X = X.str.split()
        
        # 用句子去計算該句子在各議題的分數，用各議題的分數作為特徵
        def word_list_to_group_score(token_list:list):
            group_score_list = self.lsi[self.dictionary.doc2bow(token_list)]
            if group_score_list:
                group_score_unzip = list(zip(*group_score_list))
                groups = group_score_unzip [0]
                scores = group_score_unzip [1]
                return pd.Series(scores, index=[f'group_{group}' for group in groups])
            else:
                return pd.Series([0 for _ in range(self.num_topics)], index=[f'group_{group}' for group in range(self.num_topics)])

        return X.apply(word_list_to_group_score)


# 主要程式: 開始爬蟲和預測(用來讓control.py import用)
def crawl_and_predict():

    ### CLASSEs
    class GoogleNews:

        def __init__(self,account, password, proxy_list,lang="en",period="",start="",end="",encode="utf-8",region=None):
            self.account = account
            self.password = password
            self.proxy_list = proxy_list
            self.__texts = []
            self.__links = []
            self.__results = []
            self.__totalcount = 0
            self.user_agent = 'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:64.0) Gecko/20100101 Firefox/64.0'
            self.__lang = lang
            if region:
                self.accept_language= lang + '-' + region + ',' + lang + ';q=0.9'
                self.headers = {'User-Agent': self.user_agent, 'Accept-Language': self.accept_language}
            else:
                self.headers = {'User-Agent': self.user_agent}
            self.__period = period
            self.__start = start
            self.__end = end
            self.__encode = encode
            self.__version = '1.6.2'
            self.proxy_works = []
            self.proxy = self.proxy_check(proxy_list=self.proxy_list, headers=self.headers, auo_account=self.account, auo_password=self.password)

        def proxy_check(self, proxy_list, headers, auo_account, auo_password):
            url = "https://www.google.com/?hl=zh_TW"
            retries = 3
            print('proxy check')
            for proxy in proxy_list:
                print('proxy= ', proxy, end='')
                req = urllib.request.Request(url, headers=headers)
                # set proxy
                proxy_support = urllib.request.ProxyHandler(
                        {
                            'http':f'http://{auo_account}:{auo_password}@{proxy}', 
                            'https':f'https://{auo_account}:{auo_password}@{proxy}',
                        })
                opener = urllib.request.build_opener(proxy_support)
                urllib.request.install_opener(opener)
                for attempt in range(retries):
                    try:
                        response = urllib.request.urlopen(req, timeout=2)
                        # 成功連線
                        response.close()
                        print(" ok")
                        self.proxy_works.append(proxy)
                        break
                    except Exception as e:
                        print(f' failed (Exception: {e}')

            if self.proxy_works == []: print("無有效proxy，請稍後再嘗試")
            assert self.proxy_works != [], "無有效proxy，請稍後再嘗試"

            return self.proxy_works[0]

        def getVersion(self):
            return self.__version

        def set_lang(self, lang):
            self.__lang = lang

        def setlang(self, lang):
            """Don't remove this, will affect old version user when upgrade"""
            self.set_lang(lang)

        def set_period(self, period):
            self.__period = period

        def setperiod(self, period):
            """Don't remove this, will affect old version user when upgrade"""
            self.set_period(period)

        def set_time_range(self, start, end):
            self.__start = start
            self.__end = end

        def setTimeRange(self, start, end):
            """Don't remove this, will affect old version user when upgrade"""
            self.set_time_range(start, end)

        def set_encode(self, encode):
            self.__encode = encode

        def setencode(self, encode):
            """Don't remove this, will affect old version user when upgrade"""
            self.set_encode(encode)

        def search(self, key):
            """
            Searches for a term in google.com in the news section and retrieves the first page into __results.
            Parameters:
            key = the search term
            """
            self.__key = "+".join(key.split(" "))
            if self.__encode != "":
                self.__key = urllib.request.quote(self.__key.encode(self.__encode))
            self.get_page()

        def build_response(self):
            self.req = urllib.request.Request(self.url.replace("search?","search?hl="+self.__lang+"&gl="+self.__lang+"&"), headers=self.headers)
            # set proxy
            proxy_support = urllib.request.ProxyHandler(
                    {'http':f'http://{auo_account}:{auo_password}@{proxy}', 
                        'https':f'https://{auo_account}:{auo_password}@{proxy}'})
            opener = urllib.request.build_opener(proxy_support)
            urllib.request.install_opener(opener)
            self.response = urllib.request.urlopen(self.req)
            self.page = self.response.read()
            self.content = Soup(self.page, "html.parser")
            stats = self.content.find_all("div", id="result-stats")
            if stats and isinstance(stats, ResultSet):
                stats = re.search(r'[\d,]+', stats[0].text)
                self.__totalcount = int(stats.group().replace(',', ''))
            else:
                #TODO might want to add output for user to know no data was found
                return
            result = self.content.find_all("div", id="search")[0].find_all("g-card")
            return result

        def page_at(self, page=1):
            """
            Retrieves a specific page from google.com in the news sections into __results.

            Parameter:
            page = number of the page to be retrieved
            """
            results = []
            try:
                if self.__start != "" and self.__end != "":
                    self.url = "https://www.google.com/search?q={}&lr=lang_{}&biw=1920&bih=976&source=lnt&&tbs=lr:lang_1{},cdr:1,cd_min:{},cd_max:{},sbd:1&tbm=nws&start={}".format(self.__key,self.__lang,self.__lang,self.__start,self.__end,(10 * (page - 1)))
                elif self.__period != "":
                    self.url = "https://www.google.com/search?q={}&lr=lang_{}&biw=1920&bih=976&source=lnt&&tbs=lr:lang_1{},qdr:{},,sbd:1&tbm=nws&start={}".format(self.__key,self.__lang,self.__lang,self.__period,(10 * (page - 1))) 
                else:
                    self.url = "https://www.google.com/search?q={}&lr=lang_{}&biw=1920&bih=976&source=lnt&&tbs=lr:lang_1{},sbd:1&tbm=nws&start={}".format(self.__key,self.__lang,self.__lang,(10 * (page - 1))) 
            except AttributeError:
                raise AttributeError("You need to run a search() before using get_page().")
            try:
                result = self.build_response()
                for item in result:
                    try:
                        tmp_text = item.find("div", {"role" : "heading"}).text.replace("\n","")
                    except Exception:
                        tmp_text = ''
                    try:
                        tmp_link = item.find("a").get("href")
                    except Exception:
                        tmp_link = ''
                    try:
                        tmp_media = item.findAll("g-img")[0].parent.text
                    except Exception:
                        tmp_media = ''
                    try:
                        tmp_date = item.find("div", {"role" : "heading"}).next_sibling.findNext('div').text
                        tmp_date,tmp_datetime=lexical_date_parser(tmp_date)
                    except Exception:
                        tmp_date = ''
                        tmp_datetime=None
                    try:
                        tmp_desc = item.find("div", {"role" : "heading"}).next_sibling.text
                    except Exception:
                        tmp_desc = ''
                    try:
                        tmp_img = item.findAll("g-img")[0].find("img").get("src")
                    except Exception:
                        tmp_img = ''
                    self.__texts.append(tmp_text)
                    self.__links.append(tmp_link)
                    results.append({'title': tmp_text, 'media': tmp_media,'date': tmp_date,'datetime':define_date(tmp_date),'desc': tmp_desc, 'link': tmp_link,'img': tmp_img})
                
            except Exception as e_parser:
                print(e_parser)
                pass
            
            finally:
                self.response.close()
            
            return results

        def get_page(self, page=1):
            """
            Retrieves a specific page from google.com in the news sections into __results.

            Parameter:
            page = number of the page to be retrieved 
            """
            try:
                if self.__start != "" and self.__end != "":
                    self.url = "https://www.google.com/search?q={}&lr=lang_{}&biw=1920&bih=976&source=lnt&&tbs=lr:lang_1{},cdr:1,cd_min:{},cd_max:{},sbd:1&tbm=nws&start={}".format(self.__key,self.__lang,self.__lang,self.__start,self.__end,(10 * (page - 1)))
                elif self.__period != "":
                    self.url = "https://www.google.com/search?q={}&lr=lang_{}&biw=1920&bih=976&source=lnt&&tbs=lr:lang_1{},qdr:{},,sbd:1&tbm=nws&start={}".format(self.__key,self.__lang,self.__lang,self.__period,(10 * (page - 1))) 
                else:
                    self.url = "https://www.google.com/search?q={}&lr=lang_{}&biw=1920&bih=976&source=lnt&&tbs=lr:lang_1{},sbd:1&tbm=nws&start={}".format(self.__key,self.__lang,self.__lang,(10 * (page - 1))) 
            except AttributeError:
                raise AttributeError("You need to run a search() before using get_page().")
            try:
                result = self.build_response()
                for item in result:
                    try:
                        tmp_text = item.find("div", {"role" : "heading"}).text.replace("\n","")
                    except Exception:
                        tmp_text = ''
                    try:
                        tmp_link = item.find("a").get("href")
                    except Exception:
                        tmp_link = ''
                    try:
                        tmp_media = item.findAll("g-img")[0].parent.text
                    except Exception:
                        tmp_media = ''
                    try:
                        tmp_date = item.find("div", {"role" : "heading"}).next_sibling.findNext('div').text
                        tmp_date,tmp_datetime=lexical_date_parser(tmp_date)
                    except Exception:
                        tmp_date = ''
                        tmp_datetime=None
                    try:
                        tmp_desc = item.find("div", {"role" : "heading"}).next_sibling.text
                    except Exception:
                        tmp_desc = ''
                    try:
                        tmp_img = item.findAll("g-img")[0].find("img").get("src")
                    except Exception:
                        tmp_img = ''
                    self.__texts.append(tmp_text)
                    self.__links.append(tmp_link)
                    self.__results.append({'title': tmp_text, 'media': tmp_media,'date': tmp_date,'datetime':define_date(tmp_date),'desc': tmp_desc, 'link': tmp_link,'img': tmp_img})
                
            except Exception as e_parser:
                print(e_parser)
                pass
            
            finally:
                self.response.close()

        def getpage(self, page=1):
            """Don't remove this, will affect old version user when upgrade"""
            self.get_page(page)  

        def utc_to_taipei(self, date: str):
            try:
                utc = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')
                from_zone = tz.gettz('UTC')
                to_zone = tz.gettz('Asia/Taipei')
                # Tell the datetime object that it's in UTC time zone since 
                # datetime objects are 'naive' by default
                utc = utc.replace(tzinfo=from_zone)
                taipei = utc.astimezone(to_zone)
                taipei_strtime = taipei.strftime("%Y-%m-%d %H:%M:%S")
                return taipei_strtime
            except Exception as e:
                return None

        def get_news(self, key="",deamplify=False):

            # key在這邊就是後面會傳入的一個keyword字串

            # keyword的空格改成'+'，這邊my_key應該是想把用'+'連接的key保留下來，多做了一次同樣步驟，但應該不影響。
            if key != '':
                key = "+".join(key.split(" "))
                my_key= "+".join(key.split(" "))

                # __encode預設是'utf-8'，如果encode不為""時，urllib.request.quote([編碼])可以把字串用[編碼]解析，再用url解析，以便後面放到self.url中做搜尋。
                if self.__encode != "":
                    key = urllib.request.quote(key.encode(self.__encode))
                self.url = 'https://news.google.com/search?q={}+when:{}&hl={}'.format(key,self.__period,self.__lang.lower())

            # __lang預設是'en'
            else:

                self.url = 'https://news.google.com/?hl={}'.format(self.__lang)

            __results = []
            for proxy in self.proxy_works:
                while True:
                    try:
                        self.proxy = proxy

                        # 搜尋self.url
                        self.req = urllib.request.Request(self.url, headers=self.headers)
                        # set proxy
                        proxy_support = urllib.request.ProxyHandler(
                            {'http':f'http://{auo_account}:{auo_password}@{self.proxy}', 
                            'https':f'https://{auo_account}:{auo_password}@{self.proxy}'})
                        opener = urllib.request.build_opener(proxy_support)
                        urllib.request.install_opener(opener)

                        self.response = urllib.request.urlopen(self.req)

                        self.page = self.response.read()
                        self.content = Soup(self.page, "html.parser")

                        # 把各文章的元素來迭帶，並從各文章中抓出需要的元素，加到self.__results
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
                                    # date,datetime_tmp = lexial_date_parser(date)
                                except:
                                    date = None
                                # datetime
                                try:
                                    datetime_chars=article.find('time').get('datetime')
                                    # zulutime
                                    # datetime_obj=article.find('time').get('datetime')
                                    #datetime_obj = parse(datetime_chars).replace(tzinfo=None)
                                except:
                                    datetime_chars=None 
                                    # datetime_obj=None
                                # link
                                if deamplify:
                                    try:
                                        link = 'news.google.com/' + article.find("h3").find("a").get("href")
                                    except Exception as deamp_e:
                                        print(deamp_e)
                                        link = article.find("article").get("jslog").split('2:')[1].split(';')[0]
                                else:
                                        link = 'news.google.com/' + article.find("h3").find("a").get("href")
                                self.__texts.append(title)
                                self.__links.append(link)
                                if link.startswith('https://www.youtube.com/watch?v='):
                                    desc = 'video'
                                # image
                                try:
                                    img = article.find("img").get("src")
                                except:
                                    img = None
                                # site
                                try:
                                    # site=article.select('a[class="wEwyrc AVN2gc WfKKme"]')[0].text #2023.6.7 網站更新class name
                                    site=article.select('a[class*="wEwyrc"]')[0].text
                                except:
                                    site=None
                                # collection
                                __results.append({'title':title,
                                                    'desc':desc,
                                                    'date':date,
                                                    'datetime':self.utc_to_taipei(datetime_chars),
                                                    # 'datetime':define_date(date),
                                                    'link':link,
                                                    'img':img,
                                                    'media':None,
                                                    'site':site})
                            except Exception as e_article:                   # 解析錯誤
                                print(e_article, f"{m_key}一篇文章解析錯誤")
                        
                        self.__results = __results
                        print(f"搜尋並解析成功: 關鍵字-{my_key}")
                        break                                                # articles迭帶完，停止 while
                    except urllib.error.URLError as e:  # 連線錯誤，articles尚未迭帶
                        print(f"搜尋時連線錯誤: 關鍵字-{my_key}(等待10秒後重試)")
                        print(e.reason)
                        sleep(10)
                        continue
                    except Exception as e_parser:       # 非連線錯誤
                        print(my_key, f"搜尋時發生非連線錯誤: 關鍵字-{my_key}")
                        print(e_parser)
                        pass
                    
                    finally:
                        self.response.close()                                # 關閉連線
                if __results != None or articles == None: break     # 停止proxy_works

        def total_count(self):
            return self.__totalcount

        def result(self,sort=False):
            """Don't remove this, will affect old version user when upgrade"""
            return self.results(sort)

        def results(self,sort=False):
            """Returns the __results.
            New feature: include datatime and sort the articles in decreasing order"""
            results=self.__results
            if sort:
                try:
                    results.sort(key = lambda x:x['datetime'],reverse=True)
                except Exception as e_sort:
                    print(e_sort)
                    results=self.__results
            return results

        def get_texts(self):
            """Returns only the __texts of the __results."""
            return self.__texts

        def gettext(self):
            """Don't remove this, will affect old version user when upgrade"""
            return self.get_texts()

        def get_links(self):
            """Returns only the __links of the __results."""
            return self.__links

        def clear(self):
            self.__texts = []
            self.__links = []
            self.__results = []
            self.__totalcount = 0


    # set parameters
    # 連到外網需要proxy才能連，而proxy中要打入nt帳密
    proxy_list = pd.read_excel("./crawler_config/proxy.xlsx", header=None)[0].tolist()
    nt_account = pd.read_excel("./crawler_config/account.xlsx")
    auo_account = nt_account["NTaccount"][0]
    auo_password = nt_account["password"][0]

    # 爬蟲會先用預設好的關鍵字列表去搜尋新聞，並在後續會依照關鍵字所屬的類別來分類新聞
    key_words_df = pd.read_excel("./crawler_config/keyword.xlsx")
    key_words = key_words_df.keyword

    all_results = dict()
    period = '1d'
    googlenews = GoogleNews(lang='zh-TW', period=period, account=auo_account, password=auo_password, proxy_list=proxy_list)
    proxy = googlenews.proxy

    # 所有Keyword放到google news去搜尋，並將搜到一天內的新聞的標題、描述、日期...等資訊抓出，放到all_result[keyword]中，值則是一個串列，裡面包含了許多字典(字典含有文章的標題、描述、日期)
    def Google_news_getter(keyword: str, all_results_dic: dict, googlenews_obj):
        googlenews_obj.get_news(keyword)
        result = googlenews_obj.results()
        all_results[keyword] = result

    tqdm.pandas(desc="用所有關鍵字去爬取一天內的Google News，並紀錄標題、描述、日期...等資訊")
    _ = key_words.progress_apply(lambda keyword:  Google_news_getter(keyword, all_results, googlenews))

    print('開始以關鍵字類別分類新聞')
    # 全部新聞
    df_list = list()
    # all_results是字典，鍵是關鍵字，所以迭帶時會迭帶關鍵字
    for key in all_results:
        # 創建跟這個關鍵字相關文章的df
        df = pd.DataFrame(all_results[key])
        # 指定"keyword"欄的值是關鍵字
        df["keyword"] = key
        # 指定"category"欄的值是keyword在keyword.xlsx中所對應的類別
        df["category"] = key_words_df.loc[key_words_df["keyword"]==key, "category"].iloc[0]
        # 把這個df放到df_list
        df_list.append(df)

    # # merge all DataFrame
    merged_df = pd.concat(df_list)
    now = datetime.datetime.now()
    start_day = now.strftime("%Y%m%d%H%M")
    previous_day = (now - datetime.timedelta(days=1)).strftime("%Y%m%d%H%M")


    # 把類別排序
    merged_df["sort"] = merged_df["category"]
    merged_df["sort"] = merged_df['sort'].apply(lambda x: {'產業':0, '法令':1, 'HR':2}[x] if x in {'產業', '法令', 'HR'} else 3)
    merged_df = merged_df.sort_values(["sort"])


    print('開始以條件過濾篩選新聞')
    # 只保留標題中含有關鍵字的文章
    print("\t過濾掉非關鍵字title")
    merged_df['contained'] = merged_df.apply(lambda x: any([keyword in x.title for keyword in list(key_words)]), axis=1)

    # 只保留標題是繁體中文，或標題含有'彭双浪'的文章
    print("\t過濾掉非繁體字title")
    merged_df['traditional_cn'] = merged_df.apply(lambda x: (hanzidentifier.is_traditional(x.title)) | ("彭双浪" in x.title)  , axis=1)
    merged_df = merged_df.loc[merged_df['contained']]
    merged_df = merged_df.loc[merged_df['traditional_cn']]

    # 只保留來源自指定的新聞出處的文章
    print("\t保留指定site來源")
    news_site = pd.read_excel("./crawler_config/NewsSite.xlsx")
    news_site["出處"] = news_site["出處"].str.replace(" ","").str.lower()
    include_site = news_site["出處"]

    # merged_df = merged_df.loc[~merged_df["site"].isin(drop_site)]
    merged_df["site"] = merged_df["site"].str.replace(" ","").str.lower()
    merged_df = merged_df.loc[merged_df["site"].isin(include_site)]

    # 只保留標題內不含dropword.xlsx指定文字的文章
    print("\t過濾含特定字眼的title")
    drop_title = list(pd.read_excel("./crawler_config/dropword.xlsx", header=None)[0])

    for i in drop_title:
        merged_df = merged_df[(merged_df["title"].str.contains(i)==False)]


    for i in drop_title:
        merged_df = merged_df[(merged_df["title"].str.contains(i)==False)]


    # # # drop_duplicate
    # 排除link欄位重複的資料，保留第一筆
    print("\t排除link(重複)的")
    merged_df = merged_df.drop_duplicates(subset='link', keep="first")
    # 排除title、datetime、site三個欄位為整體，都重複才排除
    merged_df = merged_df.drop_duplicates(subset=['title','datetime','site'], keep='first')

    print("過濾後共有:",merged_df.shape[0],'則新聞')

    # 縮短網址
    def url_shorten(auo_account, auo_password, proxy, url):
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"

        session = pyshorteners.Shortener(
            proxies={
                'http':f'http://{auo_account}:{auo_password}@{proxy}',
                'https':f'http://{auo_account}:{auo_password}@{proxy}',
            },
            timeout=10 # 設置超時時間為10秒
        )
        while True:
            patience = 3
            try:
                short_url = session.tinyurl.short(url)
                return short_url
            except requests.exceptions.Timeout:
                print("錯誤: 短網址請求超過10秒未回應")
                print("等待10秒後重試")
                sleep(10)
            except requests.exceptions.RequestException as e:
                print(f"發生連線錯誤: {e}")
                print("等待10秒後重試")
                sleep(10)
            except Exception:
                print("縮短一個連結時發生連線外錯誤，等待10秒後重試")
                sleep(10)
                patience += 1 
                if patience >= 3:
                    print("已重試三次，跳過此連結")
                    return None

    tqdm.pandas(desc="縮短所有Google News連結")
    merged_df['short_link'] = merged_df["link"].progress_apply(lambda link: url_shorten(auo_account, auo_password, proxy, link))


    selected_col = ['title','datetime','link', 'short_link','site','category']
    content_df = merged_df[selected_col]

    print("開始對標題分詞")
    # ckip分詞模型父資料夾
    ckip_path = ".\\ckiplab\\"

    # 建立分詞、詞性標記、命名實體物件
    ws_driver = CkipWordSegmenter(model_name=ckip_path + 'albert-base-chinese-ws') # 分詞
    pos_driver = CkipPosTagger(model_name=ckip_path + 'albert-base-chinese-pos')    # 詞性標記(POS)

    # 開始分詞、詞性標記，分詞結果放入ws，詞性標註結果放入pos
    ws = ws_driver(content_df.title)
    pos = pos_driver(ws)

    print("去除非名詞、非動詞")
    # 用來保留主要詞彙的函式
    def clean(sentence_ws, sentence_pos): 

        short_sentence = []
        stop_pos = set(['Nep', 'Nh']) # 這 2種詞性不保留(指代定詞、代名詞)

        for word_ws, word_pos in zip(sentence_ws, sentence_pos):

            # 部分的分詞內有包含空白字元，會導致停用詞不會被辨識到，所以在這邊用replace處理掉空白字元
            word_ws = word_ws.replace(' ','')

            # 只留名詞和動詞
            is_N_or_V = word_pos.startswith("V") or word_pos.startswith("N") or word_pos.startswith("FW")
            # 去掉名詞裡的某些詞性(指代定詞、代名詞、專有名詞)
            is_not_stop_pos = word_pos not in stop_pos
            # 只剩一個字的詞也不留
            is_not_one_charactor = not (len(word_ws) == 1)

            # 組成串列
            if is_N_or_V and is_not_stop_pos and is_not_one_charactor:
                short_sentence.append(f"{word_ws}")

        return " ".join(short_sentence)

    # 把前面的ws分詞、pos詞性標駐合併為同一個表格，並用apply套用上段函式
    df = pd.DataFrame({'ws':pd.Series(ws), 'pos':pd.Series(pos)})
    texts = df.apply(lambda row: clean(row.ws, row.pos),axis=1)
    texts.name = 'texts' # 先給texts一個欄位名稱

    print("讀取預測模型")
    # ### **讀取模型**

    # 模型路徑
    variable_names = ["click", "DailyNews"]
    model_path = "./models/"

    # 建立包含預測結果的result_df
    result_df = merged_df[selected_col]

    for name in variable_names:
        model_filename = f"model_{name}_best_f1.pkl"

        print(f"{name}模型預測")

        # 開啟模型
        with open( model_path + model_filename, "rb") as f:
            model = dill.load(f)


        # 提供預測結果及其機率值
        pred = model.predict(texts)
        prob = model.predict_proba(texts)[:,1]

        # 前面content_df的title在分詞前被處理過
        result_df[f'{name}_prob'] = prob
        result_df[f'{name}_prediction'] = pred

    print("預測表格彙整")

    # 更改欄位順序
    new_columns = [col for col in result_df.columns if "prediction" not in col] + [col for col in result_df.columns if "prediction" in col]
    result_df = result_df[new_columns]


    # 依照類別切出子表
    categories = result_df.category.unique().tolist()
    sub_result_df_list = []
    for category in categories:
        sub_df_mask = result_df.category == category
        sub_df = result_df[sub_df_mask]

        # 以機率值排序，讓機率高的優先排在前面
        sub_df["total_prob"] = sub_df.click_prob + sub_df.DailyNews_prob
        sub_df = sub_df.sort_values(by=["total_prob"], ascending=False)
        sub_df = sub_df.drop(columns=["total_prob"])

        # 彙整子表
        sub_result_df_list.append(sub_df)

    # 合併子表
    result_df = pd.concat(sub_result_df_list)

    # 寫出
    print("寫出預測表格")
    today = datetime.datetime.now().strftime("%Y%m%d")
    result_df.to_excel(f".\\history_predict\\News_{today}_predict.xlsx", encoding='utf-8-sig', index=False)

    print("爬蟲及預測完成")

    return result_df


if __name__ == '__main__':
    result_df = crawl_and_predict()