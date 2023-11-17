import urllib.request
from bs4 import BeautifulSoup as Soup
import pandas as pd

import langchain
langchain.verbose = False
langchain.verbose
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate
import os
from tqdm import tqdm
from time import sleep

def summary_result_table(title_list, link_list):
    

    def get_to_link(link):
        """連到傳入的link，回傳soup解析的html"""
        # set parameters
        # 連到外網需要proxy才能連，而proxy中要打入nt帳密
        proxy_list = pd.read_excel("./crawler_config/proxy.xlsx", header=None)[0].tolist()
        nt_account = pd.read_excel("./crawler_config/account.xlsx")
        auo_account = nt_account["NTaccount"][0]
        auo_password = nt_account["password"][0]

        headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:64.0) Gecko/20100101 Firefox/64.0'}
        retries = 3
        proxy_works = []
        
        print(f"嘗試連到連結: {link}")

        for proxy in proxy_list:
            print('proxy= ', proxy, end='')
            req = urllib.request.Request(link, headers=headers)
            # set proxy
            proxy_support = urllib.request.ProxyHandler(
                    {
                        'http':f'http://{auo_account}:{auo_password}@{proxy}', 
                        'https':f'https://{auo_account}:{auo_password}@{proxy}',
                    })
            opener = urllib.request.build_opener(proxy_support)
            urllib.request.install_opener(opener)


            work=False
            for attempt in range(retries):
                response = None

                try:
                    response = urllib.request.urlopen(req, timeout=10)
                    page = response.read()
                    # html_content = Soup(page, "html.parser")
                    html_content = Soup(page, "lxml")

                    # 成功連線
                    print("ok")
                    proxy_works.append(proxy)
                    work=True

                except Exception as e:
                    print(f'連線失敗: (Exception: {e}\n重新連線，3秒後重試')
                    sleep(3)

                finally:
                    if response:
                        response.close()
                    if work:
                        return html_content
                    
        class no_response:
            def __init__(self):
                self.text = "網站連線無回應"
                print(self.text)
        
        response = no_response()
        return response 

    def html_text_map_reduce( news_title, html_text, split_char_num = 2000):
        """將新聞標題傳入、新聞網頁的html.text傳入，用split_char_num個字元分割html.text，以map_reduce架構跟chatgpt摘要，最後回傳摘要"""

        # 先將html_text內的文字，以split_char_num個token為單位做分割
        split_text_func = lambda s: [s[:split_char_num]] + split_text_func(s[split_char_num:]) if len(s) >=split_char_num else [s]
        html_text_split = split_text_func(html_text)

        # 將分割後的字串轉為Document物件，並集合成doc串列
        docs = [Document(page_content=t) for t in html_text_split]

        # prompt template
        template = \
        """下文中可能包含與標題:『{title}』相關的新聞內文:

        {text}

        請用繁體中文撰寫與標題有關的簡潔摘要:"""

        prompt_template = PromptTemplate(template=template, input_variables=["title", "text"])

        # llm
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", verbose=True)

        # chain
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt_template, combine_prompt=prompt_template)

        # run
        result = chain.run(input_documents=docs, title=news_title, return_only_outputs=True)

        return result

    # 使用api.openai.com為安全的憑證
    os.environ['REQUESTS_CA_BUNDLE'] = "\\\\homehc\\BenBLLee\\openaiCert\\TMGCert.crt"
    # 設定proxy
    os.environ['http_proxy'] = "http://benbllee:Aa0937454850*@auhqproxy.cdn.corpnet.auo.com:8080"
    os.environ['https_proxy'] = "https://benbllee:Aa0937454850*@auhqproxy.cdn.corpnet.auo.com:8080"
    # 設定api key及
    os.environ['OPENAI_API_KEY'] = "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"


    # 將連結內的文字彙整到html_text_list
    html_text_list = []

    for link in link_list:
        middle_html = get_to_link(link)

        if middle_html:

            real_link = [a for a in middle_html.find_all('a') if "http" in a.text][0].text

            final_html_content = get_to_link(real_link)

            html_text_list.append(final_html_content.text)

    # 將html_text內的文字，用chatgpt做摘要後，彙整到result_list
    result_list = []

    with get_openai_callback() as cb:    # get_openai_callback用來紀錄並計算在cb內的token及cost

        for new_title, html_text in tqdm(list(zip(title_list, html_text_list)), desc='已摘要文章'): 

            html_text = html_text.replace(" ","")

            result = html_text_map_reduce( new_title, html_text)

            result_list.append(result)


        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
    
    summary_df = pd.DataFrame(columns=["新聞摘要"])
    for title, summary in list(zip(title_list,result_list)):
        summary_df.loc[len(summary_df)] = [title]
        summary_df.loc[len(summary_df)] = [summary]
    
    return summary_df