import os
import warnings
from copy import copy
import numpy as np
import pandas as pd
from datetime import datetime

# 進度條
from tqdm import tqdm

# 分詞
from ckip_transformers import __version__
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
from torch import cuda

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
from sklearn.metrics import precision_score, f1_score

# from sklearn 訓練用
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# 評估指標
from sklearn.metrics import precision_score, f1_score, confusion_matrix, classification_report


# 儲存模型用
import dill

def retrain():

    # ckip分詞模型父資料夾
    ckip_path = r"./ckiplab/"

    # 新聞標題及label的表格路徑
    folder_path = r"./history_tag"
    # 用串列抓出所有路徑內的資料夾

    folders = os.listdir(folder_path)

    df_list = []
    for folder in tqdm(folders, desc='讀取路徑內全部xslx檔案'):
        files = os.listdir(folder_path + "/" + folder)
        for file in files:
            tmp = pd.read_excel(folder_path + "/" + folder + "/" + file, dtype={'datetime': 'datetime64[ns]'})
            # tmp["Click"]=tmp["Click"].astype(str)
            # tmp["DailyNews"]=tmp["DailyNews"].astype(str)
            try:
                df_list.append(tmp[["title","datetime","short_url","site","Click", "DailyNews"]])
            except:
                print(file)
    content_df = pd.concat(df_list).dropna(subset=['title'])


    uni_dict = {"0":0,"1":1,"0.0":0,"1.0":1,"V":1,0:0,1:1}
    content_df.DailyNews = content_df.DailyNews.apply(lambda x : uni_dict[x] if x in uni_dict else np.nan)

    uni_dict = {"0":0,"1":1,"0.0":0,"1.0":1,"V":1,0:0,1:1}
    content_df.Click = content_df.Click.apply(lambda x : uni_dict[x] if x in uni_dict else np.nan)


    # ### 字元處理

    # In[4]:


    #暫時關閉future warning
    warnings.filterwarnings("ignore")

    # 去除非字元的字符
    content_df.title = content_df.title.str.replace(r'\W', r'')
    # 英文大寫
    content_df.title = content_df.title.str.upper()
    # 去除數字
    content_df.title = content_df.title.str.replace(r'\d', r'')

    #啟動future warning
    warnings.filterwarnings("default") 


    content_df = content_df.dropna()
    # ### CKIP Transformer分詞、詞性標註

    # In[5]:


    # 裝置
    device = 0 if cuda.is_available() else -1

    # 建立分詞、詞性標記、命名實體物件
    ws_driver = CkipWordSegmenter(model_name=ckip_path + 'albert-base-chinese-ws',device=device) # 分詞
    pos_driver = CkipPosTagger(model_name=ckip_path + 'albert-base-chinese-pos',device=device)    # 詞性標記(POS)


    # In[6]:


    ws = ws_driver(content_df.title)
    pos = pos_driver(ws)


    # In[7]:


    content_df = content_df.dropna(subset=['title'])


    # ### 移除部分詞性
    # 把CKIP中的部分詞性的詞、以及前面設定的停用詞移除

    # In[8]:


    # 用來保留主要詞彙的函式
    def clean(sentence_ws, sentence_pos):
        short_with_pos = []
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


    # # **數據預處理**

    # ### 讓後續模型在訓練時，用變數texts_df當作資料表

    # 把分詞和斷詞後的texts，和原本content_df的click、dailynews欄位合併

    # In[9]:


    texts.name = 'texts' # 先給texts一個欄位名稱
    texts_df = pd.concat([texts, content_df.reset_index(drop=True)[['Click', 'DailyNews']]], axis=1) #合併表格


    # In[10]:

    texts_df = texts_df.dropna()


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


    # In[13]:


    # 確認單次訓練的混淆矩陣和分類報告
    def predict_and_report(model, X_train, y_train, X_test, y_test, y_pred):

        precision, f1 = precision_score(y_test, y_pred), f1_score(y_test, y_pred)

        confmat = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i,
                        s=confmat[i, j],
                        va='center', ha='center')
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.show()
    #
        target_names = ['Not Click', 'Click']

        # 跑BalancedRandomForestClassifier會有很多future warning警告，這邊把它關掉
        warnings.filterwarnings("ignore")

        visualizer = ClassificationReport(model, classes=target_names, support=True)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)  
        visualizer.show()

        # 把future warning警告設定回來
        warnings.filterwarnings("default") 
        return precision, f1


    # # **打開模型**

    # In[14]:


    model_dir_path = "./models"
    Click_model_path = model_dir_path + "/model_click_best_f1.pkl"
    DailyNews_model_path = model_dir_path + "/model_DailyNews_best_f1.pkl"


    # In[15]:


    with open(Click_model_path, 'rb') as f:
        Click_model = dill.load(f)

    with open(DailyNews_model_path, 'rb') as f:
        DailyNews_model = dill.load(f)


    # # **切分Click訓練集測試集**

    # Click

    # In[16]:


    X_train, X_test, y_click_train, y_click_test = train_test_split(texts_df.texts, texts_df.Click, test_size=0.2, random_state=42)


    # # Click模型再訓練

    # In[17]:


    Click_model.fit(X_train, y_click_train)


    # In[18]:


    y_click_pred = Click_model.predict(X_test)


    # In[19]:


    click_precision = precision_score(y_click_test, y_click_pred)
    click_f1 = f1_score(y_click_test, y_click_pred)


    # In[20]:


    print("Click模型再訓練表現")
    print("precision_score: {:.2f}".format(click_precision))
    print("f1_score: {:.2f}".format(click_f1))


    # # 儲存Click模型

    # In[21]:


    # 儲存模型到指定路徑
    with open(Click_model_path, "wb") as f:
        dill.dump(Click_model, f)


    # # **切分DailyNews訓練集測試集**

    # DailyNews

    # In[22]:


    X_train, X_test, y_DailyNews_train, y_DailyNews_test = train_test_split(texts_df.texts, texts_df.DailyNews, test_size=0.2, random_state=42)


    # # DailyNews模型再訓練

    # In[23]:


    DailyNews_model.fit(X_train, y_DailyNews_train)


    # In[24]:


    y_DailyNews_pred = DailyNews_model.predict(X_test)


    # In[25]:


    DailyNews_precision = precision_score(y_DailyNews_test, y_DailyNews_pred)
    DailyNews_f1 = f1_score(y_DailyNews_test, y_DailyNews_pred)


    # In[26]:


    print("DailyNews模型再訓練表現")
    print("precision_score: {:.2f}".format(DailyNews_precision))
    print("f1_score: {:.2f}".format(DailyNews_f1))


    # # 儲存DailyNews模型

    # In[28]:


    # 儲存模型到指定路徑
    with open(DailyNews_model_path, "wb") as f:
        dill.dump(DailyNews_model, f)


    # # 備份模型

    # In[31]:


    now = datetime.now().strftime("%Y%m%d")


    # In[34]:


    model_dir_path = model_dir_path + "/" + now


    # In[35]:


    Click_model_path = model_dir_path + "/model_click_best_f1.pkl"
    DailyNews_model_path = model_dir_path + "/model_DailyNews_best_f1.pkl"


    # In[36]:


    if not os.path.exists(model_dir_path): os.mkdir(model_dir_path)

    with open(Click_model_path, "wb") as f:
        dill.dump(Click_model, f)

    with open(DailyNews_model_path, "wb") as f:
        dill.dump(DailyNews_model, f)

    

# In[ ]:


if __name__ == '__main__':
    retrain()