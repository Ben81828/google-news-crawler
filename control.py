import os
import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
import sys
from time import sleep
import re

# 爬蟲和預測的程式
from News_Crawl_Predict import crawl_and_predict, LSI, LSA


#訓練程式
from retrain import retrain
# TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
# LSA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
# LSI
from gensim import corpora, models
from sklearn.model_selection import RandomizedSearchCV, train_test_split
# 寫出excel表格
from excel_result_table import result_table

# GUI介面程式
from UI import Ui_MainWindow
from PyQt5.QtCore import QThread, QCoreApplication, pyqtSignal
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QWidget, QHBoxLayout, QCheckBox, QLabel


# 日誌        
class Logger(object):
    def __init__(self,  output_box):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.output_box = output_box
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        sys.stdout = self.stdout

    def write(self, message):
        self.stdout.write(message)
        self.output_box.insertPlainText(message)
        self.output_box.moveCursor(QTextCursor.End)
        self.output_box.update()

    def flush(self):
        pass

# 核取方塊插件
class CheckBoxWidget(QWidget):
    def __init__(self):
        super().__init__(parent=None)

        self.checkbox = QCheckBox()

        layout = QHBoxLayout()
        layout.addWidget(self.checkbox)

        self.setLayout(layout)

    def isChecked(self):
        return self.checkbox.isChecked()

    def setChecked(self, checked):
        self.checkbox.setChecked(checked)

# 表格插件
class MyTable(QTableWidget):
    """用來將pd.DataFrame轉換成QTableWidget物件，並將含有關鍵字的欄位值轉乘checkbox。此物件可以用來渲染到scrollarea。這裡多寫的get_table_data也能將使用者互動後的表格做讀取"""
    def __init__(self, df, chbox_keyword=None, auto_checked=False):
        super().__init__()
        self.df = df
        self.initTable(chbox_keyword, auto_checked)

    def initTable(self, chbox_keyword, auto_checked):
        self.setRowCount(len(self.df))
        self.setColumnCount(len(self.df.columns))
        self.setHorizontalHeaderLabels(self.df.columns)

        for i in range(len(self.df)):
            for j in range(len(self.df.columns)):
                value = str(self.df.iloc[i, j])
                
                # 若chbox_keyword不為None，且欄位名稱包含有keyword，則將該次迭帶的值放入QCheckBox，再用setCellidget包。若無keyword則其值轉為文字放入
                if chbox_keyword and (chbox_keyword in self.df.columns[j]) :
                    item_widget = QCheckBox()
                    if auto_checked:
                        checked = True if eval(value) else False
                    else:
                        checked = False
                    item_widget.setChecked(checked)
                    self.setCellWidget(i, j, item_widget)    
                
                
                elif '.com' in value and "/" in value:
                    item_widget = QLabel()
                    item_widget.setText(f"<a href={value}>{value}</a>")
                    item_widget.setOpenExternalLinks(True)
                    self.setCellWidget(i, j, item_widget)
                else:
                    item = QTableWidgetItem(value)
                    self.setItem(i, j, item)
    
    def get_table_data(self):
        """ 從 MyTable物件，取得表格資料並轉回pd.DataFrame，並寫出到tag的路徑。
        Click和DailyNews可能因預測不準，而被使用者重新tag，所以此方法主要就是把這邊的Check Box的值更新並寫出"""
        # 取得表格資料
        table_data = []
        for row in range(self.rowCount()):
            row_data = []
            for column in range(self.columnCount()):
                widget = self.cellWidget(row, column)
                # 如果原本是Checkbox，用cellWidget後就不會是Nnone
                if widget is not None:
                    if isinstance(widget, QCheckBox):
                        value = "1" if widget.isChecked() else "0"
                    elif isinstance(widget, QLabel):
                        string = widget.text()
                        pattern = r"<a href=(\S*)>\1</a>"
                        result = re.search( pattern, string)
                        value = result.group(1)
                        
                else:
                    value = self.item(row, column).text()
                row_data.append(value)
            table_data.append(row_data)
            
        df = pd.DataFrame(table_data, columns=self.df.columns)
        
        return df

# 分線程執行程式
class WorkThread(QThread):
    """把要傳入的function傳入此物件，呼叫run方法讓該function在另一個線程執行(寫在QThread的run方法內的程式)"""
    end = pyqtSignal()
    
    def __init__(self, function, parent=None):
        super(WorkThread, self).__init__(parent)
        self.function = function
        
    def run(self):
        try:
            self.function()
            sleep(2) # 睡兩秒後，再傳結束的動作。
        except:
            self.end.emit()
        self.end.emit()
        
# GUI介面        
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 初始化各頁面所需變數
        self.terminal_init()
        self.crawl_init()
        self.keyword_init()
        self.account_init()
        self.train_init()
        
        # 爬蟲頁面相關UI
        self.ui.pushButton.clicked.connect(self.start_crawl_and_predict) # 點pushButton調用start_crawl_and_predict(執行剛剛建立的線程，開始爬蟲)
        self.ui.pushButton_2.clicked.connect(self.save_and_output)       # 點pushButton調用save_and_output(將爬蟲結束經使用者tag後的table，寫出到history_tag資料夾)
        # 關鍵字頁面相關UI
        self.ui.pushButton_6.clicked.connect(self.append_keyword)        # 調用append_keyword方法(在self.keyword_df最後新增空白列，再走一次顯示關鍵字表格的程序)
        self.ui.pushButton_3.clicked.connect(self.save_keyword)
        # NT帳號頁面相關UI
        self.ui.pushButton_4.clicked.connect(self.save_account)
        # 訓練頁面相關UI
        self.ui.pushButton_5.clicked.connect(self.start_train)
        
        
    ##  方法: 終端機模擬
    def terminal_init(self):
        """讓系統output渲染到textBrowser元素上"""
        self.terminal = self.ui.textBrowser
        self.logger = Logger(self.terminal)    
              
    ## 方法: 爬蟲預測頁面相關
    def crawl_init(self):
        self.table = None                                                # 爬蟲爬完並預測的table。型態是MyTable物件
        self.out_table = None                                            # 使用者互動後的table。型態會回傳為pd.DataFrame
        self.result_df = None                                            # 由out_table中擷取出來的部分資料，讓User可輸出excel樣式的新聞表格
            
    def start_crawl_and_predict(self):
        """開始初始化物件時，放到thread物件中的爬蟲方法，該方法會寫出一個excel的predict table"""
        self.crawl_init()
        self.ui.pushButton.setEnabled(False)
        self.ui.scrollArea.setWidget(self.table)
        self.crawl_thread = WorkThread(crawl_and_predict, parent=self)   # 建立分線程物件，把爬蟲程式放到另一個線程，以免主程式嚴重延遲
        self.crawl_thread.end.connect(self.show_table)                   # 爬蟲程式結束時，調用show_table方法(將爬蟲結果顯示到scrollArea)
        self.crawl_thread.start() # 用start()會呼叫thread物件內的run()方法
        
    def show_table(self):
        """把start_crawl_and_predict方法輸出的預測excel表格讀進來，放到Mytable物件，渲染到self.scrllArea"""
        self.ui.pushButton.setEnabled(True)
        today = datetime.now().strftime("%Y%m%d")
        result = pd.read_excel(f"./history_predict/News_{today}_predict.xlsx")
        self.table = MyTable(result, chbox_keyword="prediction", auto_checked=False)
        self.ui.scrollArea.setWidget(self.table)
        
        
    def save_and_output(self):
        """調用MyTable物件中的get_table_data方法，把使用者tag後的table寫出"""
        if type(self.table) != type(None):
            # self.table.get_table_data()會將與使用者互動後的table寫出，並回傳為pd.DataFrame
            self.out_table = self.table.get_table_data()

            # result_columns是最後要的表格欄位，這邊把預測機率prob欄位去掉
            result_columns = [col for col in self.out_table.columns if "prob" not in col]
            self.out_table = self.out_table[result_columns]

            # 依據歷史tag資料的命名方式，這邊將欄位click_prediction、DailyNews_prediction名稱，改回Click、 DailyNews
            correct_names = {'click_prediction': 'Click', 'DailyNews_prediction': 'DailyNews'}
            self.out_table.rename(columns=correct_names, inplace=True)

            # 檔案寫出路徑，如果該月份的路徑尚未存在，則先做出路徑
            now = datetime.now()
            current_month = now.strftime("%Y%m") 
            tag_path = f"./history_tag/{current_month}/"
            if not os.path.exists(tag_path): os.mkdir(tag_path)

            # 寫出self.out_table到路徑
            current_time = now.strftime("%Y%m%d")
            self.out_table.to_excel( tag_path + f'News_{current_time}_tag.xlsx', index=False, encoding='utf-8-sig')
            print("已將表格輸出至history_tag、history_result資料夾")
        else:
            print("爬蟲及預測尚未執行完畢")
            
        self.result_df = result_table(self.out_table)

            
    ## 方法: 關鍵字頁面相關
    def keyword_init(self):
        """讀取keyword.xlsx，並將該pd.DataFrame轉MyTable物件，放到scrollArea_2中做渲染"""
        self.keyword_excel_path = f".\\crawler_config\\keyword.xlsx"     
        self.keyword_df = pd.read_excel(self.keyword_excel_path)         # 讀取要用google搜尋的關鍵字 excel表
        self.keyword_table = MyTable(self.keyword_df)                    # 將關鍵字表格放入MyTable物件，以用來後續作渲染
        self.ui.scrollArea_2.setWidget(self.keyword_table)               # 初步渲染到scrollArea_2
        self.out_keyword_table = None                                    # 使用者互動後的table。使用self.save_keyword方法後，型態會回傳為pd.DataFrame
        
    def append_keyword(self):
        """在最後一欄新增空字串，以讓使用者輸入"""
        self.keyword_df.loc[len(self.keyword_df)] = ["",""]
        self.keyword_table = MyTable(self.keyword_df)
        self.ui.scrollArea_2.setWidget(self.keyword_table)
        
    def save_keyword(self):
        """儲存使用者新增、修正的關鍵字，並且要檢查有沒有空字串""" 
        out_keyword_table = self.keyword_table.get_table_data()
        
        ## out_keyword_table會有兩欄，分別是關鍵字、類別。
        ## 以下if-else邏輯: 當使用者在修改時，若表格沒空值，就直接儲存修改; 若有遺漏且該列內兩欄都為空值，則刪除該列並儲存; 若只有一欄為空值，則不儲存修改，並提醒使用者是哪欄有空。
        
        # 沒有空字串，直接儲存
        if (out_keyword_table=="").sum().sum()==0:
            self.out_keyword_table = out_keyword_table

        # 有空字串
        else:
            # 有空字串的row編號
            missing_rows = np.where(out_keyword_table=="")[0].tolist()
            # one_col_left_list 用來存下只有一個col有值的row index。若兩個col都消失的row，其row index在missing_rows中會重複出現兩次。
            one_col_left_check = lambda l: one_col_left_check(l[2:]) if (len(l)>=2) and (l[0]==l[1]) else [l[0]] + one_col_left_check(l[1:]) if l else l
            one_col_left_list = one_col_left_check(missing_rows)
            # all_clean為True，表示所有row的遺漏值都是遺漏兩個col。反之若為False，則有row只有一個col有值。
            all_clean = True if not one_col_left_list else False
            # all_clean若為True，則把全部的missing_rows從原本的df刪掉並儲存即可。反之若為False，則不可儲存，需向使用者回報哪一個row的哪一個col遺漏。
            if all_clean:
                self.out_keyword_table = out_keyword_table.loc[~out_keyword_table.index.isin(missing_rows)]
            else:
                out_keyword_table = out_keyword_table.loc[one_col_left_list]
                idxs = out_keyword_table.index.tolist()
                missing_cols = np.where(out_keyword_table=="")[1].tolist()
                for idx, col in zip(idxs, missing_cols):
                    print(f"第{idx}列的{out_keyword_table.columns[col]}尚有遺漏")
                          
        # 儲存self.out_keyword_table回一開始讀取的路徑
        if type(self.out_keyword_table) != type(None):
            self.out_keyword_table.to_excel(self.keyword_excel_path, index=False, encoding='utf-8-sig')
            sleep(1)
            self.keyword_init()
            print("關鍵字已儲存")
            
            
    ## 方法: NT帳號頁面相關
    def account_init(self):
        """讀取account.xlsx，將原本紀錄在裡面的NT帳密渲染在lineEdit元素"""
        self.account_excel_path = f".\\crawler_config\\account.xlsx"
        self.account_df = pd.read_excel(self.account_excel_path)
        self.ui.lineEdit.setText(self.account_df.NTaccount[0])
        self.ui.lineEdit_2.setText(self.account_df.password[0])   
    
    def save_account(self):
        """讓使用者在lineEdit元素上改帳密後，點按pushButton_4時，會連結到此方法，把修改存到原本NT帳密的excel"""
        account = self.ui.lineEdit.text()
        password = self.ui.lineEdit_2.text()
        self.account_df.NTaccount[0] = account
        self.account_df.password[0] = password
        self.account_df.to_excel(self.account_excel_path, index=False)
        sleep(1)
        self.account_init()
        print("帳號密碼已更新")
        
    ## 方法: 重新訓練相關
    def train_init(self):
        """讀取train_history.xlsx，並印出上次的訓練時間"""
        self.train_excel_path = f".\\crawler_config\\train_history.xlsx"
        self.history_df = pd.read_excel(self.train_excel_path)
        print(self.history_df.loc[len(self.history_df)-1]) # 印出歷史訓練時間
    
    def start_train(self):
        """把retrain.py的retrain方法，放到新線程，重新訓練模型"""
        self.train_thread = WorkThread(retrain, parent=self)
        self.train_thread.start()
        

# 主程式
if __name__ == '__main__':
    
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())