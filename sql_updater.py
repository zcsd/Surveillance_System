# Class SqlUpdater, it's a threaded class.

'''
Update information(NAME, DATETIME, ACTION) to office MySQL server.
'''
from threading import Thread
import pymysql

HOST = "172.19.80.25"
PORT = "3306"  # default port is 3306
USER = "zclin"
PWD = "1024"
DB = "office-iot"
TB = 'TIMELOG'

class SqlUpdater(Thread):
    def __init__(self, q):
        Thread.__init__(self)
        self.info_queue = q
        self.connection = None
        self.cursor = None
        self.running = False

    def connect(self):
        try:
            # establish sql database connection
            self.connection = pymysql.connect(HOST, USER, PWD, DB)

            # create a cursor/handler
            self.cursor = self.connection.cursor()
            # query database version
            self.cursor.execute("SELECT VERSION()")
            # fetch one piece of data
            sql_version = self.cursor.fetchone()

            print("[INFO] MySQL Server Connected! Version: %s " % sql_version)
        except Exception as e:
            print(e)

    def close(self):
        if self.running:
            self.connection.close()
            print("[INFO] MySQL Server Close Safely.")
        self.running = False
        print("[INFO] SQL Thread Exit.")

    def truncate(self):
        # Delete all data in the table
        # TRUNCATE command will delete all data in the table very quickly
        sql = "TRUNCATE TABLE TIMELOG"
        try:
            self.cursor.execute(sql)
            self.connection.commit()
            print("[INFO] Delete all data in table.")
        except:
            self.connection.rollback()

    def insert(self, dict):
        sql = "INSERT INTO TIMELOG(NAME, DATETIME, ACTION)\
               VALUES ('{}', '{}', '{}')".\
               format(dict['NAME'], dict['DATETIME'], dict['ACTION'])
        # print(self.connection.ping())

        try:
            # execute sql
            self.cursor.execute(sql)
            # commit info to database
            self.connection.commit()
        except:
            # Backup to local file if insert failed
            seq = str(dict['NAME']) + "  " + str(dict['DATETIME']) + "  " + str(dict['ACTION']) + "\n"
            with open('timelog/backup.txt','a') as f:
                f.writelines(seq)
            f.close()
            #self.connection.rollback()

    def run(self):
        print("[INFO] SQL Thread Created.")
        self.connect()
        if self.connection == None:
            print("[INFO] Failed to Connect SQL. ")
            self.running = False
        else:
            self.running = True
            # self.truncate()  # Delete all data in database table
        while self.running:
            #print(self.info_queue.qsize())
            if self.info_queue.qsize() == 0:
                continue

            new_info_dict = self.info_queue.get(True, 3)

            if self.running:
                self.insert(new_info_dict)
