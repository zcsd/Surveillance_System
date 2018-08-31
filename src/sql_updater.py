# Class SqlUpdater, update info to SQL server

'''
Update information(NAME, DATETIME, ACTION) to office MySQL server.
'''
from threading import Thread
from queue import Queue
import pymysql

HOST = "172.19.80.25"
PORT = "3306"  # default port is 3306
USER = "zclin"
PWD = "1024"
DB = "office-iot"
TB = 'TIMELOG'


# Write timelog information to text file if sql connection fail
def backup_to_timelog(q):
    seq_list = []
    # put all information in queue to a list
    for i in range(q.qsize()):
        dict = q.get()
        seq = str(dict['NAME']) + "  " + str(dict['DATETIME']) + \
            "  " + str(dict['ACTION']) + "\n"
        seq_list.append(seq)
    # write list information to txt file
    with open('timelog/backup.txt', 'a') as f:
        f.writelines(seq_list)

    f.close()
    print("[INFO] Wrote to backup timelog.")


class SqlUpdater:
    def __init__(self):
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

            if self.connection == None:
                print("[INFO] Failed to Connect SQL. ")
                self.running = False
            else:
                self.running = True
                print("[INFO] MySQL Server Connected! Version: %s " %
                      sql_version)
        except Exception as e:
            print("[INFO] Failed to Connect SQL. ")
            print(e)

    def close(self):
        if self.running:
            self.connection.close()
            print("[INFO] MySQL Server Close Safely.")
        self.running = False

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
        sql = "INSERT INTO TIMELOG(NAME, TIMESTAMP, VIDEO_PATH)\
               VALUES ('{}', '{}', '{}')".\
            format(dict['NAME'], dict['TIMESTAMP'], dict['VIDEO_PATH'])
        # print(self.connection.ping())

        try:
            # execute sql
            self.cursor.execute(sql)
            # commit info to database
            self.connection.commit()
        except:
            # Backup to local file if insert failed
            seq = str(dict['NAME']) + "  " + str(dict['TIMESTAMP']
                                                 ) + "  " + str(dict['VIDEO_PATH']) + "\n"
            with open('timelog/backup.txt', 'a') as f:
                f.writelines(seq)
            f.close()
            # self.connection.rollback()
