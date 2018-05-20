# Class SqlUpdater

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
            self.cursor = connection.cursor()
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
            # Write to file
            self.connection.rollback()

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

'''
class SqlUpdater:
    def __init__(self, _host=HOST, _port=PORT, _user=USER, _pwd=PWD):
        self._host = _host
        self._port = _port
        self._user = _user
        self._pwd = _pwd

    def connect(self, db=None):
        connection = None
        cursor = None

        if db is None:
            # this is database name
            db = DB
        try:
            # establish sql database connection
            connection = pymysql.connect(self._host, self._user, self._pwd, db)

            # create a cursor/handler
            cursor = connection.cursor()
            # query database version
            cursor.execute("SELECT VERSION()")
            # fetch one piece of data
            sql_version = cursor.fetchone()

            print("[INFO] MySQL Server Connected! Version: %s " % sql_version)
        except Exception as e:
            print(e)

        return connection, cursor

    def close(self, db_connection):
        db_connection.close()
        print("[INFO] MySQL Server Close Safely.")

    def insert(self, db_connection, db_cursor, dict):
        sql = "INSERT INTO TIMELOG(NAME, DATETIME, ACTION)\
               VALUES ('{}', '{}', '{}')".\
               format(dict['NAME'], dict['DATETIME'], dict['ACTION'])
        # print(db_connection.ping())

        try:
            # execute sql
            db_cursor.execute(sql)
            # commit info to database
            db_connection.commit()
        except:
            db_connection.rollback()

    def truncate(self, db_connection, db_cursor):
        # Delete all data in the table
        # TRUNCATE command will delete all data in the table very quickly
        sql = "TRUNCATE TABLE TIMELOG"
        try:
            db_cursor.execute(sql)
            db_connection.commit()
            print("[INFO] Delete all data in table.")
        except:
            db_connection.rollback()
'''
