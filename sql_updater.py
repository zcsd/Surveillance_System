# Class SqlUpdater

'''
Update information(NAME, DATETIME, ACTION) to office MySQL server. 
'''

import pymysql
from flask import g

HOST = "172.19.80.25"
PORT = "3306"  # default port is 3306
USER = "zclin"
PWD = "1024"
DB = "office-iot"
TB = 'TIMELOG'

class SqlUpdater:
    def __init__(self, _host=HOST, _port=PORT, _user=USER, _pwd=PWD):
        self._host = _host
        self._port = _port
        self._user = _user
        self._pwd = _pwd

    def connect(self, db=None):
        _host = self._host
        _user = self._user
        _pwd = self._pwd

        connection = None
        cursor = None

        if db is None:
            # this is database name
            db = DB
        try:
            # establish sql database connection
            connection = pymysql.connect(_host, _user, _pwd, db)

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
            db_connection.rollback