from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.sql.expression import update, delete
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
import sqlalchemy
import sqlite3
#from sqlite3 import IntegrityError

import apocolib.sqlite_config as config
#from apocolib.apocolog4p import apoLogger as apolog

class SQLiteDbPool:
    def __init__(self, db_uri):
        self.db_url = config.SQLITE_DBFILE

        #apolog.info("db_url" + self.db_url)
        
        self.engine = create_engine(db_uri, echo=True)
        #apolog.info("create engine successfully.")
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        #self.Session.execute("PRAGMA read_uncommitted = 0;")
        #self.Session.execute("PRAGMA read_committed = 0;")
        #self.Session.execute("PRAGMA repeatable_read = 0;")
        #self.Session.execute("PRAGMA serializable = 1;")

        #apolog.info("create session successfully.")

    def execute(self, statement, params=None):
        with self.Session() as session:
            try:
                #session.execute("PRAGMA serializable = 1;")
                result = session.execute(text(statement), params)
                session.commit()
                #apolog.info("database return result -----:" + str(result))
                result = result.fetchall()
                return 0,result
#            except sqlite3.IntegrityError as e:
            except sqlalchemy.exc.IntegrityError as e:
                #apolog.error(str(e))
                return -1,'duplicate data'
            except Exception as e:
                #apolog.error(str(e))
                return -2,'database error'

    def select(self, table, columns="*", condition=None, params=None):
        query = f"SELECT {columns} FROM {table}"
        if condition:
            query += f" WHERE {condition}"
        return self.execute(query, params)

    def insert(self, table, data):
        keys = ", ".join(data.keys())
        values = ", ".join([f":{key}" for key in data])
        #query = f"INSERT INTO {table} ({keys}) VALUES ({values})"
        query = f"INSERT INTO {table} ({keys}) VALUES ({values})"

        return self.execute(query, data)

    def update(self, table, data, condition=None, params=None):
        values = ", ".join([f"{key} = :{key}" for key in data])
        query = f"UPDATE {table} SET {values}"
        if condition:
            query += f" WHERE {condition}"
        return self.execute(query, {**data, **params} if params else data)

    def delete(self, table, condition=None, params=None):
        query = f"DELETE FROM {table}"
        if condition:
            query += f" WHERE {condition}"
        return self.execute(query, params)
