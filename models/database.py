"""
Database utilities and helper functions to reduce code duplication
"""
import mysql.connector
from datetime import datetime
import pytz


def dbConnect(dbhost, dbuser, dbpassword, dbdatabase):
    """Create and return a database connection"""
    return mysql.connector.connect(
        host=dbhost,
        user=dbuser,
        password=dbpassword,
        autocommit=True,
        database=dbdatabase,
        buffered=True
    )


def convertTimezone(prevTime):
    """Convert datetime to Asia/Shanghai timezone"""
    conversionTimezone = pytz.timezone('Asia/Shanghai')
    return conversionTimezone.localize(prevTime)


def dbfetchedConvertDate(dbcursorFetched):
    """Convert datetime objects in database results and apply timezone conversion"""
    returnResult = []
    for entry in dbcursorFetched:
        returnResult.append([])
        try:
            for element in entry:
                if isinstance(element, datetime):
                    returnResult[-1].append(convertTimezone(element))
                else:
                    returnResult[-1].append(element)
        except:
            pass
    return returnResult


def dbfetchedOneConvertDate(dbcursorFetched):
    """Convert datetime objects in a single database result row"""
    returnResult = []
    for element in dbcursorFetched:
        if isinstance(element, datetime):
            returnResult.append(convertTimezone(element))
        else:
            returnResult.append(element)
    return returnResult


def getRecord(connection, table, columns, where_clause, params):
    """Generic function to fetch records from database"""
    with connection.cursor() as dbcursor:
        query = f"SELECT {columns} FROM {table} WHERE {where_clause}"
        dbcursor.execute(query, params)
        result = dbfetchedConvertDate(dbcursor.fetchall())
    
    if len(result) < 1:
        return None
    
    return result


def getSingleRecord(connection, table, columns, where_clause, params):
    """Generic function to fetch a single record from database"""
    with connection.cursor() as dbcursor:
        query = f"SELECT {columns} FROM {table} WHERE {where_clause}"
        dbcursor.execute(query, params)
        result = dbfetchedOneConvertDate(dbcursor.fetchone())
    
    if not result:
        return None
    
    return result


def recordExists(connection, table, where_clause, params):
    """Check if a record exists in the database"""
    with connection.cursor() as dbcursor:
        query = f"SELECT 1 FROM {table} WHERE {where_clause}"
        dbcursor.execute(query, params)
        result = dbcursor.fetchone()
    
    return result is not None


def executeQuery(connection, query, params):
    """Execute a query (INSERT, UPDATE, DELETE) and return result"""
    with connection.cursor() as dbcursor:
        dbcursor.execute(query, params)
        return dbcursor


def getLastInsertId(connection):
    """Get the last inserted row ID"""
    with connection.cursor() as dbcursor:
        dbcursor.execute('SELECT LAST_INSERT_ID()')
        result = dbfetchedConvertDate(dbcursor.fetchall())
    
    return result[0][0] if result else None
