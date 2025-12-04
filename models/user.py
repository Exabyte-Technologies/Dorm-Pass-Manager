"""
User model - operations related to user records
"""
from .database import dbConnect, getRecord, getSingleRecord, executeQuery, getLastInsertId, dbfetchedConvertDate
from .validation import generateSHA256


def getUserNameFromOid(connection, oid):
    """Get user name by OID"""
    result = getRecord(
        connection,
        "users",
        "name",
        "oid = %s",
        (oid,)
    )
    
    return result[0][0] if result else None


def getUserIdFromOid(connection, oid):
    """Get user ID by OID"""
    result = getRecord(
        connection,
        "users",
        "userid",
        "oid = %s",
        (oid,)
    )
    
    return result[0][0] if result else None


def getUserNameFromId(connection, userid):
    """Get user name by user ID"""
    result = getRecord(
        connection,
        "users",
        "name",
        "userid = %s",
        (userid,)
    )
    
    return result[0][0] if result else None


def getUserEmailFromOid(connection, oid):
    """Get user email by OID"""
    result = getRecord(
        connection,
        "users",
        "email",
        "oid = %s",
        (oid,)
    )
    
    return result[0][0] if result else None


def getOidFromUserId(connection, userid):
    """Get OID from user ID"""
    result = getRecord(
        connection,
        "users",
        "oid",
        "userid = %s",
        (userid,)
    )
    
    return result[0][0] if result else None


def checkUserInformation(connection, usergetparam, oid):
    """Get specific user information by OID"""
    result = getRecord(
        connection,
        "users",
        usergetparam,
        "oid = %s",
        (oid,)
    )
    
    return result[0] if result else None


def verifyPassword(connection, userid, password):
    """Verify user password"""
    result = getRecord(
        connection,
        "users",
        "password",
        "userid = %s",
        (userid,)
    )
    
    if not result or len(result) < 1:
        return False
    
    return generateSHA256(password) == result[0][0]
