"""
Session model - operations related to session management
"""
from .database import dbConnect, getRecord, getSingleRecord, executeQuery, getLastInsertId, dbfetchedConvertDate
from datetime import timedelta
import string
import random


class sessionStorage:
    """Class to manage user sessions"""
    
    @staticmethod
    def create(connection, oid, keepstatedays, getSettingsValue_func, currentDatetime_func, isstudent=False):
        """Create a new session"""
        passkeylength = int(getSettingsValue_func(connection, 'passkeyLength'))
        passkey = ''.join(random.choices(string.ascii_uppercase + string.digits, k=passkeylength))
        expdate = currentDatetime_func() + timedelta(days=keepstatedays)
        
        executeQuery(
            connection,
            'INSERT INTO sessions (oid, passkey, expdate, active, isstudent) VALUES (%s, %s, %s, %s, %s)',
            (oid, passkey, expdate, True, isstudent)
        )
        
        sessionid = getLastInsertId(connection)
        
        return [str(sessionid), passkey]
    
    @staticmethod
    def verify(connection, sessionid, passkey, currentDatetime_func):
        """Verify a session"""
        result = getRecord(
            connection,
            "sessions",
            "oid, expdate, isstudent",
            "sessionid = %s AND passkey = %s AND active = true",
            (sessionid, passkey)
        )
        
        if not result or len(result) < 1:
            return None

        oid = result[0][0]
        expdate = result[0][1]
        isstudent = result[0][2]
        
        if expdate < currentDatetime_func():
            executeQuery(
                connection,
                'UPDATE sessions SET active = false WHERE sessionid = %s AND passkey = %s',
                (sessionid, passkey)
            )
            return None
        
        userrole = None
        
        if not isstudent:
            result = getRecord(
                connection,
                "users",
                "role",
                "oid = %s",
                (oid,)
            )
            userrole = result[0][0] if result else None

        return [oid, userrole, isstudent]
    
    @staticmethod
    def deactivate(connection, sessionid, passkey):
        """Deactivate a session"""
        executeQuery(
            connection,
            'UPDATE sessions SET active = FALSE WHERE sessionid = %s AND passkey = %s',
            (sessionid, passkey)
        )
        return True
