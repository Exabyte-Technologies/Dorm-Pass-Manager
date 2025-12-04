"""
Settings model - operations related to application settings
"""
from .database import dbConnect, getRecord, getSingleRecord, executeQuery, getLastInsertId, dbfetchedConvertDate


def getSettingsValue(connection, settingName):
    """Get a settings value by name"""
    result = getRecord(
        connection,
        "settings",
        "value",
        "name = %s",
        (settingName,)
    )
    
    return result[0][0] if result else None


def setSettingsValue(connection, settingName, settingValue):
    """Update a settings value"""
    executeQuery(
        connection,
        "UPDATE settings SET value = %s WHERE name = %s",
        (settingValue, settingName)
    )
