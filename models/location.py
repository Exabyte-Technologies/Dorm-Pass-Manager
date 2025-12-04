"""
Location model - operations related to location records
"""
from .database import dbConnect, getRecord, getSingleRecord, executeQuery, getLastInsertId, dbfetchedConvertDate


def getLocationNameFromId(connection, locationid):
    """Get location name by location ID"""
    result = getRecord(
        connection,
        "locations",
        "name",
        "locationid = %s",
        (locationid,)
    )
    
    return result[0][0] if result else None


def getLocationIdFromName(connection, location_name):
    """Get location ID by location name"""
    result = getRecord(
        connection,
        "locations",
        "locationid",
        "name = %s",
        (location_name,)
    )
    
    return result[0][0] if result else None


def getLocationType(connection, locationid):
    """Get location type by location ID"""
    result = getRecord(
        connection,
        "locations",
        "type",
        "locationid = %s",
        (locationid,)
    )
    
    return result[0][0] if result else None


def getLocationsInformation(connection, location_type, locationid=None):
    """Get location information by type, optionally filtered by location ID"""
    if locationid == None:
        result = getRecord(
            connection,
            "locations",
            "locationid, name",
            "type = %s",
            (location_type,)
        )
    else:
        result = getRecord(
            connection,
            "locations",
            "locationid, name",
            "locationid = %s AND type = %s",
            (locationid, location_type)
        )
    
    return result if result else None


def joinLocations(locationList):
    """Join location names with commas"""
    joinedString = ""
    for i in range(len(locationList)):
        joinedString += str(locationList[i][1])
        joinedString += ','
    joinedString = joinedString[:-1]
    return joinedString
