"""
Student model - operations related to student records
"""
from .database import dbConnect, getRecord, getSingleRecord, executeQuery, getLastInsertId, dbfetchedConvertDate
from datetime import datetime, timedelta


def getStudentInfoFromId(connection, studentid):
    """Get complete student information by student ID"""
    result = getRecord(
        connection,
        "students",
        "name, grade, floorid, disabledlocations, cardid, email",
        "studentid = %s",
        (studentid,)
    )
    
    if not result or len(result) < 1:
        return 'nostudent'
    
    try:
        return result[0]
    except IndexError:
        return 'nostudent'


def getStudentNameFromId(connection, studentid):
    """Get student name by student ID"""
    result = getRecord(
        connection,
        "students",
        "name",
        "studentid = %s",
        (studentid,)
    )
    
    return result[0][0] if result else None


def getStudentGradeFromId(connection, studentid):
    """Get student grade by student ID"""
    result = getRecord(
        connection,
        "students",
        "grade",
        "studentid = %s",
        (studentid,)
    )
    
    return result[0][0] if result else None


def getStudentFloorIdFromId(connection, studentid):
    """Get student floor ID by student ID"""
    result = getRecord(
        connection,
        "students",
        "floorid",
        "studentid = %s",
        (studentid,)
    )
    
    return result[0][0] if result else None


def getStudentCardidFromId(connection, studentid):
    """Get student card ID by student ID"""
    result = getRecord(
        connection,
        "students",
        "cardid",
        "studentid = %s",
        (studentid,)
    )
    
    return result[0][0] if result else None


def getStudentEmailFromId(connection, studentid):
    """Get student email by student ID"""
    result = getRecord(
        connection,
        "students",
        "email",
        "studentid = %s",
        (studentid,)
    )
    
    return result[0][0] if result else None


def getStudentImageFromId(connection, studentid):
    """Get student image by student ID"""
    result = getRecord(
        connection,
        "students",
        "image",
        "studentid = %s",
        (studentid,)
    )
    
    return result[0][0] if result else None


def isStudentSuspended(connection, studentid, currentDatetime_func):
    """Check if student is suspended"""
    result = getSingleRecord(
        connection,
        "students",
        "suspension, suspensionED",
        "studentid = %s",
        (studentid,)
    )
    
    if not result:
        return False
    
    suspension, suspensionED = result
    
    if not suspension:
        return False
    
    if suspensionED:
        if isinstance(suspensionED, str):
            try:
                suspensionED_dt = datetime.strptime(suspensionED, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                suspensionED_dt = datetime.fromisoformat(suspensionED)
        else:
            suspensionED_dt = suspensionED
        
        if suspensionED_dt < currentDatetime_func():
            # Suspension has expired, clear it
            executeQuery(
                connection,
                "UPDATE students SET suspension = NULL, suspensionED = NULL WHERE studentid = %s",
                (studentid,)
            )
            return False
    
    return suspension


def suspend_student(connection, studentid, message, suspension_end=None):
    """Suspend a student"""
    if not studentid or not message:
        return False

    try:
        executeQuery(
            connection,
            'UPDATE students SET suspension = %s, suspensionED = %s WHERE studentid = %s',
            (message, suspension_end, studentid)
        )
        return True
    except Exception as e:
        return False
