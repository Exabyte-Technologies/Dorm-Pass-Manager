"""
Utility functions - miscellaneous helper functions
"""
from datetime import datetime, timedelta
from pathlib import Path


def listToJson(lst):
    """Convert a list of tuples to a JSON-like dictionary"""
    res_dict = {}
    for i in range(len(lst)):
        res_dict[str(lst[i][0])] = lst[i][1:]
    return res_dict


def currentDatetime(convertTimezone_func):
    """Get current datetime in the application timezone"""
    return convertTimezone_func(datetime.now())


def calculateElapsedSeconds(timestamp, currentDatetime_func):
    """Calculate elapsed seconds between timestamp and current time"""
    rawtime = currentDatetime_func() - timestamp
    return rawtime.days * 86400 + rawtime.seconds


def convertSecondsToTime(seconds):
    """Convert seconds to hours, minutes, seconds format"""
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return [hour, minutes, seconds]


def getScriptDir():
    """Get the directory of the current script"""
    return Path(__file__).resolve().parent.parent


def encrypt(data, fernet_func):
    """Encrypt data using Fernet"""
    encrypted_data = str(fernet_func.encrypt(data.encode()).decode('ascii'))
    return encrypted_data


def decrypt(encrypted_data, fernet_func):
    """Decrypt data using Fernet"""
    decrypted_data = fernet_func.decrypt(encrypted_data).decode()
    return decrypted_data
