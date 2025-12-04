"""
Models package - contains database models and utilities
"""

from .database import (
    dbConnect,
    convertTimezone,
    dbfetchedConvertDate,
    dbfetchedOneConvertDate,
    getRecord,
    getSingleRecord,
    recordExists,
    executeQuery,
    getLastInsertId
)

from .validation import (
    generateSHA256,
    checkNameLength,
    checkGrade,
    checkCardidLength,
    checkEmailLength,
    checkPassword,
    validate_json_payload
)

from .student import (
    getStudentInfoFromId,
    getStudentNameFromId,
    getStudentGradeFromId,
    getStudentFloorIdFromId,
    getStudentCardidFromId,
    getStudentEmailFromId,
    getStudentImageFromId,
    isStudentSuspended,
    suspend_student
)

from .user import (
    getUserNameFromOid,
    getUserIdFromOid,
    getUserNameFromId,
    getUserEmailFromOid,
    getOidFromUserId,
    checkUserInformation,
    verifyPassword
)

from .location import (
    getLocationNameFromId,
    getLocationIdFromName,
    getLocationType,
    getLocationsInformation,
    joinLocations
)

from .settings import (
    getSettingsValue,
    setSettingsValue
)

from .session import sessionStorage

from .utils import (
    listToJson,
    currentDatetime,
    calculateElapsedSeconds,
    convertSecondsToTime,
    getScriptDir,
    encrypt,
    decrypt
)

__all__ = [
    'dbConnect',
    'convertTimezone',
    'dbfetchedConvertDate',
    'dbfetchedOneConvertDate',
    'getRecord',
    'getSingleRecord',
    'recordExists',
    'executeQuery',
    'getLastInsertId',
    'generateSHA256',
    'checkNameLength',
    'checkGrade',
    'checkCardidLength',
    'checkEmailLength',
    'checkPassword',
    'validate_json_payload',
    'getStudentInfoFromId',
    'getStudentNameFromId',
    'getStudentGradeFromId',
    'getStudentFloorIdFromId',
    'getStudentCardidFromId',
    'getStudentEmailFromId',
    'getStudentImageFromId',
    'isStudentSuspended',
    'suspend_student',
    'getUserNameFromOid',
    'getUserIdFromOid',
    'getUserNameFromId',
    'getUserEmailFromOid',
    'getOidFromUserId',
    'checkUserInformation',
    'verifyPassword',
    'getLocationNameFromId',
    'getLocationIdFromName',
    'getLocationType',
    'getLocationsInformation',
    'joinLocations',
    'getSettingsValue',
    'setSettingsValue',
    'sessionStorage',
    'listToJson',
    'currentDatetime',
    'calculateElapsedSeconds',
    'convertSecondsToTime',
    'getScriptDir',
    'encrypt',
    'decrypt'
]
