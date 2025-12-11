from flask import (
    Flask, render_template, request, session, redirect, 
    send_file, jsonify, make_response
)
from flask_session_captcha import FlaskSessionCaptcha
from cryptography.fernet import Fernet
from msal import ConfidentialClientApplication
from mysql.connector import Error as MySQLError
import mysql.connector
import os
import re
from datetime import datetime, timedelta, timezone
import time
import string
import random
import hashlib
import pytz
import json
import smtplib
from email.mime.text import MIMEText
from functools import wraps
from typing import Dict, Any, Union
import logging
import secrets
from flask_socketio import SocketIO, join_room, leave_room
import base64
import io
import numpy as np
from PIL import Image, ImageFile
import face_recognition
from io import BytesIO
from pathlib import Path
import cv2
from deepface import DeepFace
import tempfile

# Import from models package
from models import (
    dbConnect as db_connect,
    convertTimezone,
    dbfetchedConvertDate,
    dbfetchedOneConvertDate,
    getRecord,
    getSingleRecord,
    generateSHA256,
    checkNameLength as validate_name_length,
    checkGrade as validate_grade,
    checkCardidLength as validate_cardid_length,
    checkEmailLength as validate_email_length,
    checkPassword,
    validate_json_payload,
    getStudentInfoFromId as getStudentInfoFromId_model,
    getStudentNameFromId as getStudentNameFromId_model,
    getStudentGradeFromId as getStudentGradeFromId_model,
    getStudentFloorIdFromId as getStudentFloorIdFromId_model,
    getStudentCardidFromId as getStudentCardidFromId_model,
    getStudentEmailFromId as getStudentEmailFromId_model,
    getStudentImageFromId as getStudentImageFromId_model,
    isStudentSuspended as check_student_suspended,
    suspend_student,
    getUserNameFromOid as getUserNameFromOid_model,
    getUserIdFromOid as getUserIdFromOid_model,
    getUserNameFromId as getUserNameFromId_model,
    getUserEmailFromOid as getUserEmailFromOid_model,
    getOidFromUserId as getOidFromUserId_model,
    checkUserInformation as checkUserInformation_model,
    verifyPassword as verifyPassword_model,
    getLocationNameFromId as getLocationNameFromId_model,
    getLocationIdFromName as getLocationIdFromName_model,
    getLocationType as getLocationType_model,
    getLocationsInformation as getLocationsInformation_model,
    joinLocations,
    getSettingsValue as get_setting,
    setSettingsValue,
    sessionStorage,
    listToJson,
    currentDatetime as get_current_datetime,
    calculateElapsedSeconds as calculateElapsedSeconds_model,
    convertSecondsToTime,
    getScriptDir,
    encrypt as encrypt_data,
    decrypt as decrypt_data
)

app = Flask(__name__)

with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    serverConfig = json.load(f)

app.config['SECRET_KEY'] = os.urandom(serverConfig['session']['keyLength'])
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=serverConfig['session']['lifetime'])

app.config['CAPTCHA_ENABLE'] = serverConfig['captcha']['enable']
app.config['CAPTCHA_LENGTH'] = serverConfig['captcha']['length']
app.config['CAPTCHA_WIDTH'] = serverConfig['captcha']['width']
app.config['CAPTCHA_HEIGHT'] = serverConfig['captcha']['height']
app.config['CAPTCHA_INCLUDE_ALPHABET'] = serverConfig['captcha']['alphabet']
app.config['CAPTCHA_INCLUDE_NUMERIC'] = serverConfig['captcha']['numeric']

captcha = FlaskSessionCaptcha(app)

encryption_key = Fernet.generate_key()
fernet = Fernet(encryption_key)
debug = serverConfig['debug']['enable']

dbhost = serverConfig['database']['dbhost']
dbuser = serverConfig['database']['dbuser']
dbpassword = serverConfig['database']['dbpassword']
dbdatabase = serverConfig['database']['dbdatabase']

socketio = SocketIO(app)

ImageFile.LOAD_TRUNCATED_IMAGES = True

def dprint(text):
    if debug:
        print(text)


# Create database connection wrapper with config parameters
def dbConnect():
    global dbhost, dbuser, dbpassword, dbdatabase
    return db_connect(dbhost, dbuser, dbpassword, dbdatabase)


# Create encryption/decryption wrappers
def encrypt(data):
    return encrypt_data(data, fernet)


def decrypt(encrypted_data):
    return decrypt_data(encrypted_data, fernet)

def get_msal_app():
    return ConfidentialClientApplication(
        getSettingsValue('msauthClientId'),
        authority=getSettingsValue('msauthauthority'),
        client_credential=getSettingsValue('msauthclientsecret')
    )

def generate_csrf_token():
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_urlsafe(32)
    return session['_csrf_token']

app.jinja_env.globals['csrf_token'] = generate_csrf_token
    

def checkUserInformation_wrapper(usergetparam, oid):
    """Wrapper to match original API"""
    with dbConnect() as connection:
        return checkUserInformation_model(connection, usergetparam, oid)


# Alias for backward compatibility
checkUserInformation = checkUserInformation_wrapper


def checkStudentInformation(studentgetparam, oid):
    """Get student information by OID"""
    with dbConnect() as connection:
        result = getRecord(
            connection,
            "students",
            studentgetparam,
            "oid = %s",
            (oid,)
        )
    
    return result[0] if result else None


def getLocationsInformation_wrapper(location_type, locationid=None):
    """Wrapper for getLocationsInformation from models"""
    with dbConnect() as connection:
        return getLocationsInformation_model(connection, location_type, locationid)


def getLocationType_wrapper(locationid):
    """Wrapper for getLocationType from models"""
    with dbConnect() as connection:
        return getLocationType_model(connection, locationid)


# Aliases for backward compatibility  
getLocationsInformation = getLocationsInformation_wrapper
getLocationType = getLocationType_wrapper

def ensureLoggedIn(session, allowedroles = 3, studentPortal = False, kioskAllowed = False):
    try:
        with dbConnect() as connection:
            sessionid = decrypt(str(session.get('sessionid')))
            passkey = decrypt(str(session.get('passkey')))
            isKiosk = session.get('kiosk')

            if not kioskAllowed and isKiosk:
                return False
            
            userinfo = sessionStorage.verify(sessionid, passkey)
            if studentPortal:
                if userinfo[2] == False:
                    return False
                return True
            if userinfo != None and userinfo[1] <= allowedroles and userinfo[2] == False:
                if isKiosk:
                    return 'kiosk'
                return True
            else:
                return False
    except:
        return False
    
def currentDatetime():
    return get_current_datetime(convertTimezone)

def getPassStatus(passid):
    with dbConnect() as connection:
        with connection.cursor() as dbcursor:
            dbcursor.execute('SELECT fleavetime, darrivetime, dleavetime, farrivetime FROM passes WHERE passid = %s', (passid,))
            result = dbfetchedConvertDate(dbcursor.fetchall())
            
    if result[0][0] == None:
        return 0
    elif result[0][1] == None:
        return 1
    elif result[0][2] == None:
        return 2
    elif result[0][3] == None:
        return 3
    else:
        return None
    
def currentDatetime():
    return get_current_datetime(convertTimezone)


def calculateElapsedSeconds_func(timestamp):
    return calculateElapsedSeconds_model(timestamp, currentDatetime)


# Alias for backward compatibility
calculateElapsedSeconds = calculateElapsedSeconds_func

def validate_name_length_wrapper(name):
    """Wrapper to get setting values"""
    with dbConnect() as connection:
        min_length = int(get_setting(connection, 'minNameLength'))
        max_length = int(get_setting(connection, 'maxNameLength'))
    return validate_name_length(name, min_length, max_length)


def validate_grade_wrapper(grade):
    """Wrapper to get setting values"""
    with dbConnect() as connection:
        min_grade = int(get_setting(connection, 'minGrade'))
        max_grade = int(get_setting(connection, 'maxGrade'))
    return validate_grade(grade, min_grade, max_grade)


def validate_cardid_length_wrapper(cardid):
    """Wrapper to get setting values"""
    with dbConnect() as connection:
        min_length = int(get_setting(connection, 'minCardidLength'))
        max_length = int(get_setting(connection, 'maxCardidLength'))
    return validate_cardid_length(cardid, min_length, max_length)


def validate_email_length_wrapper(email):
    """Wrapper to get setting values"""
    with dbConnect() as connection:
        min_length = int(get_setting(connection, 'minEmailLength'))
        max_length = int(get_setting(connection, 'maxEmailLength'))
    return validate_email_length(email, min_length, max_length)


# Aliases for the validation wrappers
checkNameLength = validate_name_length_wrapper
checkGrade = validate_grade_wrapper
checkCardidLength = validate_cardid_length_wrapper
checkEmailLength = validate_email_length_wrapper


# Wrapper for getSettingsValue to match the original function signature
def getSettingsValue_wrapper(settingName):
    with dbConnect() as connection:
        return get_setting(connection, settingName)


getSettingsValue = getSettingsValue_wrapper


def compareBase64Faces(test_image_base64, ref_image_base64, confidence_threshold):
    try:
        # Add padding to base64 strings if necessary
        def add_padding(base64_str):
            # Remove any data URL prefix if present
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
                
            missing_padding = len(base64_str) % 4
            if missing_padding:
                base64_str += '=' * (4 - missing_padding)
            return base64_str

        test_image_base64 = add_padding(test_image_base64)
        ref_image_base64 = add_padding(ref_image_base64)
        
        # Decode base64 strings to image data
        test_data = base64.b64decode(test_image_base64)
        ref_data = base64.b64decode(ref_image_base64)
        
        # Convert to PIL Images
        test_img = Image.open(BytesIO(test_data))
        ref_img = Image.open(BytesIO(ref_data))
        
        # Convert to RGB if necessary
        if test_img.mode != 'RGB':
            test_img = test_img.convert('RGB')
        if ref_img.mode != 'RGB':
            ref_img = ref_img.convert('RGB')
        
        # Convert to numpy arrays
        test_array = np.array(test_img)
        ref_array = np.array(ref_img)
        
        # Use DeepFace to verify the faces using numpy arrays
        result = DeepFace.verify(
            img1_path=test_array,
            img2_path=ref_array,
            model_name='Facenet512',
            distance_metric='cosine',
            enforce_detection=False,  # Set to False to handle cases where no face is detected
            detector_backend='opencv'
        )
        
        # Check if faces were detected in both images
        if not result.get('verified', False):
            return False
        
        # Convert distance to confidence (distance is between 0 and 1, lower is better)
        confidence = 1 - result['distance']
        return confidence >= confidence_threshold
        
    except Exception as e:
        # Handle any errors during the process
        print(f"Error during face comparison: {str(e)}")
        return False

def compress_image(base64_str, max_size=1024*1024):
    # Decode base64 to image data
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    
    # Convert image to RGB format if necessary for JPEG compression
    if image.mode in ('RGBA', 'LA'):
        # Handle transparency by pasting onto a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    quality = 95  # Start with high quality
    output_buffer = BytesIO()
    
    # Reduce quality iteratively until size is under max_size
    while quality > 10:
        output_buffer.seek(0)
        output_buffer.truncate(0)
        image.save(output_buffer, format='JPEG', quality=quality)
        current_size = output_buffer.tell()
        if current_size <= max_size:
            break
        quality -= 5
    else:
        # If quality reduction isn't sufficient, resize the image to half dimensions
        new_width = image.size[0] // 2
        new_height = image.size[1] // 2
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        output_buffer.seek(0)
        output_buffer.truncate(0)
        image.save(output_buffer, format='JPEG', quality=quality)
        # Note: Further checks could be added for resizing, but this is a fallback
    
    compressed_data = output_buffer.getvalue()
    compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
    return compressed_base64

def sendEmail(subject, body, recipient_email):
    msg = MIMEText(body, 'html')
    msg['Subject'] = subject
    msg['From'] = 'DPM - Exabyte Tech'
    msg['To'] = recipient_email
    
    try:
        with smtplib.SMTP_SSL(getSettingsValue('smtpServer'), 465) as smtp_server:
            smtp_server.login(getSettingsValue('smtpEmail'), getSettingsValue('smtpPassword'))

            smtp_server.sendmail(getSettingsValue('smtpEmail'), recipient_email, msg.as_string())

            return True
    except:
        return False

def validate_json_payload(payload: Union[str, dict]) -> bool:
    if payload == None:
        return True

    issues = []
    
    payload_dict = str(payload)

    # Check for suspicious patterns
    suspicious_patterns = [
        r"(?:UNION|SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXECUTE|EXEC|DECLARE|CAST)",
        r"(?:OR\s+1\s*=\s*1|--|\/\*)",
        r"(?:WHERE|FROM|GROUP|HAVING)",
        r"(?:UNION\s+ALL\s+SELECT)",
        r"(?:INSERT\s+INTO)",
        r"(?:UPDATE\s+[^\)]+\s+SET)",
        r"(?:DROP\s+TABLE)",
        r"(?:CREATE\s+TABLE)",
        r"(?:ALTER\s+TABLE)",
        r"(?:TRUNCATE\s+TABLE)"
    ]

    def scan_value(value):
        if isinstance(value, str):
            value_lower = value.lower()
            for pattern in suspicious_patterns:
                if re.search(pattern, value_lower):
                    issues.append(f"Suspicious SQL pattern detected: {value}, {pattern}")
        
        elif isinstance(value, dict):
            for v in value.values():
                scan_value(v)
                
        elif isinstance(value, list):
            for item in value:
                scan_value(item)

    scan_value(payload_dict)

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"- {issue}")
        return False

    return True


# Wrapper for isStudentSuspended from models
def isStudentSuspended_wrapper(studentid):
    with dbConnect() as connection:
        return check_student_suspended(connection, studentid, currentDatetime)


# Wrapper for sessionStorage - creates new class that delegates to models
class sessionStorage:
    @staticmethod
    def create(oid, keepstatedays, isstudent=False):
        with dbConnect() as connection:
            result = sessionStorage_model.create(
                connection, oid, keepstatedays,
                get_setting,
                lambda: get_current_datetime(convertTimezone),
                isstudent
            )
        return result
    
    @staticmethod
    def verify(sessionid, passkey):
        with dbConnect() as connection:
            return sessionStorage_model.verify(connection, sessionid, passkey, lambda: get_current_datetime(convertTimezone))
    
    @staticmethod
    def deactivate(sessionid, passkey):
        with dbConnect() as connection:
            return sessionStorage_model.deactivate(connection, sessionid, passkey)


# Import the model sessionStorage as sessionStorage_model to avoid naming conflict
from models import sessionStorage as sessionStorage_model


def verifyPassword_wrapper(userid, password):
    with dbConnect() as connection:
        return verifyPassword_model(connection, userid, password)
    
def getOidFromSession(session):
    sessionid_enc = session.get('sessionid')
    passkey_enc = session.get('passkey')

    if not sessionid_enc or not passkey_enc:
        raise Exception("Session ID or passkey missing please check" + str(session))
    
    sessionid = decrypt(str(sessionid_enc))
    passkey = decrypt(str(passkey_enc))

    oid = sessionStorage.verify(sessionid, passkey)[0]

    return oid


# Wrapper functions to delegate to models with simplified signatures
def getOidFromUserId_wrapper(userid):
    with dbConnect() as connection:
        return getOidFromUserId_model(connection, userid)


def getLocationNameFromId_original(locationid):
    with dbConnect() as connection:
        return getLocationNameFromId_model(connection, locationid)


def getStudentNameFromId_original(studentid):
    with dbConnect() as connection:
        return getStudentNameFromId_model(connection, studentid)


def getStudentGradeFromId_original(studentid):
    with dbConnect() as connection:
        return getStudentGradeFromId_model(connection, studentid)


def getStudentFloorIdFromId_original(studentid):
    with dbConnect() as connection:
        return getStudentFloorIdFromId_model(connection, studentid)


def getStudentCardidFromId_original(studentid):
    with dbConnect() as connection:
        return getStudentCardidFromId_model(connection, studentid)


def getStudentEmailFromId_original(studentid):
    with dbConnect() as connection:
        return getStudentEmailFromId_model(connection, studentid)


def getUserNameFromOid_original(oid):
    with dbConnect() as connection:
        return getUserNameFromOid_model(connection, oid)


def getUserIdFromOid_original(oid):
    with dbConnect() as connection:
        return getUserIdFromOid_model(connection, oid)


def getUserNameFromId_original(userid):
    with dbConnect() as connection:
        return getUserNameFromId_model(connection, userid)


def getUserEmailFromOid_original(oid):
    with dbConnect() as connection:
        return getUserEmailFromOid_model(connection, oid)


def getStudentImageFromId_original(studentid):
    with dbConnect() as connection:
        with connection.cursor() as dbcursor:
            dbcursor.execute('SELECT image FROM students WHERE studentid = %s', (studentid,))
            result = dbfetchedConvertDate(dbcursor.fetchall())
    
    if len(result) < 1:
        return None
    
    return result[0][0]


def getStudentInfoFromId_original(studentid):
    with dbConnect() as connection:
        return getStudentInfoFromId_model(connection, studentid)


def getScriptDir():
    return Path(__file__).resolve().parent


# Create aliases for wrapper functions to match the original function names throughout the code
# This allows us to use the models functions without changing all the calls
getOidFromUserId = getOidFromUserId_wrapper
getLocationNameFromId = getLocationNameFromId_original

# Provide a thin wrapper that matches the original single-argument API
# This calls the imported model function with a DB connection and keeps
# the expected signature for existing callers.
def getLocationIdFromName(location_name):
    with dbConnect() as connection:
        return getLocationIdFromName_model(connection, location_name)
getStudentNameFromId = getStudentNameFromId_original
getStudentGradeFromId = getStudentGradeFromId_original
getStudentFloorIdFromId = getStudentFloorIdFromId_original
getStudentCardidFromId = getStudentCardidFromId_original
getStudentEmailFromId = getStudentEmailFromId_original
getUserNameFromOid = getUserNameFromOid_original
getUserIdFromOid = getUserIdFromOid_original
getUserNameFromId = getUserNameFromId_original
getUserEmailFromOid = getUserEmailFromOid_original
getStudentImageFromId = getStudentImageFromId_original
getStudentInfoFromId = getStudentInfoFromId_original
isStudentSuspended = isStudentSuspended_wrapper
verifyPassword = verifyPassword_wrapper


class KIOSKPin:
    def generateNewKIOSKPin(userid):
        while True:
            newPin = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT userid FROM users WHERE kioskpin = %s', (newPin,))
                    result = dbfetchedConvertDate(dbcursor.fetchall())
            
            if len(result) < 1:
                break

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('UPDATE users SET kioskpin = %s WHERE userid = %s', (newPin, userid))

        return newPin
    
    def getKIOSKPin(userid):
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT kioskpin FROM users WHERE userid = %s', (userid,))
                result = dbfetchedConvertDate(dbcursor.fetchall())
        
        if len(result) < 1:
            return None
        
        return result[0][0]

    def verifyKIOSKPin(pin):
        if pin == None or pin == '':
            return False
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT userid FROM users WHERE kioskpin = %s', (str(pin),))
                result = dbfetchedConvertDate(dbcursor.fetchall())
        
        if len(result) < 1:
            return False
        
        KIOSKPin.generateNewKIOSKPin(result[0][0])
        
        return result[0][0]

def broadcastMasterCommand(command, payload = None, roles = ['admin', 'proctor', 'approver']):
    for role in roles:
        socketio.emit('command', {'command': command, 'payload': payload}, to=role)

def getStudentIdFromOid_wrapper(oid):
    """Wrapper for getStudentIdFromOid from models"""
    with dbConnect() as connection:
        result = getRecord(
            connection,
            "students",
            "studentid",
            "oid = %s",
            (oid,)
        )
    
    return result[0][0] if result else None


getStudentIdFromOid = getStudentIdFromOid_wrapper


logging.basicConfig(
    filename='./networklogs.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info('started')

def getLocationIdFromName_wrapper(location_name):
    """Wrapper for getLocationIdFromName from models"""
    with dbConnect() as connection:
        return getLocationIdFromName_model(connection, location_name)

@socketio.on('join')
def socketiojoin(data):    
    try:
        oid = getOidFromSession(session)
    except Exception:
        return

    if oid == None:
        return
    
    role = checkUserInformation("role", oid)
    userid_result = checkUserInformation("userid", oid)
    if userid_result is None:
        return
    userid = userid_result[0]

    if role == None:
        return
    
    role = role[0]

    if role == 1:
        role = 'admin'
    elif role == 2:
        role = 'proctor'
    elif role == 3:
        role = 'approver'

    join_room(role)
    join_room('user' + str(userid))

def log_response_info(response, ip=None):
    try:
        status = response.status
        headers = dict(response.headers)
        # Uncomment the next line if you want to log the response body (be careful with large or binary responses)
        body = response.get_data(as_text=True)

        if body != None:
            if len(body) > 1000:
                body = body[:1000]

        logging.info(
            "Response sent to %s < Status: %s | Headers: %s | Body: %s >\n", ip, status, headers, body
        )
    except Exception as e:
        logging.error("Failed to log response info: %s", str(e))
    return response

@app.before_request
def beforeRequest():
    try:
        # Exclude static files (js, css, images, fonts, etc.)
        static_exts = ('.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.woff', '.woff2', '.ttf', '.eot', '.map')
        if request.path.lower().endswith(static_exts) or request.path.startswith('/static/'):
            return  # Skip logging for static files

        ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        method = request.method
        path = request.path
        headers = dict(request.headers)
        args = request.args.to_dict()
        form = request.form.to_dict()
        data = request.get_data(as_text=True) if request.data else None

        if data != None:
            if len(data) > 1000:
                data = data[:1000]
            
        user_id = None
        student_id = None

        # Try to get user id and student id from session
        try:
            if 'sessionid' in session and 'passkey' in session:
                oid = getOidFromSession(session)
                user_info = checkUserInformation("userid", oid)
                if user_info:
                    user_id = user_info[0]
                student_id_val = getStudentIdFromOid(oid)
                if student_id_val:
                    student_id = student_id_val
        except Exception as e:
            pass  # Ignore errors in extracting user/student id

        logging.info(
            "Request from IP: %s < Method: %s | Path: %s | Headers: %s | Args: %s | Form: %s | Raw Data: %s | UserID: %s | StudentID: %s >\n",
            ip, method, path, headers, args, form, data, user_id, student_id
        )

        validation = validate_json_payload(data)

        if not validation:
            logging.warning("Attempted attack from IP: %s < UserID: %s | StudentID: %s | Raw Data: %s >\n",
            ip, user_id, student_id, data
            )

            if student_id != '' or student_id != None:
                studentName = getStudentNameFromId(student_id)
                suspend_student(student_id, f'[Auto Suspension] Detected malicious requests sent on behalf of account ({currentDatetime()})', None)
                try:
                    sendEmail('Student Auto Suspended', f'{studentName} has been suspended due to malicious activities.', getSettingsValue('adminEmail'))
                except:
                    pass

            return jsonify({'status': 'error', 'errorinfo': 'You have been detected for suspicious activities.'})
        
    except Exception as e:
        logging.error("Failed to log request info: %s", str(e))

@app.after_request
def afterRequest(response):
    # Exclude static files (js, css, images, fonts, etc.) from response logging
    static_exts = ('.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.woff', '.woff2', '.ttf', '.eot', '.map')
    if request.path.lower().endswith(static_exts) or request.path.startswith('/static/'):
        return response  # Skip logging for static files

    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net/ https://code.jquery.com/ https://*.mylivechat.com https://mylivechat.com;"
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://*.mylivechat.com/ https://mylivechat.com; "
        "font-src 'self' https://fonts.gstatic.com https://*.mylivechat.com/ https://mylivechat.com; "
        "img-src 'self' data: https://*.mylivechat.com/ https://mylivechat.com; "
        "connect-src 'self' data: https://cdnjs.cloudflare.com; "
        "frame-src 'self' https://*.mylivechat.com/ https://mylivechat.com/; "
        "object-src 'none'; "
        "base-uri 'self' https://*.mylivechat.com/ https://mylivechat.com/; "
        "form-action 'self' https://*.mylivechat.com/ https://mylivechat.com/;"
    )
    ip = request.headers.get('X-Forwarded-For', request.remote_addr)

    # Get user_id and student_id for logging
    user_id = None
    student_id = None
    try:
        if 'sessionid' in session and 'passkey' in session:
            oid = getOidFromSession(session)
            user_info = checkUserInformation("userid", oid)
            if user_info:
                user_id = user_info[0]
            student_id_val = getStudentIdFromOid(oid)
            if student_id_val:
                student_id = student_id_val
    except Exception:
        pass

    try:
        status = response.status
        headers = dict(response.headers)
        logging.info(
            "Response sent to %s < Status: %s | Headers: %s | UserID: %s | StudentID: %s >\n",
            ip, status, headers, user_id, student_id
        )
    except Exception as e:
        logging.error("Failed to log response info: %s", str(e))
    return response

@app.route('/')
def home():
    if ensureLoggedIn(session, 3):
        rejectionmessage = request.args.get('reject')
        dprint('rejectionmessage')
        return redirect('/passCatalog?reject=' + rejectionmessage) if rejectionmessage else redirect('/passCatalog')
    
    elif ensureLoggedIn(session, studentPortal = True):
        return redirect('/student')
    
    else:
        rejectionmessage = request.args.get('reject')
        if rejectionmessage != None:
            session.clear()
            return redirect('/')
        return render_template('signin.html', campusName = serverConfig["campusInfo"]["name"], campusAddress = serverConfig["campusInfo"]["address"][:25], campusInfo = f"{serverConfig['campusInfo']['phone']} | {serverConfig['campusInfo']['email']} | {serverConfig['campusInfo']['url']}", campusLogo = serverConfig["campusInfo"]["logo"], campusBackground = serverConfig["campusInfo"]["background"])
    
@app.route('/student')
def student():
    if ensureLoggedIn(session, studentPortal = True):
        oid = getOidFromSession(session)

        studentid = getStudentIdFromOid(oid)

        suspensionInfo = isStudentSuspended(studentid)
        suspensionInfoText = ''

        if suspensionInfo:
            suspensionInfoText = f'You are suspended: {suspensionInfo}'   
            
        return render_template('student.html', suspensionInfo = suspensionInfoText)
    else:
        return redirect('/?reject=You are not authorized to view this page')
   
@app.route('/west6')
def west6():
    return 'Best floor in Keystone!'
    
@app.route('/kiosk', methods=['GET', 'POST'])
def kiosk():
    if request.method == 'POST':
        pin = request.form['kioskpin']
        userid = KIOSKPin.verifyKIOSKPin(pin)
        if userid != False:
            oid = getOidFromUserId(userid)

            keepSessionDays = int(getSettingsValue('keepSessionDays'))
            sessioninfo = sessionStorage.create(oid, keepSessionDays)
            session['sessionid'] = str(encrypt(sessioninfo[0]))
            session['passkey'] = str(encrypt(sessioninfo[1]))
            session['login'] = True
            session['kiosk'] = True

            return render_template('kiosk.html')
        
        else:
            return render_template('errorPage.html', errorTitle = 'Error Logging In', errorText = 'Invalid KIOSK PIN', errorDesc = 'Please try again', errorLink = '/kiosk')
        
    if not ensureLoggedIn(session, 3):
        return render_template('kiosklogin.html')
    
    return render_template('kiosk.html')

@app.route('/DPM-KIOSK-SEB.seb')
def kioskConfig():
    return send_file(str(getScriptDir()) + '/DPM-KIOSK-SEB.seb', as_attachment=True)

@app.route('/getKioskConfigURL', methods=['GET'])
def getKioskConfigURL():
    if ensureLoggedIn(session, 3):
        serverUrl = getSettingsValue('serverURL')
        kioskConfigUrl = serverUrl + '/DPM-KIOSK-SEB.seb'
        kioskConfigUrl = kioskConfigUrl.replace('http', 'https').replace('https', 'seb')
        return kioskConfigUrl
    else:
        return redirect('/?reject=You are not authorized to view this page')
    
@app.route('/signout')
def signout():
    try:
        deactivateResult = sessionStorage.deactivate(decrypt(str(session.get('sessionid'))), decrypt(str(session.get('passkey'))))
    except:
        return redirect('/')
        
    if deactivateResult:
        session.clear()
        return redirect('/')
    else:
        return render_template('errorPage.html', errorTitle = 'Error Signing Out', errorText = 'Sever encountered an error while deactivating your session', errorDesc = 'Please try again later', errorLink = '/passCatalog')

@app.route('/userSignin')
def userSignin():
    return render_template('passwordLogin.html', campusName = serverConfig["campusInfo"]["name"], campusAddress = serverConfig["campusInfo"]["address"][:25], campusInfo = f"{serverConfig['campusInfo']['phone']} | {serverConfig['campusInfo']['email']} | {serverConfig['campusInfo']['url']}", campusLogo = serverConfig["campusInfo"]["logo"], campusBackground = serverConfig["campusInfo"]["background"])
    
@app.route('/mslogin')
def login():
    msal_app = get_msal_app()
    auth_url = msal_app.get_authorization_request_url(getSettingsValue('msauthscope').split('$'), redirect_uri=getSettingsValue('msauthRedirectURL'))
    return redirect(auth_url)

@app.route('/microsoftLoginCallback')
def microsoftLoginCallback():
    code = request.args.get('code')
    if not code:
        return render_template('errorPage.html', errorTitle = 'Microsoft Signin Error', errorText = 'Error in auth token', errorDesc = 'Please try signing in again', errorLink = '/'), 400
    msal_app = get_msal_app()
    result = msal_app.acquire_token_by_authorization_code(code, scopes=getSettingsValue('msauthscope').split('$'), redirect_uri=getSettingsValue('msauthRedirectURL'))
    dprint(result)
    if 'access_token' in result:
        msUserInfo = result.get('id_token_claims')
        oid = msUserInfo["oid"]
        email = msUserInfo["preferred_username"]
        urin = checkUserInformation("userid, name, oid, email, role, locationid", oid)
        stin = checkStudentInformation("studentid, name, oid, email, cardid, grade", oid)
        if urin != None:
            userInfo = urin
            dprint(userInfo)
            keepSessionDays = int(getSettingsValue('keepSessionDays'))
            sessioninfo = sessionStorage.create(oid, keepSessionDays, False)
            session['sessionid'] = str(encrypt(sessioninfo[0]))
            session['passkey'] = str(encrypt(sessioninfo[1]))

            session['login'] = True
            return redirect('/passCatalog')
        elif stin != None:
            studentInfo = stin
            dprint(studentInfo)
            keepSessionDays = int(getSettingsValue('keepSessionDays'))
            sessioninfo = sessionStorage.create(oid, keepSessionDays, True)
            session['sessionid'] = str(encrypt(sessioninfo[0]))
            session['passkey'] = str(encrypt(sessioninfo[1]))

            session['login'] = True
            return redirect('/student')
        
        if urin == None:
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT * FROM users WHERE email = %s', (email,))
                    result = dbfetchedConvertDate(dbcursor.fetchall())

            if len(result) > 0:
                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dbcursor.execute('UPDATE users SET oid = %s WHERE email = %s', (oid, email,))   
            
                keepSessionDays = int(getSettingsValue('keepSessionDays'))
                sessioninfo = sessionStorage.create(oid, keepSessionDays, False)
                session['sessionid'] = str(encrypt(sessioninfo[0]))
                session['passkey'] = str(encrypt(sessioninfo[1]))  

                userName = result[0][1] 

                session['login'] = True
                return render_template('firstLanding.html', userName = userName)
                
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT studentid, name, grade, cardid, floorid, disabledlocations, image, oid, email, suspension, suspensionED FROM students WHERE email = %s', (email,))
                    result = dbfetchedConvertDate(dbcursor.fetchall())

            if len(result) > 0:
                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dbcursor.execute('UPDATE students SET oid = %s WHERE email = %s', (oid, email,))
                session['login'] = True
                keepSessionDays = int(getSettingsValue('keepSessionDays'))
                sessioninfo = sessionStorage.create(oid, keepSessionDays, True)
                session['sessionid'] = str(sessioninfo[0])
                session['passkey'] = str(sessioninfo[1])

                session['login'] = True
                return redirect('/student')
            
            else:
                render_template('userNotRegistered.html', email = email, msUserInfo = msUserInfo, oid = oid)
                    
    else:
        return render_template('errorPage.html', errorTitle = 'Microsoft Signin Error', errorText = 'Our servers encountered an error while signing you in.', errorDesc = 'Please try signing in again', errorLink = '/'), 400
    
@app.route('/passwordLoginCallback', methods=['POST'])
def passwordLoginCallback():
    form_token = request.form.get('csrf_token')
    session_token = session.get('_csrf_token')
    if not form_token or not session_token or form_token != session_token:
        return render_template('errorPage.html', errorTitle="CSRF Error", errorText="Invalid CSRF token", errorDesc="Please try again.", errorLink="/userSignin")
    
    if captcha.validate():
        username = request.form['username']
        password = request.form['password']
        
        if username == '' or password == '' or username == None or password == None:
            return render_template('userNotRegistered.html', email = 'Password or username is empty')

        passwordhash = generateSHA256(password)

        dprint(passwordhash)

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT oid FROM users WHERE name = %s AND password = %s', (username, passwordhash,))
                result = dbfetchedConvertDate(dbcursor.fetchall())

        if len(result) > 0:
            dprint('Password login successful')
            oid = result[0][0]
            keepSessionDays = int(getSettingsValue('keepSessionDays'))
            sessioninfo = sessionStorage.create(oid, keepSessionDays)
            session['sessionid'] = str(encrypt(sessioninfo[0]))
            session['passkey'] = str(encrypt(sessioninfo[1]))
            session['login'] = True
            return redirect('/passCatalog')
        else:
            return render_template('userNotRegistered.html', email = username)
    else:
        return render_template('incorrectCaptcha.html')
    
@app.route('/api/checkUserPassword', methods=['POST'])
def checkUserPassword():
    if ensureLoggedIn(session, 3, kioskAllowed = True):
        password = request.json.get('password')
        
        retinfo = {}

        isCorrectPassword = verifyPassword(getUserIdFromOid(getOidFromSession(session)), password)

        retinfo['status'] = 'ok'
        retinfo['correct'] = isCorrectPassword

        return jsonify(retinfo)
    
    else:
        retinfo = {}

        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'

        return jsonify(retinfo)
    
@app.route('/api/searchLocations', methods=['POST'])
def searchLocations():
    if ensureLoggedIn(session, 3):
        retinfo = {}
        
        searchFilters = request.json.get('searchFilters')
        
        try:
            locationStrictName = searchFilters['strictname']
            
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT locationid, name, type FROM locations WHERE name = %s', (locationStrictName,))
                    dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                                
            retinfo['status'] = 'ok'
            retinfo['locations'] = dbcursorfetch
            
            return jsonify(retinfo)
        
        except:
            pass
        
        try:
            locationName = searchFilters['name']
            
            namekeywords = locationName.split()
            
            sqlquery = 'SELECT locationid, name, type FROM locations WHERE 1 = 1 '
            sqlqueryvar = []
            
            for keyword in namekeywords:
                sqlquery += 'AND name LIKE %s '
                sqlqueryvar.append(f'%{keyword}%')
                
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute(sqlquery, sqlqueryvar)
                    dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                                        
            retinfo['status'] = 'ok'
            retinfo['locations'] = dbcursorfetch
            
            return jsonify(retinfo)
        
        except:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Invalid search'
            
            return jsonify(retinfo),
        
@app.route('/api/editLocation', methods=['POST'])
def editLocation():
    if ensureLoggedIn(session, 1):
        retinfo = {}
        
        locationid = request.json.get('locationid')
        
        try:
            deleteLocation = request.json.get('delete')
            if deleteLocation == 'true':
                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dbcursor.execute('SELECT * FROM locations WHERE locationid = %s', (locationid,))
                        dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                        
                if len(dbcursorfetch) < 1:
                    retinfo['status'] = 'error'
                    retinfo['errorinfo'] = 'Location does not exist'
                    
                    return jsonify(retinfo)
                
                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dbcursor.execute('DELETE FROM locations WHERE locationid = %s', (locationid,))
                
                retinfo['status'] = 'ok'
                
                return jsonify(retinfo)
        except:
            pass
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT * FROM locations WHERE locationid = %s', (locationid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(dbcursorfetch) < 1:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Location does not exist'
            
            return jsonify(retinfo)
        
        locationName = request.json.get('name')
        
        if locationName == None or locationName == '':
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please fill in all of the required fields'
            
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT * FROM locations WHERE name = %s', (locationName,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(dbcursorfetch) > 1:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Name of location has already been taken'
            
            return jsonify(retinfo)
        
        locationType = request.json.get('type')
        
        if locationType == 'destination':
            locationType = 1
        elif locationType == 'dorm':
            locationType = 2
        else:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a valid location type'
            
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('UPDATE locations SET name = %s, type = %s WHERE locationid = %s', (locationName, locationType, locationid,))
                
        retinfo['status'] = 'ok'
        
        return jsonify(retinfo)
    
@app.route('/api/getLocationId', methods=['POST'])
def getLocationId():
    if ensureLoggedIn(session, 3, kioskAllowed=True):
        retinfo = {}
        
        locationtype = request.json.get('type')
        
        locations = getLocationsInformation(locationtype)
                        
        retinfo['status'] = 'ok'
        retinfo['locationJson'] = listToJson(locations)
        
        dprint(retinfo)
        
        
        return jsonify(retinfo)

    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/generateKioskToken', methods=['POST'])
def generateKioskToken():
    if ensureLoggedIn(session, 2):
        retinfo = {}
        
        oid = getOidFromSession(session)
        
        keepSessionDays = int(getSettingsValue('keepSessionDays'))
        sessioninfo = sessionStorage.create(oid, keepSessionDays)
        sessionid = str(sessioninfo[0])
        passskey = str(sessioninfo[1])
        
        kioskToken = encrypt(f'{sessionid}-{passskey}')
        
        retinfo['status'] = 'ok'
        retinfo['kioskToken'] = kioskToken
        
        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/updatePass', methods=['POST'])
def updatePass():
    if ensureLoggedIn(session, 3):
        retinfo = {}
        
        oid = getOidFromSession(session)
        
        urin = checkUserInformation("userid, name, oid, email, role, locationid", oid)
        if urin == None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Not authorized to perform this action'
            
            return jsonify(retinfo)
        userinfo = urin
        userid = userinfo[0]
        userlocation = userinfo[5]
        
        passid = request.json.get('passid')
        
        retinfo["elapsedtime"] = None
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT passid, studentid, floorid, destinationid, creationtime, fleavetime, flapprover, darrivetime, daapprover, dleavetime, dlapprover, farrivetime, faapprover, flagged, keywords FROM passes WHERE passid = %s', (passid,))
                result = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(result) < 1:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'invalidpass'
            
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT locationid FROM users WHERE userid = %s', (userid,))
                userlocationresult = dbfetchedConvertDate(dbcursor.fetchall())
        
        studentid = result[0][1]
        userlocationid = userlocationresult[0][0]
        floorid = result[0][2]
        destinationid = result[0][3]

        if userlocationid in [floorid, destinationid]:
            proxyApprove = False
        else:
            if ensureLoggedIn(session, 1):
                proxyApprove = True
            else:
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'User location does not match location of pass and user is not authorized to proxy approve passes'
            
                return jsonify(retinfo)

        suspendInfo = isStudentSuspended(studentid)

        if suspendInfo:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = f'Student is suspended: {suspendInfo}'

            return jsonify(retinfo)
        
        try:
            if not ensureLoggedIn(session, 2):
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Not authorized to perform this action'
            
                return jsonify(retinfo)
            
            delete = request.json.get('delete')
            if delete == 'true':
                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dbcursor.execute('SELECT passid, studentid, floorid, destinationid, creationtime, fleavetime, flapprover, darrivetime, daapprover, dleavetime, dlapprover, farrivetime, faapprover, flagged, keywords FROM passes WHERE passid = %s', (passid,))
                        result = dbfetchedConvertDate(dbcursor.fetchall())

                        if len(result) < 1:
                            retinfo['status'] = 'error'
                            retinfo['errorinfo'] = 'Pass does not exist'
                            return jsonify(retinfo)

                        dbcursor.execute('DELETE FROM passes WHERE passid = %s', (passid,))

                retinfo['status'] = 'ok'
                retinfo['message'] = 'Pass deleted successfully'
                return jsonify(retinfo)
        except KeyError:
            pass
                
        floorname = getLocationsInformation(2, floorid)[0][1]
        destinationname = getLocationsInformation(1, destinationid)[0][1]
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT name, grade FROM students WHERE studentid = %s', (studentid,))
                result = dbfetchedConvertDate(dbcursor.fetchall())
                
        dprint(result)
        
        studentname = result[0][0]
        studentgrade = 'Grade ' + str(result[0][1])
        
        sqlquery = 'UPDATE passes SET '
        sqlqueryvar = []
        
        updateflag = request.json.get('flag')
        if updateflag != None:
            sqlquery += 'flagged = true, '
        
        updatefloorid = request.json.get('floorid')
        if updatefloorid != None:
            floorname = getLocationsInformation(2, updatefloorid)[0][1]
            sqlquery += 'floorid = %s, '
            sqlqueryvar += str(updatefloorid)
        
        updatedestinationid = request.json.get('destinationid')
        if updatedestinationid != None:
            destinationname = getLocationsInformation(1, updatedestinationid)[0][1]
            sqlquery += 'destinationid = %s, '
            sqlqueryvar += str(updatedestinationid)

        updateapprove = request.json.get('approve')
        if updateapprove != None:
            timestamp = currentDatetime()

            stampposition = 0
            timepositions = ['fleavetime', 'darrivetime', 'dleavetime', 'farrivetime']
            approvepositions = ['flapprover', 'daapprover', 'dlapprover', 'faapprover']
            proxypositions = ['flpa', 'dapa', 'dlpa', 'fapa']
            
            stampposition = getPassStatus(passid)
                        
            if stampposition != None:
                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dbcursor.execute(f'UPDATE passes SET {timepositions[stampposition]} = "{timestamp}", {approvepositions[stampposition]} = %s, {proxypositions[stampposition]} = %s WHERE passid = %s', (userid, proxyApprove, passid,))
                        dprint(dbcursor.statement)
                        
            else:
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Cannot approve a completed pass'
                
                return jsonify(retinfo)
                    
            retinfo["elapsedtimewarning"] = None
                    
            if stampposition != 0 and stampposition != None:
                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dbcursor.execute(f"SELECT {timepositions[stampposition - 1]}, {timepositions[stampposition]} FROM passes WHERE passid = %s", (passid,))
                        stamptime = dbfetchedConvertDate(dbcursor.fetchall())
                        
                studentWarningTimeout = int(getSettingsValue('studentWarningTimeout'))
                studentAlertTimeout = int(getSettingsValue('studentAlertTimemout'))
                studentMinimumTimeout = int(getSettingsValue('studentMinimumTimeout'))
                
                dprint(studentMinimumTimeout)
                
                stamptime = stamptime[0]
                
                dprint(type(stamptime))
                dprint(stamptime)
                    
                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dbcursor.execute('SELECT fleavetime, darrivetime, dleavetime, farrivetime FROM passes WHERE passid = %s', (passid,))
                        curpass = dbfetchedConvertDate(dbcursor.fetchall())[0]
                        
                retinfo["elapsedtimewarning"] = None
                                
                elapsedtime = None
                    
                if (curpass[1] != None and curpass[2] == None) or (curpass[3] != None) :
                                        
                    elapsedSecond = calculateElapsedSeconds(stamptime[0])
                    
                    elapsedtime = convertSecondsToTime(elapsedSecond)                   

                    if elapsedSecond > studentAlertTimeout:
                        retinfo["elapsedtimewarning"] = 'alert'
                    elif elapsedSecond > studentWarningTimeout:
                        retinfo["elapsedtimewarning"] = 'warning'
                    elif elapsedSecond < studentMinimumTimeout:
                        retinfo["elapsedtimewarning"] = 'min'
                    
                retinfo["elapsedtime"] = elapsedtime
                
                dprint('set')
                dprint(studentWarningTimeout)
                dprint(studentAlertTimeout)
        
        sqlquery += 'keywords = %s WHERE passid = %s'
        sqlqueryvar += [f'{studentname} {studentgrade} {floorname} {destinationname}', passid]
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dprint(sqlquery)
                dprint(sqlqueryvar)
                dbcursor.execute(sqlquery, sqlqueryvar)

        retinfo['status'] = 'ok'
        
        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/approvePassByCard', methods=['POST'])
def approvePassByCard():
    retinfo = {}

    authResult = ensureLoggedIn(session, 3, kioskAllowed=True)

    if not authResult:
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        return jsonify(retinfo)
    
    cardid = request.json.get('cardid')
    if not cardid:
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Missing card ID'
        return jsonify(retinfo)

    # Get student ID from card ID
    with dbConnect() as connection:
        with connection.cursor() as dbcursor:
            dbcursor.execute('SELECT studentid FROM students WHERE cardid = %s', (cardid,))
            student_result = dbfetchedConvertDate(dbcursor.fetchall())
    if not student_result:
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Student not found'
        return jsonify(retinfo)
    studentid = student_result[0][0]

    retinfo['studentid'] = studentid

    studentTestImageBase64 = None

    studentName = getStudentNameFromId(studentid)
    studentFloorId = getStudentFloorIdFromId(studentid)

    oid = getOidFromSession(session)

    userid_result = checkUserInformation("userid", oid)
    if userid_result is None:
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        return jsonify(retinfo)
    userid = userid_result[0]

    if authResult == 'kiosk':
        isKiosk = True
    else:
        isKiosk = False

    if isKiosk == True:
        studentTestImageBase64 = request.json.get('image')

        if not studentTestImageBase64:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Missing test image'
            return jsonify(retinfo)
        
        studentImageBase64 = getStudentImageFromId(studentid)

        compareFaceResult = compareBase64Faces(studentTestImageBase64, studentImageBase64, 0.72)

        returnSocketInfo = {'studentName': studentName, 'studentFloorId': studentFloorId, 'studentImage': studentImageBase64, 'studentScanImage': studentTestImageBase64}

        if not compareFaceResult:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Face does not match'

            returnSocketInfo['status'] = 'error'
            returnSocketInfo['errorinfo'] = 'Face does not match'
        
            socketio.emit('kioskUpdate', returnSocketInfo, room='user' + str(userid))

            return jsonify(retinfo)

    suspendInfo = isStudentSuspended(studentid)

    if suspendInfo:
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = f'Student is suspended: {suspendInfo}'

        if isKiosk == True:
            returnSocketInfo['status'] = 'error'
            returnSocketInfo['errorinfo'] = f'Student is suspended: {suspendInfo}'

            socketio.emit('kioskUpdate', returnSocketInfo, room='user' + str(userid))

        return jsonify(retinfo)

    # Get the latest active pass for the student
    with dbConnect() as connection:
        with connection.cursor() as dbcursor:
            dbcursor.execute(
                'SELECT passid, floorid, destinationid, fleavetime, darrivetime, dleavetime, farrivetime, flagged '
                'FROM passes WHERE studentid = %s ORDER BY passid DESC LIMIT 1', (studentid,))
            pass_result = dbfetchedConvertDate(dbcursor.fetchall())
    if not pass_result:
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'No active pass found for student'

        if isKiosk == True:
            returnSocketInfo['status'] = 'error'
            returnSocketInfo['errorinfo'] = 'No active pass found for student'

            socketio.emit('kioskUpdate', returnSocketInfo, room='user' + str(userid))

        return jsonify(retinfo)

    passid, floorid, destinationid, fleavetime, darrivetime, dleavetime, farrivetime, flagged = pass_result[0]

    # Determine the next location to approve
    # 0: Dorm (fleavetime), 1: Destination (darrivetime), 2: Destination leave (dleavetime), 3: Dorm return (farrivetime)
    if fleavetime is None:
        next_location_id = floorid
        time_field = 'fleavetime'
        approver_field = 'flapprover'
    elif darrivetime is None:
        next_location_id = destinationid
        time_field = 'darrivetime'
        approver_field = 'daapprover'
    elif dleavetime is None:
        next_location_id = destinationid
        time_field = 'dleavetime'
        approver_field = 'dlapprover'
    elif farrivetime is None:
        next_location_id = floorid
        time_field = 'farrivetime'
        approver_field = 'faapprover'
    else:
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'No active pass found for student'

        if isKiosk == True:
            returnSocketInfo['status'] = 'error'
            returnSocketInfo['errorinfo'] = 'No active pass found for student'

            socketio.emit('kioskUpdate', returnSocketInfo, room='user' + str(userid))

        return jsonify(retinfo)
    
    retinfo['passid'] = passid

    # Get current user's location
    urin = checkUserInformation("userid, name, oid, email, role, locationid", oid)
    if urin is None:
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'User not found'

        if isKiosk == True:
            returnSocketInfo['status'] = 'error'
            returnSocketInfo['errorinfo'] = 'User not found'

            socketio.emit('kioskUpdate', returnSocketInfo, room='user' + str(userid))

        return jsonify(retinfo)
    userinfo = urin
    userid = userinfo[0]
    userlocation = userinfo[5]

    if flagged:
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'PASS IS FLAGGED Requires manual approval'

        if isKiosk == True:
            returnSocketInfo['status'] = 'error'
            returnSocketInfo['errorinfo'] = 'PASS IS FLAGGED Requires manual approval'

            socketio.emit('kioskUpdate', returnSocketInfo, room='user' + str(userid))

        return jsonify(retinfo)

    # Check if user's location matches the next required location
    if str(userlocation) != str(next_location_id):
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'User location does not match the next required location for approval'

        if isKiosk == True:
            returnSocketInfo['status'] = 'error'
            returnSocketInfo['errorinfo'] = 'User location does not match the next required location for approval'

            socketio.emit('kioskUpdate', returnSocketInfo, room='user' + str(userid))

        return jsonify(retinfo)
    
    warningPresent = False

    if time_field != 'fleavetime' and time_field != 'dleavetime' and time_field != None:
        lastpos = ''
        if time_field == 'darrivetime':
            lastpos = 'fleavetime'
        elif time_field == 'farrivetime':
            lastpos = 'dleavetime'
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute(f"SELECT {lastpos}, {time_field} FROM passes WHERE passid = %s", (passid,))
                stamptime = dbfetchedConvertDate(dbcursor.fetchall())
                
        studentWarningTimeout = int(getSettingsValue('studentWarningTimeout'))
        studentAlertTimeout = int(getSettingsValue('studentAlertTimemout'))
        studentMinimumTimeout = int(getSettingsValue('studentMinimumTimeout'))
        
        dprint(studentMinimumTimeout)
        
        stamptime = stamptime[0]
        
        dprint(type(stamptime))
        dprint(stamptime)
            
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT fleavetime, darrivetime, dleavetime, farrivetime FROM passes WHERE passid = %s', (passid,))
                curpass = dbfetchedConvertDate(dbcursor.fetchall())[0]
                
        retinfo["elapsedtimewarning"] = None
        
        dprint(curpass)
        
        elapsedtime = None
            
        if (curpass[0] != None and curpass[1] == None) or (curpass[2] != None and curpass[3] != None) :
                                
            elapsedSecond = calculateElapsedSeconds(stamptime[0])
            
            elapsedtime = convertSecondsToTime(elapsedSecond)                  
            if elapsedSecond > studentAlertTimeout:
                warningPresent = True
                retinfo["elapsedtimewarning"] = 'alert'
            elif elapsedSecond > studentWarningTimeout:
                warningPresent = True
                retinfo["elapsedtimewarning"] = 'warning'
            elif elapsedSecond < studentMinimumTimeout:
                warningPresent = True
                retinfo["elapsedtimewarning"] = 'min'
            
        retinfo["elapsedtime"] = elapsedtime

    isKiosk = request.json.get('kiosk')

    if warningPresent and isKiosk == True:
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Elapsed time warning present, kiosk approval not allowed. Please find your duty teacher to approve this pass'

        if isKiosk == True:
            returnSocketInfo['status'] = 'error'
            returnSocketInfo['errorinfo'] = 'Elapsed time warning present, kiosk approval not allowed. Please find your duty teacher to approve this pass'

            socketio.emit('kioskUpdate', returnSocketInfo, room='user' + str(userid))

        return jsonify(retinfo)
    
    image_field = approver_field.replace('approver', 'image')
    ka_field = approver_field.replace('approver', 'ka')
    
    timestamp = currentDatetime()
    with dbConnect() as connection:
        with connection.cursor() as dbcursor:
            dbcursor.execute(
                f'UPDATE passes SET {time_field} = %s, {approver_field} = %s, {image_field} = %s, {ka_field} = %s WHERE passid = %s',
                (timestamp, userid, studentTestImageBase64, isKiosk, passid)
            )
    
    retinfo['status'] = 'ok'

    if isKiosk == True:
        returnSocketInfo['status'] = 'ok'
        socketio.emit('kioskUpdate', returnSocketInfo, room='user' + str(userid))

    return jsonify(retinfo)
    
@app.route('/api/getStudents', methods=['POST'])
def getStudents():
    if ensureLoggedIn(session, 3, kioskAllowed=True):
        retinfo = {}
        
        oid = getOidFromSession(session)
        
        urin = checkUserInformation("userid, name, oid, email, role, locationid", oid)

        if urin == None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Not authorized to perform this action'
            
            return jsonify(retinfo)
        
        searchScope = request.json.get('searchScope')
        userinfo = urin
        userid = userinfo[0]
        userlocation = userinfo[5]
        dateScope = request.json.get('dateScope')

        try:
            if searchScope != 'Use Current Location':
                userlocation = getLocationIdFromName(searchScope)

            userlocationtype = getLocationType(userlocation)
        except:
            pass
                
        searchfilters = request.json.get('filter')
        
        sqlquery = "SELECT passid, studentid, floorid, destinationid, fleavetime, darrivetime, dleavetime, farrivetime, flagged FROM passes WHERE 1 = 1 "
        sqlqueryvar = []
        
        allfilter = False
        
        dftrue = False
        
        try:
            if searchfilters['all']:
                allfilter = True
        except KeyError:
            pass

        if dateScope != '' and dateScope != None:
            sqlquery += "AND (DATE(creationtime) = %s OR DATE(fleavetime) = %s OR DATE(darrivetime) = %s OR DATE(dleavetime) = %s OR DATE(farrivetime) = %s) "
            sqlqueryvar += [dateScope, dateScope, dateScope, dateScope, dateScope]
        
        try:
            studentfilter = searchfilters['studentid']
            sqlquery += "AND studentid = %s "
            sqlqueryvar += [studentfilter]
            dftrue = True
        except KeyError:
            pass
        
        try:
            destinationfilter = searchfilters['destination']
            sqlquery += "AND destinationid = %s "
            sqlqueryvar += [destinationfilter]
            dftrue = True
        except:
            pass
        
        try:
            floorfilter = searchfilters['floor']
            sqlquery += "AND floorid = %s "
            sqlqueryvar += [floorfilter]
            dftrue = True
        except:
            pass
        
        if not allfilter and not dftrue:
            try:
                if userlocationtype == 1:
                    sqlquery += "AND destinationid = %s "
                    sqlqueryvar += [userlocation]
                else:
                    sqlquery += "AND floorid = %s "
                    sqlqueryvar += [userlocation]
            except KeyError:
                pass            
            
        try:
            flagfilter = searchfilters['flag']
            if flagfilter:
                sqlquery += "AND flagged = TRUE "
        except KeyError:
            pass
        
        try:
            statusfilter = int(searchfilters['status'])
            if userlocationtype == 2:
                statusfilter -= 2
            nullstatus = ['AND fleavetime IS NULL ', 'AND fleavetime IS NOT NULL AND darrivetime IS NULL ', 'AND darrivetime IS NOT NULL AND dleavetime IS NULL ', 'AND dleavetime IS NOT NULL AND farrivetime IS NULL ']
            dprint(nullstatus[statusfilter])
            sqlquery += nullstatus[statusfilter]
        except KeyError:
            pass
        
        try:
            searchfilter = str(searchfilters['search'])
            searchkeywords = searchfilter.split()
            dprint(searchkeywords)
            for keyword in searchkeywords:
                dprint(keyword)
                sqlquery += 'AND keywords LIKE %s '
                sqlqueryvar.append(f'%{keyword}%')
        except KeyError:
            pass
        
        try:
            includeCompletedPasses = request.json.get('showCompletedPass')
            if not includeCompletedPasses:
                sqlquery += 'AND farrivetime IS NULL '
        except KeyError:
            sqlquery += 'AND farrivetime IS NULL '
        
        dprint(sqlquery)
        dprint(sqlqueryvar)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dprint(sqlquery)
                dprint(sqlqueryvar)
                dbcursor.execute(sqlquery, sqlqueryvar)
                dprint('execed')
                dprint(dbcursor.statement)
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        studentWarningTimeout = int(getSettingsValue('studentWarningTimeout'))
        studentAlertTimeout = int(getSettingsValue('studentAlertTimemout'))
                
        curpasscur = 0
        for curpass in dbcursorfetch:
            studentinfo = getStudentInfoFromId(curpass[1])

            if (curpass[4] != None and curpass[5] == None) or (curpass[6] != None and curpass[7] == None) :
                for i in range(4):
                    if curpass[-2 - i] != None:
                        elapsedSecond = calculateElapsedSeconds(curpass[-2 - i])
        
                        if elapsedSecond > studentAlertTimeout:
                            dbcursorfetch[curpasscur] += ('alert',)
                        elif elapsedSecond > studentWarningTimeout:
                            dbcursorfetch[curpasscur] += ('warning',)
                        else:
                            dbcursorfetch[curpasscur] += (None,)
        
                        break
            else:
                dbcursorfetch[curpasscur] += (None,)
        
            dbcursorfetch[curpasscur] += (studentinfo,)

            dbcursorfetch[curpasscur] = list(dbcursorfetch[curpasscur])

            for i in range(4):
                if dbcursorfetch[curpasscur][i + 4] != None:
                    dbcursorfetch[curpasscur][i + 4] = str(dbcursorfetch[curpasscur][i + 4])
        
            curpasscur += 1
                
        retinfo['status'] = 'ok'
        retinfo['students'] = dbcursorfetch
        dprint(dbcursorfetch)
        return jsonify(retinfo) 
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/addStudent', methods=['POST'])
def addStudent():
    if ensureLoggedIn(session, 1):
        retinfo = {}
        
        studentName = request.json.get('name')
        studentGrade = request.json.get('grade')
        studentFloor = request.json.get('floor')
        studentCardid = request.json.get('cardid')
        studentEmail = request.json.get('email')
        studentImage = request.json.get('image')
        
        dprint([studentName, studentGrade, studentFloor, studentCardid])
        
        if studentName == None or studentGrade == None or studentFloor == None or studentName == '' or studentGrade == '' or studentFloor == '' or studentEmail == None or studentEmail == '':
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please fill in all of the required fields'
            
            return jsonify(retinfo)
        
        if checkNameLength(studentName) == False:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a name with a valid length'
            
            return jsonify(retinfo)
        
        if checkGrade(studentGrade) == False:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a valid grade within a valid range'
            
            return jsonify(retinfo)
        
        if checkCardidLength(studentCardid) == False:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a card ID with a valid length'
            
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT studentid, name, grade, cardid, floorid, disabledlocations, image, oid, email, suspension, suspensionED FROM students WHERE name = %s', (studentName,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(dbcursorfetch) > 0:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Name of student has already been taken'
            
            return jsonify(retinfo)
        
        if studentImage != None and studentImage != '':   
            if not studentImage.startswith('data:image/png;base64,iVBORw0KGgo'):
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Please upload a valid image'

                return jsonify(retinfo)

            if len(studentImage) > 2100000:
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Please upload an image that is less than 1.5MB'

                return jsonify(retinfo)
            
        else:
            studentImage = None
            
        if studentCardid != None and studentCardid != '':
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT studentid, name, grade, cardid, floorid, disabledlocations, image, oid, email, suspension, suspensionED FROM students WHERE cardid = %s', (studentCardid,))
                    dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())

            if len(dbcursorfetch) > 0:
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Card ID has already been taken'

                return jsonify(retinfo)
        else:
            studentCardid = None

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT studentid, name, grade, cardid, floorid, disabledlocations, image, oid, email, suspension, suspensionED FROM students WHERE email = %s', (studentEmail,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())

        if len(dbcursorfetch) > 0:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Email of student has already been taken'
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT * FROM users WHERE email = %s', (studentEmail,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())

        if len(dbcursorfetch) > 0:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Email of student has already been taken'
            
            return jsonify(retinfo)

        try: 
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT type, locationid FROM locations WHERE name = %s', (studentFloor,))
                    dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                    
            dprint(dbcursorfetch)

            if len(dbcursorfetch) < 1:
                dprint('de')
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Please enter a valid floor name'

                return jsonify(retinfo)

            if dbcursorfetch[0][0] != 2:
                dprint('dl')
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Location selected is a destination'

                return jsonify(retinfo)
            
            studentFloorId = dbcursorfetch[0][1]
        except:
            dprint('dd')
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a valid floor name'

            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('INSERT INTO students (name, grade, floorid, cardid, image, email) VALUES (%s, %s, %s, %s, %s, %s)', (studentName, studentGrade, studentFloorId, studentCardid, studentImage, studentEmail,))
                dbcursor.execute('SELECT LAST_INSERT_ID()')
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        retinfo['status'] = 'ok'
        retinfo['studentid'] = dbcursorfetch[0][0]
        
        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/getUserLocation', methods=['POST'])
def getUserLocation():
    if ensureLoggedIn(session, 3, kioskAllowed=True):
        retinfo = {}
        
        oid = getOidFromSession(session)
        
        urin = checkUserInformation("locationid", oid)
        if urin == None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Not authorized to perform this action'
            
            return jsonify(retinfo)
        
        locationid = urin[0]
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT name FROM locations WHERE locationid = %s', (locationid,))
                result = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(result) < 1:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Location not found'
            
            return jsonify(retinfo)
        
        locationName = result[0][0]
        
        retinfo['status'] = 'ok'
        retinfo['location'] = locationName
        
        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/addUser', methods=['POST'])
def addUser():
    if ensureLoggedIn(session, 1):
        retinfo = {}
        
        userName = request.json.get('name')
        userEmail = request.json.get('email')
        userRole = request.json.get('role')
        userLocation = request.json.get('location')
        userPassword = request.json.get('password')
        
        dprint([userName, userEmail, userRole, userLocation, userPassword])
        
        if userName == None or userEmail == None or userRole == None or userLocation == None or userName == '' or userEmail == '' or userRole == '' or userLocation == '':
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please fill in all of the required fields'
            
            return jsonify(retinfo)
        
        if checkNameLength(userName) == False:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a name with a valid length'
            
            return jsonify(retinfo)
        
        if checkEmailLength(userEmail) == False:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter an email with a valid length'
            
            return jsonify(retinfo)
                
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT * FROM users WHERE name = %s', (userName,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(dbcursorfetch) > 0:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Name of user has already been taken'
            
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT * FROM users WHERE email = %s', (userEmail,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(dbcursorfetch) > 0:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Email of user has already been taken'
            
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT studentid, name, grade, cardid, floorid, disabledlocations, image, oid, email, suspension, suspensionED FROM students WHERE email = %s', (userEmail,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())

        if len(dbcursorfetch) > 0:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Email of user has already been taken by a student'
            
            return jsonify(retinfo)
        
        if userPassword == None or userPassword == '':
            userPassword = None
        else:
            if len(userPassword) < 8:
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Password must be at least 8 characters long'
                
                return jsonify(retinfo)
            
            if len(userPassword) > 32:
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Password must be less than 32 characters long'
                
                return jsonify(retinfo)
            
            userPassword = generateSHA256(userPassword)

        userLocationId = getLocationIdFromName(userLocation)

        if userLocationId == None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a valid location'
            
            return jsonify(retinfo)
        
        if userRole == 'admin':
            userRoleId = 1
        elif userRole == 'proctor':
            userRoleId = 2
        elif userRole == 'approver':
            userRoleId = 3
        else:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a valid role'
            
            return jsonify(retinfo)
        
        userOid = 'usernopcoid' + userEmail + str(random.randint(100000, 999999))
                
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('INSERT INTO users (name, email, role, locationid, password, oid) VALUES (%s, %s, %s, %s, %s, %s)', (userName, userEmail, userRoleId, userLocationId, userPassword, userOid,))
                dbcursor.execute('SELECT LAST_INSERT_ID()')
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())    
                
        retinfo['status'] = 'ok'
        retinfo['userid'] = dbcursorfetch[0][0]
        
        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/addLocation', methods=['POST'])
def addLocation():
    if ensureLoggedIn(session, 1):
        retinfo = {}
        
        locationName = request.json.get('name')
        locationType = request.json.get('type')
        
        if checkNameLength(locationName) == False:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a name with a valid length'
            
            return jsonify(retinfo)
        
        dprint([locationName, locationType])
        
        if locationName == None or locationType == None or locationName == '' or locationType == '':
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please fill in all of the required fields'
            
            return jsonify(retinfo)
                
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT * FROM locations WHERE name = %s', (locationName,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(dbcursorfetch) > 0:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Name of location has already been taken'
            
            return jsonify(retinfo)
        
        if locationType == 'dorm':
            locationType = 2
        elif locationType == 'destination':
            locationType = 1
        else:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a valid location type'
            
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('INSERT INTO locations (name, type) VALUES (%s, %s)', (locationName, locationType,))
                dbcursor.execute('SELECT LAST_INSERT_ID()')
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        retinfo['status'] = 'ok'
        retinfo['locationid'] = dbcursorfetch[0][0]
        
        return jsonify(retinfo)
    
@app.route('/api/editStudent', methods=['POST'])
def editStudent():
    if ensureLoggedIn(session, 1):
        retinfo = {}
        
        studentid = request.json.get('studentid')
        
        try:
            deleteStudent = request.json.get('delete')
            if deleteStudent == 'true':
                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dbcursor.execute('SELECT studentid, name, grade, cardid, floorid, disabledlocations, image, oid, email, suspension, suspensionED FROM students WHERE studentid = %s', (studentid,))
                        dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                        
                if len(dbcursorfetch) < 1:
                    retinfo['status'] = 'error'
                    retinfo['errorinfo'] = 'Student does not exist'
                    
                    return jsonify(retinfo)
                
                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dbcursor.execute('DELETE FROM students WHERE studentid = %s', (studentid,))
                
                retinfo['status'] = 'ok'
                
                return jsonify(retinfo)
        except:
            pass
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT studentid, name, grade, cardid, floorid, disabledlocations, image, oid, email, suspension, suspensionED FROM students WHERE studentid = %s', (studentid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(dbcursorfetch) < 1:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Student does not exist'
            
            return jsonify(retinfo)
        
        studentName = request.json.get('name')
        studentGrade = request.json.get('grade')
        studentFloor = request.json.get('floor')
        studentCardid = request.json.get('cardid')
        studentEmail = request.json.get('email')
        studentImage = request.json.get('image')
        studentSuspensionInfo = request.json.get('suspension')
        studentSuspensionEndDate = request.json.get('suspensionED')
        
        if checkNameLength(studentName) == False:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a name with a valid length'
            
            return jsonify(retinfo)
        
        if checkGrade(studentGrade) == False:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a valid grade within a valid range'
            
            return jsonify(retinfo)
        
        if checkCardidLength(studentCardid) == False:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a card ID with a valid length'
            
            return jsonify(retinfo)
        
        dprint([studentid, studentName, studentGrade, studentFloor, studentCardid])
        
        if studentName == None or studentGrade == None or studentFloor == None or studentName == '' or studentGrade == '' or studentFloor == '' or studentEmail == None or studentEmail == '':
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please fill in all of the required fields'
            
            return jsonify(retinfo)
                
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT studentid, name, grade, cardid, floorid, disabledlocations, image, oid, email, suspension, suspensionED FROM students WHERE name = %s AND studentid != %s', (studentName, studentid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(dbcursorfetch) > 0:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Name of student has already been taken'
            
            return jsonify(retinfo)
        
        if studentImage != None and studentImage != '':   
            if not studentImage.startswith('data:image/png;base64,iVBORw0KGgo'):
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Please upload a valid png image'

                return jsonify(retinfo)

            if len(studentImage) > 2100000:
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Please upload an image that is less than 1.5MB'

                return jsonify(retinfo)
            
        else:
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT image FROM students WHERE studentid = %s', (studentid,))
                    dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                    
            studentImage = dbcursorfetch[0][0]
            
        if studentCardid != None and studentCardid != '':
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT studentid FROM students WHERE cardid = %s', (studentCardid,))
                    dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())

            if len(dbcursorfetch) > 0 and dbcursorfetch[0][0] != studentid:
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Card ID has already been taken'

                return jsonify(retinfo)
        else:
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT image FROM students WHERE studentid = %s', (studentid,))
                    dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                    
            studentCardid = dbcursorfetch[0][0]
        
        try: 
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT type, locationid FROM locations WHERE name = %s', (studentFloor,))
                    dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                    
            dprint(dbcursorfetch)

            if len(dbcursorfetch) < 1:
                dprint('de')
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Please enter a valid floor name'

                return jsonify(retinfo)

            if dbcursorfetch[0][0] != 2:
                dprint('dl')
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Location selected is a destination'

                return jsonify(retinfo)
            
            studentFloorId = dbcursorfetch[0][1]
        except:
            dprint('dd')
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a valid floor name'

            return jsonify(retinfo)
        
        try:
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT email FROM students WHERE email = %s AND studentid != %s', (studentEmail, studentid,))
                    dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                    
            if len(dbcursorfetch) > 0:
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Email of student has already been taken'
                
                return jsonify(retinfo)
        except:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a valid email'
            
            return jsonify(retinfo)
        
        try:
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT email FROM users WHERE email = %s', (studentEmail,))
                    dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
            if len(dbcursorfetch) > 0:
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Email of student has already been taken by a user'
                
                return jsonify(retinfo)
        except:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a valid email'
            
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('UPDATE students SET name = %s, grade = %s, floorid = %s, cardid = %s, image = %s, email = %s, suspension = %s, suspensionED = %s WHERE studentid = %s', (studentName, studentGrade, studentFloorId, studentCardid, studentImage, studentEmail, studentSuspensionInfo, studentSuspensionEndDate, studentid,))
        
        retinfo['status'] = 'ok'
        
        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
                
@app.route('/api/editUser', methods=['POST'])
def editUser():
    if (request.json.get('settingsEdit') == 'true' and ensureLoggedIn(session, 3)) or ensureLoggedIn(session, 1):
        retinfo = {}
        
        oid = getOidFromSession(session)
        settingsEdit = request.json.get('settingsEdit')
        setuserid_result = checkUserInformation("userid", oid)
        setuserrole_result = checkUserInformation("role", oid)
        if setuserid_result is None or setuserrole_result is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Not authorized to perform this action'
            return jsonify(retinfo)
        setuserid = setuserid_result[0]
        setuserrole = setuserrole_result[0]

        editUserLocation = None

        if settingsEdit == 'true':
            seTrue = True
            
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT role, locationid FROM users WHERE userid = %s', (setuserid,))
                    dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                    
            userRoleId = dbcursorfetch[0][0]
            if not ensureLoggedIn(session, 1):
                editUserLocation = dbcursorfetch[0][1]
            userid = setuserid
        else:
            seTrue = False
            userid = request.json.get('userid')

        try:
            deleteUser = request.json.get('delete')
            if deleteUser == 'true':
                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dbcursor.execute('SELECT * FROM users WHERE userid = %s', (userid,))
                        dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                        
                if len(dbcursorfetch) < 1:
                    retinfo['status'] = 'error'
                    retinfo['errorinfo'] = 'User does not exist'
                    
                    return jsonify(retinfo)
                
                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dbcursor.execute('DELETE FROM users WHERE userid = %s', (userid,))
                
                retinfo['status'] = 'ok'
                
                return jsonify(retinfo)
        except:
            pass
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT * FROM users WHERE userid = %s', (userid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(dbcursorfetch) < 1:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'User does not exist'
            
            return jsonify(retinfo)
        
        userName = request.json.get('name')
        userEmail = request.json.get('email')
        userRole = request.json.get('role')
        userLocation = request.json.get('location')
        userPassword = request.json.get('password')
        
        if checkNameLength(userName) == False:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a name with a valid length'
            
            return jsonify(retinfo)
        
        if checkEmailLength(userEmail) == False:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter an email with a valid length'
            
            return jsonify(retinfo)
        
        dprint([userName, userEmail, userRole, userLocation, userPassword])
        
        if userName == None or userEmail == None or (userRole == None and seTrue != True) or userLocation == None or userName == '' or userEmail == '' or (userRole == '' and seTrue != True) or userLocation == '':
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please fill in all of the required fields'
            
            return jsonify(retinfo)
                
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT * FROM users WHERE name = %s AND userid != %s', (userName, userid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(dbcursorfetch) > 0:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Name of user has already been taken'
            
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT * FROM users WHERE email = %s AND userid != %s', (userEmail, userid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(dbcursorfetch) > 0:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Email of user has already been taken'
            
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT studentid, name, grade, cardid, floorid, disabledlocations, image, oid, email, suspension, suspensionED FROM students WHERE email = %s', (userEmail,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())

        if len(dbcursorfetch) > 0:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Email of user has already been taken by a student'
            
            return jsonify(retinfo)
        
        if userPassword == None or userPassword == '':
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT password FROM users WHERE userid = %s', (userid,))
                    dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                    
            userPassword = dbcursorfetch[0][0]
            
        else:
            userPassword = generateSHA256(userPassword)
            
        if editUserLocation == None:
            userLocationId = getLocationIdFromName(userLocation)
        else:
            userLocationId = editUserLocation

        if userLocationId == None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Please enter a valid location'
            
            return jsonify(retinfo)
        
        if userRole == 'admin':
            userSetRole = 1
        elif userRole == 'proctor':
            userSetRole = 2
        elif userRole == 'approver':
            userSetRole = 3
        else:
            userSetRole = 0

        if not seTrue and userid != setuserid:
            if userSetRole == 0:
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'Please enter a valid role'

                return jsonify(retinfo)
            
        elif userRole != None and userSetRole != setuserrole:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'You cannot change your own role'

            return jsonify(retinfo)
        
        if userSetRole == 0:
            userRoleId = setuserrole
        else:
            userRoleId = userSetRole
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('UPDATE users SET name = %s, email = %s, role = %s, locationid = %s, password = %s WHERE userid = %s', (userName, userEmail, userRoleId, userLocationId, userPassword, userid,))
        
        retinfo['status'] = 'ok'
        
        return jsonify(retinfo)
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/searchStudents', methods=['POST'])
def searchStudents():
    if ensureLoggedIn(session, 3):
        retinfo = {}
        
        searchFilter = request.json.get('searchFilter')
        strictNameSearch = False
        
        dprint(searchFilter)
        
        sqlquery = "SELECT studentid, name, grade, cardid, floorid, disabledlocations, email, suspension, suspensionED FROM students WHERE 1 = 1 "
        sqlqueryvar = []
        
        try:
            studentIdFilter = int(searchFilter['studentid'])
            dprint(studentIdFilter)
            sqlquery += 'AND studentid = %s '
            sqlqueryvar.append(studentIdFilter)
        except KeyError:
            pass
        except TypeError:
            pass
        
        try:
            nameFilter = str(searchFilter['name'])
            nameKeywords = nameFilter.split()
            dprint(nameKeywords)
            for keyword in nameKeywords:
                dprint(keyword)
                sqlquery += 'AND name LIKE %s '
                sqlqueryvar.append(f'%{keyword}%')
        except KeyError:
            pass
        except TypeError:
            pass
        
        try:
            strictNameFilter = str(searchFilter['strictname'])
            strictNameSearch = True
            dprint(strictNameFilter)
            sqlquery += 'AND name = %s '
            sqlqueryvar.append(strictNameFilter)
        except KeyError:
            pass    
        except TypeError:
            pass
        
        try:
            gradeFilter = int(searchFilter['grade'])
            dprint(gradeFilter)
            sqlquery += 'AND grade = %s '
            sqlqueryvar.append(gradeFilter)
        except KeyError:
            pass
        except TypeError:
            pass
        
        try:
            cardidFilter = str(searchFilter['cardid'])
            dprint(cardidFilter)
            sqlquery += 'AND cardid = %s '
            sqlqueryvar.append(cardidFilter)
        except KeyError:
            pass
        except TypeError:
            pass
        
        try:
            flooridFilter = str(searchFilter['floorid'])
            dprint(flooridFilter)
            sqlquery += 'AND floorid = %s '
            sqlqueryvar.append(flooridFilter)
        except KeyError:
            pass
        except TypeError:
            pass
        
        dprint(sqlquery)
        dprint(sqlqueryvar)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dprint(sqlquery)
                dprint(sqlqueryvar)
                dbcursor.execute(sqlquery, sqlqueryvar)
                dprint('execed')
                dprint(dbcursor.statement)
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
        
        if strictNameSearch:
            try:
                isStudentSuspended(dbcursorfetch[0][0])

                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dprint(sqlquery)
                        dprint(sqlqueryvar)
                        dbcursor.execute(sqlquery, sqlqueryvar)
                        dprint('execed')
                        dprint(dbcursor.statement)
                        dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
            except IndexError:
                pass
                
        querySearchLimit = int(getSettingsValue('querySearchLimit'))
                
        retinfo['status'] = 'ok'
        retinfo['students'] = dbcursorfetch[:querySearchLimit]

        return jsonify(retinfo)   
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/searchUsers', methods=['POST'])
def searchUsers():
    if ensureLoggedIn(session, 3):
        retinfo = {}
        
        searchFilter = request.json.get('searchFilter')
        
        dprint(searchFilter)
        
        sqlquery = "SELECT userid, name, email, role, locationid FROM users WHERE 1 = 1 "
        sqlqueryvar = []
        
        try:
            settingsEdit = str(searchFilter['settingsEdit'])
            if settingsEdit == 'true':
                oid = getOidFromSession(session)
                userid_result = checkUserInformation("userid", oid)
                if userid_result is None:
                    retinfo['status'] = 'error'
                    retinfo['users'] = None
                    return jsonify(retinfo)
                userid = userid_result[0]
                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dbcursor.execute('SELECT userid, name, email, role, locationid FROM users WHERE userid = %s', (userid,))
                        dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                        
                retinfo['status'] = 'ok'
                retinfo['users'] = dbcursorfetch[0]
                
                return jsonify(retinfo)
        except KeyError:
            pass
                
        try:
            nameFilter = str(searchFilter['name'])
            nameKeywords = nameFilter.split()
            dprint(nameKeywords)
            for keyword in nameKeywords:
                dprint(keyword)
                sqlquery += 'AND name LIKE %s '
                sqlqueryvar.append(f'%{keyword}%')
        except KeyError:
            pass
        
        try:
            strictNameFilter = str(searchFilter['strictname'])
            dprint(strictNameFilter)
            sqlquery += 'AND name = %s '
            sqlqueryvar.append(strictNameFilter)
        except KeyError:
            pass
        
        try:
            emailFilter = str(searchFilter['email'])
            dprint(emailFilter)
            sqlquery += 'AND email = %s '
            sqlqueryvar.append(emailFilter)
        except KeyError:
            pass
        
        try:
            roleFilter = str(searchFilter['role'])
            dprint(roleFilter)
            sqlquery += 'AND role = %s '
            sqlqueryvar.append(roleFilter)
        except KeyError:
            pass
        
        try:
            locationFilter = str(searchFilter['location'])
            dprint(locationFilter)
            sqlquery += 'AND locationid = %s '
            sqlqueryvar.append(locationFilter)
        except KeyError:
            pass
        
        dprint(sqlquery)
        dprint(sqlqueryvar)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dprint(sqlquery)
                dprint(sqlqueryvar)
                dbcursor.execute(sqlquery, sqlqueryvar)
                dprint('execed')
                dprint(dbcursor.statement)
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        querySearchLimit = int(getSettingsValue('querySearchLimit'))
                
        retinfo['status'] = 'ok'
        retinfo['users'] = dbcursorfetch[:querySearchLimit]

        return jsonify(retinfo)   
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/getLocationInfo', methods=['POST'])    
def getLocationInfo():
    if ensureLoggedIn(session, 3):
        retinfo = {}
        
        locationfilter = request.json.get('filters')
        
        sqlquery = 'SELECT locationid, name, type FROM locations WHERE 1 = 1 '
        sqlqueryvar = []
        
        try:
            idfilter = int(locationfilter['id'])
            sqlquery += 'AND locationid = %s '
            sqlqueryvar.append(idfilter)
        except KeyError:
            pass
        
        try:
            namefilter = str(locationfilter['name'])
            namekeywords = namefilter.split()
            for keyword in namekeywords:
                sqlquery += 'AND name LIKE %s '
                sqlqueryvar.append(f'%{keyword}%')
        except KeyError:
            pass
        
        try:
            typefilter = str(locationfilter['type'])
            sqlquery += 'AND type = %s '
            sqlqueryvar.append(typefilter)
        except KeyError:
            pass
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute(sqlquery, sqlqueryvar)
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        querySearchLimit = int(getSettingsValue('querySearchLimit'))
                
        retinfo['status'] = 'ok'
        retinfo['locationinfo'] = dbcursorfetch[:querySearchLimit]
        
        return jsonify(retinfo)   
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/updateUserLocation', methods=['POST'])
def updateUserLocation():
    if ensureLoggedIn(session, 1):
        retinfo = {}
        
        oid = getOidFromSession(session)
        
        urin = checkUserInformation("userid", oid)
        if urin == None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Not authorized to perform this action'
            
            return jsonify(retinfo)
        userid = urin[0]
        
        locationName = str(request.json.get('location'))
        
        dprint(type(locationName))
        
        locationid = getLocationIdFromName(locationName)
                                
        if locationid == None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'invalidlocation'
            
            return jsonify(retinfo)
                                
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('UPDATE users SET locationid = %s WHERE userid = %s', (locationid, userid,))
                
        retinfo['status'] = 'ok'
        
        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
        
    
@app.route('/api/getUserInfo', methods=['POST'])
def getUserInfo():
    if ensureLoggedIn(session, 3):
        retinfo = {}
        
        oid = getOidFromSession(session)
        
        urin = checkUserInformation("userid, name, email, locationid, role", oid) 
        if urin == None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Not authorized to perform this action'
            
            return jsonify(retinfo)
        userinfo = ["user"]
        userinfo += urin
        userinfo = [userinfo]
        
        dprint(userinfo)
        
        retinfo["status"] = 'ok'
        retinfo["userinfo"] = listToJson(userinfo)
        
        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/getStudentInfo', methods=['POST'])
def getStudentsInfo():
    if ensureLoggedIn(session, 3, kioskAllowed=True):
        retinfo = {}
        
        studentid = str(request.json.get('studentid'))
        
        studentinfo = getStudentInfoFromId(studentid)

        if studentinfo == 'nostudent':
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'No student maches the studentid provided'

            return jsonify(retinfo)

        retinfo['status'] = 'ok'
        retinfo['studentinfo'] = studentinfo
        
        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
        
    
@app.route('/api/newPass', methods=['POST'])
def newPass():
    if ensureLoggedIn(session, 2, kioskAllowed=True): 
        retinfo = {}
        
        studentid = request.json.get('studentid')
        destinationid = request.json.get('destinationid')

        suspendInfo = isStudentSuspended(studentid)

        if suspendInfo:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = f'Student is suspended: {suspendInfo}'

            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute("SELECT passid, studentid, floorid, destinationid, creationtime, fleavetime, flapprover, darrivetime, daapprover, dleavetime, dlapprover, farrivetime, faapprover, flagged, keywords FROM passes WHERE studentid = %s AND farrivetime IS null", (studentid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
                dprint('pa')
                dprint(dbcursorfetch)
        
        if len(dbcursorfetch) > 0:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'A pass associated with this student is already active'
            retinfo['passid'] = dbcursorfetch[0][0]
            
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute("SELECT * FROM locations WHERE locationid = %s", (destinationid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
        
        if len(dbcursorfetch) < 1:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'nulllocation'
            
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute("SELECT studentid, name, grade, cardid, floorid, disabledlocations, image, oid, email, suspension, suspensionED FROM students WHERE studentid = %s", (studentid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
        
        if len(dbcursorfetch) < 1:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'nullstudent'
            
            return jsonify(retinfo)
        
        try:
            if str(destinationid) in dbcursorfetch[0][5].split():
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'locationdisabled'
            
                return jsonify(retinfo)
            
        except AttributeError:
            pass
        
        oid = getOidFromSession(session)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute("SELECT floorid FROM students WHERE studentid = %s", (studentid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        floorid = dbcursorfetch[0][0]
        
        urin = checkUserInformation("userid, name, oid, email, role, locationid", oid)
        if urin == None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Not authorized to perform this action'
            
            return jsonify(retinfo)
        timestamp = currentDatetime()

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute("SELECT name FROM students WHERE studentid = %s", (studentid,))
                studentname = dbfetchedConvertDate(dbcursor.fetchall())[0][0]

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute("SELECT name FROM locations WHERE locationid = %s", (destinationid,))
                destinationname = dbfetchedConvertDate(dbcursor.fetchall())[0][0]

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute("SELECT name FROM locations WHERE locationid = %s", (floorid,))
                floorname = dbfetchedConvertDate(dbcursor.fetchall())[0][0]

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute("SELECT grade FROM students WHERE studentid = %s", (studentid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
        
        studentGrade = 'Grade ' + str(dbcursorfetch[0][0])

        keywordsList = [studentname, destinationname, floorname, studentGrade]
        keywords = ' '.join(keywordsList)
                
        try:
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('INSERT INTO passes (studentid, floorid, destinationid, creationtime, keywords) VALUES (%s, %s, %s, %s, %s)', (studentid, floorid, destinationid, timestamp, keywords,))
                                        
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('SELECT passid FROM passes WHERE studentid = %s AND fleavetime IS null', (studentid,))
                    dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                    
            passid = str(dbcursorfetch[0][0])
        except:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'sqlerror'
            
            return jsonify(retinfo)
        
        retinfo['status'] = 'ok'
        retinfo['passid'] = passid
        
        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
        
@app.route('/checkOid')
def checkOid():
    oid = str(decrypt(session["oid"]))
    return f"OID: {oid}"

@app.route('/studentInfoDisplay')
def studentInfoDisplay():
    return render_template('studentInfoDisplay.html')

@app.route('/studentInfoFrame')
def studentInfoFrame():
    return render_template('studentInfoFrame.html')

@app.route('/studentDestinationChooser')
def studentDestinationChooser():
    return render_template('studentDestinationChooser.html')

@app.route('/webSerialTest')
def webSerialTest():
    return render_template('webSerialTest.html')

@app.route('/passCatalog')
def passCatalog():
    if ensureLoggedIn(session, 3):
        return render_template('passCatalog.html')
    else:
        return redirect('/?reject=You are not authorized to view this page')
    
@app.route('/studentCatalog')
def studentCatalog():
    if ensureLoggedIn(session, 2):
        return render_template('studentCatalog.html')
    else:
        return redirect('/?reject=You are not authorized to view this page')

@app.route('/api/getUserRole', methods=['POST'])
def getUserRole():
    if ensureLoggedIn(session, 3):
        retinfo = {}

        oid = getOidFromSession(session)
        userid_result = checkUserInformation("userid", oid)
        if userid_result is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Not authorized to perform this action'
            return jsonify(retinfo)
        
        retinfo['status'] = 'ok'

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT role FROM users WHERE userid = %s', (userid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())

        if len(dbcursorfetch) < 1:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'User does not exist'
            return jsonify(retinfo)
        
        userRole = dbcursorfetch[0][0]

        if userRole == 1:
            retinfo['userRole'] = 'admin'
        elif userRole == 2:
            retinfo['userRole'] = 'proctor'
        elif userRole == 3:
            retinfo['userRole'] = 'approver'
        else:
            retinfo['userRole'] = 'user'

        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/getStudentImage', methods=['POST'])
def getStudentImage():
    if ensureLoggedIn(session, 3, kioskAllowed=True):
        retinfo = {}
        
        studentid = request.json.get('studentid')
                        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT image FROM students WHERE studentid = %s', (studentid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(dbcursorfetch) < 1:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'imgerr'
            
            return jsonify(retinfo)
                
        retinfo['status'] = 'ok'
        retinfo['studentBase64Image'] = dbcursorfetch[0][0]
        
        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/getStudentFloorName', methods=['POST'])
def getStudentFloorName():
    if ensureLoggedIn(session, 3, kioskAllowed=True):
        retinfo = {}
        
        studentid = request.json.get('studentid')
                        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('''
                    SELECT locations.name FROM students 
                    JOIN locations ON students.floorid = locations.locationid 
                    WHERE students.studentid = %s
                ''', (studentid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())
                
        if len(dbcursorfetch) < 1:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Floor does not exist'
            
            return jsonify(retinfo)
                
        retinfo['status'] = 'ok'
        retinfo['floorName'] = dbcursorfetch[0][0]
        
        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/getPassInfo', methods=['POST'])
def getPassInfo():
    if ensureLoggedIn(session, 3):
        retinfo = {}
        
        data = request.json
        passid = data.get('passid')
        appendApproverName = True
        if data.get('appendApproverName') == False:
            appendApproverName = False
        
        if not passid:
            return jsonify({'status': 'error', 'errorinfo': 'Pass ID is missing'})

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('''
                    SELECT studentid, floorid, destinationid, flagged, fleavetime, darrivetime, dleavetime, farrivetime, flapprover, daapprover, dlapprover, faapprover, flka, daka, dlka, faka, flpa, dapa, dlpa, fapa FROM passes WHERE passid = %s ''', (passid,))
                passinfo = dbfetchedOneConvertDate(dbcursor.fetchone())

        for i in range(4):
            if passinfo[i + 8] != None:
                passinfo[i + 4] = passinfo[i + 4].strftime("%Y-%m-%d %H:%M:%S")
                if appendApproverName:
                    passinfo[i + 4] += '\n'
                    if passinfo[i + 12]:
                        passinfo[i + 4] += '🄺 '
                    if passinfo[i + 16]:
                        passinfo[i + 4] += '🄿 '
                    passinfo[i + 4] += getUserNameFromId(passinfo[i + 8])
                
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT name, grade, cardid, image FROM students WHERE studentid = %s', (passinfo[0],))
                studentinfo = dbfetchedOneConvertDate(dbcursor.fetchone())
                
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT name FROM locations WHERE locationid = %s', (passinfo[1],))
                floorinfo = dbfetchedOneConvertDate(dbcursor.fetchone())
                
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT name FROM locations WHERE locationid = %s', (passinfo[2],))
                destinationinfo = dbfetchedOneConvertDate(dbcursor.fetchone())
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT flimage, daimage, dlimage, faimage FROM passes WHERE passid = %s', (passid,))
                passimages = dbfetchedConvertDate(dbcursor.fetchall())[0]

        if passinfo == None or studentinfo == None or floorinfo == None or destinationinfo == None or passimages == None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Pass does not exist with correct information'
            
            return jsonify(retinfo)
        
        retinfo['status'] = 'ok'
        retinfo['passinfo'] = passinfo
        retinfo['studentinfo'] = studentinfo
        retinfo['floorinfo'] = floorinfo
        retinfo['destinationinfo'] = destinationinfo
        retinfo['passimages'] = passimages

        return jsonify(retinfo)
    
    else:
        return jsonify({'status': 'error', 'errorinfo': 'Not authorized to perform this action'})
        
@app.route('/managePanel')
def managePanel():
    if ensureLoggedIn(session, 1):
        return render_template('managePanel.html')
    else:
        return redirect('/?reject=You are not authorized to view this page')
    
@app.route('/passwordReset')
def passwordReset():
    return render_template('passwordReset.html', campusBackground = serverConfig["campusInfo"]["background"])
    
@app.route('/settingsPanel')
def settingsPanel():
    if ensureLoggedIn(session, 3):
        return render_template('settingsPanel.html')
    else:
        return redirect('/?reject=You are not authorized to view this page')

@app.route('/api/passwordReset', methods=['POST'])
def requestPasswordReset():
    email = request.json.get('email')
    retinfo = {}
    
    with dbConnect() as connection:
        with connection.cursor() as dbcursor:
            dbcursor.execute('SELECT userid FROM users WHERE email = %s', (email,))
            result = dbfetchedConvertDate(dbcursor.fetchall())
    
    if len(result) < 1:
        delaySeconds = random.randint(45, 65)
        time.sleep(delaySeconds / 10)
        return jsonify({'status': 'ok'})
    
    userid = result[0][0]
    
    with dbConnect() as connection:
        with connection.cursor() as dbcursor:
            dbcursor.execute("SELECT expireTime FROM passwordreset WHERE userid = %s", (userid,))
            result = dbfetchedConvertDate(dbcursor.fetchall())
            
    for singleReset in result:
        timeSinceLastReset = int((currentDatetime() - (singleReset[0] - timedelta(hours=1))).total_seconds())
        if timeSinceLastReset < 60:
            retinfo["status"] = 'error'
            retinfo["errorinfo"] = f'Your requests are too frequent. Please wait for {60 - timeSinceLastReset} seconds then try again.'
            
            return jsonify(retinfo)
    
    token = ''.join(random.choices(string.ascii_uppercase + string.digits, k=32))
    expireTime = currentDatetime() + timedelta(hours=1)
    
    with dbConnect() as connection:
        with connection.cursor() as dbcursor:
            dbcursor.execute('INSERT INTO passwordreset (userid, token, expireTime) VALUES (%s, %s, %s)', (userid, token, expireTime))
    
    serverUrl = getSettingsValue('serverURL')
    reset_link = f"{serverUrl}/resetPassword?token={token}"
    email_body = "<html lang='en'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'><style>body{font-family:Arial,sans-serif;margin:0;padding:20px;background-color:#f0f2f5;min-height:100vh;display:flex;justify-content:center;align-items:center}.container{max-width:400px;width:100%;background-color:white;padding:30px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.1);text-align:center}h1{text-align:center;color:#1877f2;margin-bottom:25px}h2{text-align:center;color:#1877f2;margin-bottom:25px}p{line-height:1.6;color:#444;margin-bottom:25px}.reset-link{display:inline-block;color:#1877f2;text-decoration:none;font-weight:500;padding:10px 15px;border-radius:5px;transition:all 0.2s ease;margin:0 auto}.reset-link:hover{background-color:#f0f2f5;transform:translateY(-1px)}</style></head><body><div class='container'><h1>Exabyte Tech</h1><h2>DPM Password Reset</h2><p>Click the link below to reset your password:</p><a href='resetplaceholder' class='reset-link'>Reset Password</a></div></body></html>"
    email_body = email_body.replace('resetplaceholder', reset_link)
    
    if sendEmail('Password Reset Request', email_body, email):
        delaySeconds = random.randint(0, 20)
        time.sleep(delaySeconds / 10)
        return jsonify({'status': 'ok'})
    else:
        delaySeconds = random.randint(0, 20)
        time.sleep(delaySeconds / 10)
        return jsonify({'status': 'error', 'errorinfo': 'Failed to send email'})
    
@app.route('/api/getSettingsValue', methods=['POST'])
def getSettingsValueAPI():
    if ensureLoggedIn(session, 1):
        retinfo = {}
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT name, value FROM settings WHERE name != "smtpPassword" AND name != "msauthClientSecret"')
                result = dbfetchedConvertDate(dbcursor.fetchall())

        settings = {name: value for name, value in result}

        retinfo['status'] = 'ok'
        retinfo['settings'] = settings

        return jsonify(retinfo)
    
@app.route('/api/setSettingsValue', methods=['POST'])
def setSettingsValueAPI():
    if ensureLoggedIn(session, 1):
        retinfo = {}
        
        settingName = request.json.get('settingName')
        settingValue = request.json.get('settingValue')
        
        if not settingName or not settingValue:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Invalid setting name or value'
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('UPDATE settings SET value = %s WHERE name = %s', (settingValue, settingName))
        
        retinfo['status'] = 'ok'
        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'

        return jsonify(retinfo)

@app.route('/resetPassword')
def resetPassword():
        token = request.args.get('token')
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT userid FROM passwordreset WHERE token = %s AND expireTime > %s', (token, currentDatetime()))
                result = dbfetchedConvertDate(dbcursor.fetchall())

        if len(result) < 1:
            return render_template('expiredPassReset.html')
        
        userid = result[0][0]
                        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT name FROM users WHERE userid = %s', (userid,))
                result = dbfetchedConvertDate(dbcursor.fetchall())
                
        username = result[0][0]
        
        return render_template('newPassword.html', token=token, userName=username)
    
@app.route('/api/resetNewPassword', methods=['POST'])
def resetNewPassword():
    retinfo = {}
    token = request.json.get('token')
    newPassword = request.json.get('newPassword')
    
    with dbConnect() as connection:
        with connection.cursor() as dbcursor:
            dbcursor.execute('SELECT userid FROM passwordreset WHERE token = %s AND expireTime > %s', (token, currentDatetime()))
            result = dbfetchedConvertDate(dbcursor.fetchall())
    
    if len(result) < 1:
        return jsonify({'status': 'error', 'errorinfo': 'Invalid or expired token'})
    
    checkPasswordResult = checkPassword(newPassword)
    if checkPasswordResult[0] == False:
        retinfo["status"] = 'error'
        retinfo['errorinfo'] = checkPasswordResult[1]
        
        return jsonify(retinfo)
    
    userid = result[0][0]
    hashed_password = generateSHA256(newPassword)
    
    with dbConnect() as connection:
        with connection.cursor() as dbcursor:
            dbcursor.execute('UPDATE users SET password = %s WHERE userid = %s', (hashed_password, userid))
            dbcursor.execute('DELETE FROM passwordreset WHERE token = %s', (token,))
    
    return jsonify({'status': 'ok'})

@app.route('/api/generateBackup', methods=['POST'])
def generateBackup():
    if ensureLoggedIn(session, 1):
        retinfo = {}
        
        homeDir = os.path.expanduser("~")
        backupFileName = f"backup_{currentDatetime().strftime('%Y%m%d_%H%M%S')}.sql"

        try:
            os.system(f"mysqldump -u {dbuser} --password={dbpassword} '{dbdatabase}' > '{os.path.join(os.path.join(homeDir, 'DPMBackups'), backupFileName)}'")
            retinfo['status'] = 'ok'
            retinfo['backupFileName'] = backupFileName
        except Exception as e:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = str(e)
        
        return jsonify(retinfo)
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/getBackupList', methods=['POST'])
def getBackupList():
    if ensureLoggedIn(session, 1):
        retinfo = {}
        
        homeDir = os.path.expanduser("~")
        backupDir = os.path.join(homeDir, 'DPMBackups')
        
        try:
            backupFiles = [f for f in os.listdir(backupDir) if f.startswith('backup_') and f.endswith('.sql')]
            retinfo['status'] = 'ok'
            retinfo['backupFiles'] = backupFiles
        except Exception as e:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = str(e)
        
        return jsonify(retinfo)
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/deleteBackup', methods=['POST'])
def deleteBackup():
    if ensureLoggedIn(session, 1):
        retinfo = {}
        
        filename = request.json.get('filename')
        
        if not filename:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Filename is required'
            return jsonify(retinfo)
        
        homeDir = os.path.expanduser("~")
        file_path = os.path.join(os.path.join(homeDir, 'DPMBackups'), filename)
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                retinfo['status'] = 'ok'
            else:
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'File does not exist'
        except Exception as e:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = str(e)
        
        return jsonify(retinfo)
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/loadBackup', methods=['POST'])
def loadBackup():
    if ensureLoggedIn(session, 1):
        retinfo = {}
        
        filename = request.json.get('filename')
        
        if not filename:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Filename is required'
            return jsonify(retinfo)
        
        homeDir = os.path.expanduser("~")
        file_path = os.path.join(os.path.join(homeDir, 'DPMBackups'), filename)
        
        try:
            if os.path.exists(file_path):
                os.system(f"mysql -u {dbuser} --password={dbpassword} '{dbdatabase}' < '{file_path}'")

                with dbConnect() as connection:
                    with connection.cursor() as dbcursor:
                        dbcursor.execute('UPDATE sessions SET active = false')

                broadcastMasterCommand('signout')

                retinfo['status'] = 'ok'
            else:
                retinfo['status'] = 'error'
                retinfo['errorinfo'] = 'File does not exist'
        except Exception as e:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = str(e)
        
        return jsonify(retinfo)
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/uploadBackup', methods=['POST'])
def uploadBackup():
    if ensureLoggedIn(session, 1):
        retinfo = {}
        
        if 'file' not in request.files:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'No file part in the request'
            return jsonify(retinfo)
        
        file = request.files['file']
        
        if file.filename == '':
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'No selected file'
            return jsonify(retinfo)
        
        if not file.filename.lower().endswith('.sql'):
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Only .sql files are allowed'
            return jsonify(retinfo)
        
        homeDir = os.path.expanduser("~")
        backupDir = os.path.join(homeDir, 'DPMBackups')
        
        if not os.path.exists(backupDir):
            os.makedirs(backupDir)
        
        file_path = os.path.join(backupDir, file.filename)
        
        try:
            file.save(file_path)
            retinfo['status'] = 'ok'
            retinfo['filename'] = file.filename
        except Exception as e:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = str(e)
        
        return jsonify(retinfo)
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/getKIOSKPin', methods=['POST'])
def getKIOSKPin():
    if ensureLoggedIn(session, 3):
        retinfo = {}

        userid = checkUserInformation("userid", getOidFromSession(session))[0]

        if userid is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Not authorized to perform this action'
            return jsonify(retinfo)
        
        try:
            pin = KIOSKPin.getKIOSKPin(userid)
            retinfo['status'] = 'ok'
            retinfo['pin'] = pin
        except Exception as e:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = str(e)
        
        return jsonify(retinfo)
    else:
        retinfo = {}

@app.route('/api/generateKIOSKPin', methods=['POST'])
def generateKIOSKPin():
    if ensureLoggedIn(session, 3):
        retinfo = {}

        userid = checkUserInformation("userid", getOidFromSession(session))[0]

        if userid is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Not authorized to perform this action'
            return jsonify(retinfo)
                
        try:
            pin = KIOSKPin.generateNewKIOSKPin(userid)

            retinfo['status'] = 'ok'
            retinfo['pin'] = pin

        except Exception as e:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = str(e)
        
        return jsonify(retinfo)
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401

@app.route('/api/sendMasterCommand', methods=['POST'])
def sendMasterCommand():
    if ensureLoggedIn(session, 1):
        retinfo = {}
        
        command = request.json.get('command')
        payload = request.json.get('payload')

        roles = request.json.get('roles')
        
        if not command:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Command is required'
            return jsonify(retinfo)
        
        try:
            broadcastMasterCommand(command, payload, roles)
            retinfo['status'] = 'ok'
        except Exception as e:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = str(e)
        
        return jsonify(retinfo)
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/student/getStudentName', methods=['GET'])
def getStudentName():
    if ensureLoggedIn(session, studentPortal = True):
        retinfo = {}

        oid = getOidFromSession(session)

        studentid = getStudentIdFromOid(oid)

        if studentid is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Not authorized to perform this action'
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT name FROM students WHERE studentid = %s', (studentid,))
                studentname = dbfetchedOneConvertDate(dbcursor.fetchone())

        if studentname is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Student does not exist'
            return jsonify(retinfo)
        
        retinfo['status'] = 'ok'
        retinfo['name'] = studentname[0]

        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/student/getStudentImage', methods=['GET'])
def studentGetImage():
    if ensureLoggedIn(session, studentPortal = True):
        retinfo = {}

        oid = getOidFromSession(session)

        studentid = getStudentIdFromOid(oid)

        if studentid is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Not authorized to perform this action'
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT image FROM students WHERE studentid = %s', (studentid,))
                studentimage = dbfetchedOneConvertDate(dbcursor.fetchone())

        if studentimage is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Student does not exist'
            return jsonify(retinfo)
        
        retinfo['status'] = 'ok'
        retinfo['image'] = studentimage[0]

        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/student/getStudentPassInfo', methods=['GET'])
def studentGetPassInfo():
    if ensureLoggedIn(session, studentPortal = True):
        retinfo = {}

        oid = getOidFromSession(session)

        studentid = getStudentIdFromOid(oid)

        if studentid is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Not authorized to perform this action'
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT passid, destinationid, creationtime, fleavetime, flagged FROM passes WHERE studentid = %s AND farrivetime IS NULL', (studentid,))
                passinfo = dbfetchedConvertDate(dbcursor.fetchall())

        if len(passinfo) < 1:
            retinfo['status'] = 'ok'
            retinfo['passinfo'] = '<h3>No active pass</h3>'
            retinfo['passstatus'] = False
            return jsonify(retinfo)
        
        passinfo[0][2] = passinfo[0][2].strftime("%Y-%m-%d %H:%M:%S")

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT name FROM locations WHERE locationid = %s', (passinfo[0][1],))
                locationinfo = dbfetchedOneConvertDate(dbcursor.fetchone())

        approvedtext = '<span class="red-text">Not Active</span>'
        retinfo['approved'] = False
        if passinfo[0][3] is not None:
            approvedtext = '<span class="green-text">Active</span>'
            retinfo['approved'] = True

        flaggedtext = '<span class="green-text">Not Flagged</span>'
        if passinfo[0][4]:
            flaggedtext = '<span class="red-text">Flagged</span>'

        if passinfo is None or locationinfo is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'No active pass found or location does not exist'
            return jsonify(retinfo)
        
        passinfotext = f'<h3>Destination: </h3>{locationinfo[0]}<br><h3>Creation Time: </h3>{passinfo[0][2]}<br><h3>Leave Active: </h3><b>{approvedtext}</b><br><h3>Flagged: </h3><b>{flaggedtext}</b>'
        
        retinfo['status'] = 'ok'
        retinfo['passinfo'] = passinfotext
        retinfo['passstatus'] = True

        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/student/newPass', methods=['POST'])
def studentNewPass():
    if ensureLoggedIn(session, studentPortal = True):
        retinfo = {}

        oid = getOidFromSession(session)

        studentid = getStudentIdFromOid(oid)

        suspendInfo = isStudentSuspended(studentid)

        if suspendInfo:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = f'You are suspended: {suspendInfo}'

            return jsonify(retinfo)

        if studentid is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Not authorized to perform this action'
            return jsonify(retinfo)   

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute("SELECT passid FROM passes WHERE studentid = %s AND farrivetime IS NULL", (studentid,))    
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())

        if len(dbcursorfetch) > 0:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'A pass is already active'

            return jsonify(retinfo)
        
        destinationname = request.json.get('destinationname')

        if not destinationname:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Destination name is required'
            return jsonify(retinfo)
        
        destinationid = getLocationIdFromName(destinationname)

        if destinationid == None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Invalid destination name'
            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute("SELECT * FROM locations WHERE locationid = %s", (destinationid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())

        if len(dbcursorfetch) < 1:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'nulllocation'
            
            return jsonify(retinfo)

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute("SELECT studentid, name, grade, cardid, floorid, disabledlocations, image, oid, email, suspension, suspensionED FROM students WHERE studentid = %s", (studentid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())

        if len(dbcursorfetch) < 1:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'nullstudent'
            
            return jsonify(retinfo)

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute("SELECT floorid FROM students WHERE studentid = %s", (studentid,))
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())

        floorid = dbcursorfetch[0][0]

        timestamp = currentDatetime()

        studentName = getStudentNameFromId(studentid)
        if studentName is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Student does not exist'
            return jsonify(retinfo)
        destinationname = getLocationNameFromId(destinationid)
        if destinationname is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Destination does not exist'
            return jsonify(retinfo)
        floorname = getLocationNameFromId(floorid)
        if floorname is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Floor does not exist'
            return jsonify(retinfo)
        studentGrade = getStudentGradeFromId(studentid)
        if studentGrade is None:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Student grade does not exist'
            return jsonify(retinfo)
        studentGade = 'Grade ' + str(studentGrade)

        keywordsList = [studentName, destinationname, floorname, studentGade]
        keywordsList.append(' '.join(keywordsList))
                
        try:
            with dbConnect() as connection:
                with connection.cursor() as dbcursor:
                    dbcursor.execute('INSERT INTO passes (studentid, floorid, destinationid, creationtime, keywords) VALUES (%s, %s, %s, %s, %s)', (studentid, floorid, destinationid, timestamp, keywordsList[4],))
        except Exception as e:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = str(e)
            
            return jsonify(retinfo)

        return jsonify({'status': 'ok', 'message': 'Pass created successfully'})
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401
    
@app.route('/api/getStudentIdFromCard', methods=['POST'])
def getStudentIdFromCard():
    if ensureLoggedIn(session, 3, kioskAllowed=True):
        retinfo = {}
        cardid = request.json.get('cardid')
        if not cardid:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Missing card ID'
            return jsonify(retinfo)
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT studentid FROM students WHERE cardid = %s', (cardid,))
                result = dbfetchedConvertDate(dbcursor.fetchall())
        if not result:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'Student not found'
            return jsonify(retinfo)
        retinfo['status'] = 'ok'
        retinfo['studentid'] = result[0][0]
        return jsonify(retinfo)
    else:
        return jsonify({'status': 'error', 'errorinfo': 'Not authorized to perform this action'})
    
@app.route('/api/student/deletePass', methods = ['POST'])
def studentDeletePass() :
    if ensureLoggedIn(session, studentPortal = True):
        retinfo = {}

        oid = getOidFromSession(session)

        studentid = getStudentIdFromOid(oid)

        suspendInfo = isStudentSuspended(studentid)

        if suspendInfo:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = f'You are suspended: {suspendInfo}'

            return jsonify(retinfo)

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT passid, studentid, floorid, destinationid, creationtime, fleavetime, flapprover, darrivetime, daapprover, dleavetime, dlapprover, farrivetime, faapprover, flagged, keywords FROM passes WHERE fleavetime IS NULL AND studentid = %s', (studentid,))
                deletepass = dbfetchedConvertDate(dbcursor.fetchall())

        if len(deletepass) < 1:
            retinfo['status'] = 'error'
            retinfo['errorinfo'] = 'No passes to delete or cannot delete an active pass'

            return jsonify(retinfo)
        
        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('DELETE FROM passes WHERE fleavetime IS NULL AND studentid = %s', (studentid,))

        retinfo['status'] = 'ok'

        return jsonify(retinfo)
    
@app.route('/api/student/getDestinations', methods =['GET'])
def getDestinations():
    if ensureLoggedIn(session, studentPortal = True):
        retinfo = {}

        with dbConnect() as connection:
            with connection.cursor() as dbcursor:
                dbcursor.execute('SELECT name FROM locations WHERE type = 1')
                dbcursorfetch = dbfetchedConvertDate(dbcursor.fetchall())

        retinfo['status'] = 'ok'
        retinfo['destinations'] = dbcursorfetch

        return jsonify(retinfo)
    
    else:
        retinfo = {}
        
        retinfo['status'] = 'error'
        retinfo['errorinfo'] = 'Not authorized to perform this action'
        
        return jsonify(retinfo), 401

@app.route('/downloadBackup/<filename>')
def downloadBackup(filename):
    if ensureLoggedIn(session, 1):
        homeDir = os.path.expanduser("~")
        file_path = os.path.join(os.path.join(homeDir, 'DPMBackups'), filename)
        
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return render_template('errorPage.html', errorTitle='File Not Found', errorText='The requested backup file does not exist.', errorDesc='Please ensure the file exists and try again.', errorLink = '/managePanel')
    else:
        return redirect('/?reject=You are not authorized to view this page')

@app.route('/newPassEdit')
def newPassEdit():
    if ensureLoggedIn(session, 2):
        return render_template('newPass.html')
    else:
        return redirect('/?reject=You are not authorized to view this page')
    
@app.route('/studentInfo')
def studentInfo():
    if ensureLoggedIn(session, 3):
        return render_template('studentInfo.html')
    else:
        return redirect('/?reject=You are not authorized to view this page')
    
@app.route('/passInfo')
def passInfo():
    if ensureLoggedIn(session, 3):
        return render_template('passInfo.html')
    else:
        return redirect('/?reject=You are not authorized to view this page')

@app.route('/closePage')
def closePage():
    return render_template('closePage.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('errorPage.html', errorTitle = '404 Not Found', errorText = 'The requested URL was not found on the server.', errorDesc = 'If you entered the URL manually please check your spelling and try again.', errorLink = '/'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('errorPage.html', errorTitle = '500 Internal Server Error', errorText = 'The server encountered an internal error and was unable to complete your request.', errorDesc = 'Either the server is overloaded or there is an error in the application.', errorLink = '/'), 500

if __name__ == '__main__':
    socketio.run(app, port=80, host="0.0.0.0", debug=debug)