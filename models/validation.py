"""
Validation functions for user input
"""
import hashlib
import re


def generateSHA256(text):
    """Generate SHA256 hash of text"""
    if text == None:
        return None
    
    return str(hashlib.sha256(text.encode()).hexdigest())


def checkNameLength(name, min_length, max_length):
    """Check if name is within valid length range"""
    if len(name) < min_length or len(name) > max_length:
        return False
    return True


def checkGrade(grade, min_grade, max_grade):
    """Check if grade is within valid range"""
    try:
        grade_val = int(grade)
        if grade_val < min_grade or grade_val > max_grade:
            return False
        return True
    except (ValueError, TypeError):
        return False


def checkCardidLength(cardid, min_length, max_length):
    """Check if card ID is within valid length range"""
    if len(cardid) < min_length or len(cardid) > max_length:
        return False
    return True


def checkEmailLength(email, min_length, max_length):
    """Check if email is within valid length range"""
    if len(email) < min_length or len(email) > max_length:
        return False
    return True


def checkPassword(password):
    """Validate password against security requirements"""
    error_message = ""
    
    if len(password) < 8:
        error_message += "Password is too short (minimum 8 characters)\n"
    elif len(password) > 20:
        error_message += "Password is too long (maximum 20 characters)\n"
    
    if not password.isalnum():
        error_message += "Password contains special characters (only letters and numbers allowed)\n"
    
    if not any(char.isupper() for char in password):
        error_message += "Password must contain at least one uppercase letter\n"
    
    if not any(char.islower() for char in password):
        error_message += "Password must contain at least one lowercase letter\n"
    
    if not any(char.isdigit() for char in password):
        error_message += "Password must contain at least one number\n"
    
    if error_message:
        return False, error_message.strip()
    return True, "Password is valid"


def validate_json_payload(payload):
    """Validate JSON payload for SQL injection patterns"""
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
