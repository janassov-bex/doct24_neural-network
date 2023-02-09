import jwt
from typing import Union, Dict
import time
from decouple import config
from Entities.users_auth import UserSchema, UserLoginSchema

JWT_SECRET = config("secret")
JWT_ALGORITHM = config("algorithm")

def token_response(token: str):
    return {
        "access_token": token
    }

def check_user(users, data: UserLoginSchema):
    for user in users:
        if user.email == data.email and user.password == data.password:
            return True
    return False

def signJWT(user_id: str) -> Dict[str, str]:
    payload = {
        "user_id": user_id,
        "expires": time.time() + 600
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return token_response(token)

def decodeJWT(token: str) -> dict:
    try:
        decoded_token = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return decoded_token if decoded_token["expires"] >= time.time() else None
    except:
        return {}