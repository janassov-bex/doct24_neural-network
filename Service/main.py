import fastapi
from fastapi import FastAPI, Request, Body, Form, templating, Depends
from fastapi.responses import FileResponse
import uvicorn
from typing import Union, Dict, List
from pydantic import BaseModel, EmailStr, Field

import jwt
from decouple import config
from Entities.users_auth import UserSchema, UserLoginSchema
from Models.auth import signJWT, decodeJWT, check_user
from Models.models_ai import ModelsAI
from Models.auth_bearer import JWTBearer
import starlette.status as status

import os
import binascii
import time

from Logger.logger_service import LoggerMethods
logger = LoggerMethods('./logger.log')




class ModelPredict(BaseModel):
    name_model: str
    params: Union[str, None] = None
    data: str
    class Config:
        schema_extra = {
            "example": {
                "name_model": "Название модели.",
                "params": "Параметры модели в json",
                "data": "Данные для предсказания",
            }
        }

app = FastAPI()

users = []
models = ModelsAI()

@app.post("/user/signup", tags=["user"])
async def create_user(user: UserSchema = Body(...)):
    users.append(user) # replace with db call, making sure to hash the password first
    return signJWT(user.email)


@app.post("/user/login", tags=["user"])
async def user_login(user: UserLoginSchema = Body(...)):
    if check_user(users, user):
        return signJWT(user.email)
    return {
        "error": "Wrong login details!"
    }


@app.post("/predict", dependencies=[Depends(JWTBearer())], tags=["posts"])
async def predict_post(post: ModelPredict):
    try:
        return "test"
        #return models.predict_model(post.name_model, post.data, post.params)
    except:
        return {"error": "model not found"}


@app.post("/names", dependencies=[Depends(JWTBearer())], tags=["posts"])
async def predict_post() -> List[str]:
    return models.get_model_names()


@app.get("/", tags=["root"])
async def root(docs: str = None):
    if docs == None:
        return fastapi.responses.RedirectResponse(
            'http://localhost:1024/docs',
            status_code=status.HTTP_302_FOUND)
    else:
        return {"message": "Hello World"}
    #logger.write('Пользователь зашел на сайт')




if __name__ == '__main__':
    app.debug = True
    uvicorn.run(app, host='0.0.0.0', port=1024)

