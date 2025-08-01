from fastapi import FastAPI, HTTPException, status, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo import MongoClient
from bson import ObjectId
import os

# --- ENVIRONMENT VARIABLES ---
MONGODB_URL = os.environ.get("MONGODB_URL")
MONGODB_DB = os.environ.get("MONGODB_DB")
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change_me_secret")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

# --- DATABASE SETUP ---
mongo_client = MongoClient(MONGODB_URL)
db = mongo_client[MONGODB_DB]

# --- PASSWORD CONTEXT ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# --- HELPER CLASSES ---
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid objectid')
        return ObjectId(v)

# --- Pydantic Schemas --- #

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    display_name: Optional[str] = Field(None, description="Display name for the user")

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserPublic(BaseModel):
    id: PyObjectId = Field(alias="_id")
    email: EmailStr
    display_name: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None

# For applications
class ApplicationModel(BaseModel):
    id: PyObjectId = Field(alias="_id")
    name: str
    description: Optional[str]
    launch_url: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# For conversational assistant
class MessageIn(BaseModel):
    text: str = Field(..., description="User's input message")
    app_id: Optional[str] = Field(None, description="If part of an application/session conversation")
    meta: Optional[dict] = Field(None, description="Optional extra context (e.g. user agent)")

class MessageOut(BaseModel):
    sender: str  # 'assistant' or 'user'
    text: str
    timestamp: datetime

class ConversationHistoryOut(BaseModel):
    id: PyObjectId = Field(alias="_id")
    user_id: str
    app_id: Optional[str]
    messages: List[MessageOut]

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# --- Utility Functions --- #
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def get_user_from_token(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("user_id")
        email: str = payload.get("email")
        if user_id is None or email is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id, email=email)
    except JWTError:
        raise credentials_exception

    user = db.users.find_one({"_id": ObjectId(token_data.user_id)})
    if not user:
        raise credentials_exception
    return user

# --- FastAPI App Initialization --- #
app = FastAPI(
    title="Unified Assistant Chatbot Platform API",
    description="Backend API for conversational assistant, authentication, app management.",
    version="1.0.0",
    openapi_tags=[
        {"name": "auth", "description": "User Authentication"},
        {"name": "assistant", "description": "Conversational Assistant"},
        {"name": "apps", "description": "Applications Listing & Launch"},
        {"name": "history", "description": "Conversation History"},
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should be restricted in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"message": "Healthy"}

# ----------- AUTHENTICATION ROUTES -----------
# PUBLIC_INTERFACE
@app.post("/auth/register", response_model=UserPublic, tags=["auth"])
def register(user: UserRegister = Body(...)):
    """
    Register a new user.
    - email: email address
    - password: password
    """
    if db.users.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    user_doc = {
        "email": user.email,
        "hashed_password": hash_password(user.password),
        "display_name": user.display_name,
        "created_at": datetime.utcnow()
    }
    result = db.users.insert_one(user_doc)
    user_doc["_id"] = result.inserted_id
    return UserPublic(**user_doc)

# PUBLIC_INTERFACE
@app.post("/auth/login", response_model=Token, tags=["auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Log in with email and password to receive a JWT access token.
    - email: user's email (use 'username' field)
    - password: user's password
    """
    user = db.users.find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(
        data={"user_id": str(user["_id"]), "email": user["email"]}
    )
    return Token(access_token=token, token_type="bearer")

# PUBLIC_INTERFACE
@app.get("/auth/me", response_model=UserPublic, tags=["auth"])
def get_me(current_user: dict = Depends(get_user_from_token)):
    """
    Get details of the current logged-in user.
    """
    # no password in output!
    user = {**current_user}
    user.pop("hashed_password", None)
    return UserPublic(**user)

# ----------- APPLICATION LISTING & LAUNCH -----------
# PUBLIC_INTERFACE
@app.get("/apps", response_model=List[ApplicationModel], tags=["apps"])
def list_applications(current_user: dict = Depends(get_user_from_token)):
    """
    List all available applications for the platform.
    """
    return [ApplicationModel(**a) for a in db.applications.find()]

# PUBLIC_INTERFACE
@app.post("/apps/launch/{app_id}", response_model=ApplicationModel, tags=["apps"])
def launch_application(app_id: str, current_user: dict = Depends(get_user_from_token)):
    """
    Launch an application (returns app metadata). (Logic for launch is minimal in this implementation).
    """
    app_doc = db.applications.find_one({"_id": ObjectId(app_id)})
    if not app_doc:
        raise HTTPException(status_code=404, detail="Application not found")
    return ApplicationModel(**app_doc)

# ----------- CONVERSATIONAL ASSISTANT -----------
def assistant_response(user_input: str, context: Optional[List[dict]] = None) -> str:
    """
    Dummy assistant logic. In production replace with LLM or more complex conversation logic.
    """
    if user_input.strip().lower() in ["hi", "hello"]:
        return "Hello! How can I help you today?"
    elif user_input.strip().endswith("?"):
        return "That's an interesting question! I am here to help. (Demo response)"
    return f"You said: {user_input}"

# PUBLIC_INTERFACE
@app.post("/assistant/message", response_model=MessageOut, tags=["assistant"])
def send_message(message: MessageIn, current_user: dict = Depends(get_user_from_token)):
    """
    Send a message to the conversational assistant. Returns assistant's response.
    Saves interaction to conversation history.
    """
    # Determine conversation
    app_id = message.app_id
    user_id = str(current_user["_id"])
    timestamp = datetime.utcnow()

    # Conversation identification logic (one per app per user per day for demo simplicity)
    conv_filter = {
        "user_id": user_id,
        "app_id": app_id,
        "date": timestamp.date().isoformat(),
    }
    conv_doc = db.conversations.find_one(conv_filter)
    if not conv_doc:
        conv_doc = {
            "user_id": user_id,
            "app_id": app_id,
            "date": timestamp.date().isoformat(),
            "messages": [],
        }
        result = db.conversations.insert_one(conv_doc)
        conv_doc["_id"] = result.inserted_id
    
    # Save user's message
    user_msg = {
        "sender": "user",
        "text": message.text,
        "timestamp": timestamp,
    }
    db.conversations.update_one(
        {"_id": conv_doc["_id"]}, {"$push": {"messages": user_msg}}
    )

    # Assistant response (call LLM/service here in real system)
    context_msgs = db.conversations.find_one({"_id": conv_doc["_id"]})["messages"]
    assistant_text = assistant_response(message.text, context_msgs)
    assistant_msg = {
        "sender": "assistant",
        "text": assistant_text,
        "timestamp": datetime.utcnow(),
    }
    db.conversations.update_one(
        {"_id": conv_doc["_id"]}, {"$push": {"messages": assistant_msg}}
    )

    return MessageOut(**assistant_msg)

# ----------- CONVERSATION HISTORY -----------
# PUBLIC_INTERFACE
@app.get("/history", response_model=List[ConversationHistoryOut], tags=["history"])
def get_conversation_history(app_id: Optional[str] = None, current_user: dict = Depends(get_user_from_token)):
    """
    Get conversation history for current user (optionally only for one app).
    """
    user_id = str(current_user["_id"])
    query = {"user_id": user_id}
    if app_id:
        query["app_id"] = app_id
    cursor = db.conversations.find(query)
    return [
        ConversationHistoryOut(
            _id=conv["_id"],
            user_id=conv["user_id"],
            app_id=conv.get("app_id"),
            messages=[
                MessageOut(
                    sender=msg["sender"],
                    text=msg["text"],
                    timestamp=msg["timestamp"]
                ) for msg in conv.get("messages", [])
            ]
        )
        for conv in cursor
    ]
