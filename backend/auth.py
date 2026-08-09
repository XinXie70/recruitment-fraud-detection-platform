import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from backend.config import settings
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from jwt import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.rate_limit import limiter


SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes
JWT_ISSUER = settings.jwt_issuer
JWT_AUDIENCE = settings.jwt_audience

router = APIRouter(prefix="/api/auth", tags=["auth"])
password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
logger = logging.getLogger("fake_job_detection_api.auth")


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    username: str = Field(..., min_length=3, max_length=80)
    password: str = Field(..., min_length=8, max_length=30)

    @field_validator("email")
    @classmethod
    def email_must_be_valid(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if "@" not in cleaned or "." not in cleaned.rsplit("@", 1)[-1]:
            raise ValueError("Please enter a valid email address.")
        return cleaned

    @field_validator("username")
    @classmethod
    def username_must_be_simple(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username can only contain letters, numbers, hyphens, and underscores.")
        return cleaned

    @field_validator("password")
    @classmethod
    def password_must_meet_policy(cls, value: str) -> str:
        if not any("A" <= char <= "Z" for char in value):
            raise ValueError("Password must contain an uppercase letter.")
        if not any("a" <= char <= "z" for char in value):
            raise ValueError("Password must contain a lowercase letter.")
        if not any("0" <= char <= "9" for char in value):
            raise ValueError("Password must contain a number.")
        # bcrypt only processes 72 bytes. Checking encoded length prevents
        # multi-byte passwords from reaching the hasher and raising a 500.
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must not exceed 72 UTF-8 bytes.")
        return value


class LoginRequest(BaseModel):
    identifier: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=1, max_length=30)


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    is_admin: bool


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


def hash_password(password: str) -> str:
    return password_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return password_context.verify(password, password_hash)


def authenticate_user(identifier: str, password: str, db: Session) -> User:
    cleaned_identifier = identifier.strip()
    user = (
        db.query(User)
        .filter(
            or_(
                User.email == cleaned_identifier.lower(),
                User.username == cleaned_identifier,
            )
        )
        .first()
    )
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password.",
        )
    return user


def create_access_token(user: User) -> str:
    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user.id),
        "iat": issued_at,
        "exp": expires_at,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "jti": str(uuid4()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def to_user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        is_admin=user.is_admin,
    )


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_error
        user_id = int(user_id)
    except (InvalidTokenError, TypeError, ValueError) as exc:
        raise credentials_error from exc

    user = db.get(User, user_id)
    if user is None:
        raise credentials_error
    return user


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(settings.rate_limit_auth_register)
def register(request: Request, payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = (
        db.query(User)
        .filter(or_(User.email == payload.email, User.username == payload.username))
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email or username is already registered.",
        )

    try:
        user = User(
            email=payload.email,
            username=payload.username,
            password_hash=hash_password(payload.password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return TokenResponse(access_token=create_access_token(user), user=to_user_response(user))
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email or username is already registered.",
        ) from exc
    except Exception as exc:
        db.rollback()
        logger.exception("Registration failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again later.",
        ) from exc


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.rate_limit_auth_login)
def login(request: Request, payload: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(payload.identifier, payload.password, db)
    return TokenResponse(access_token=create_access_token(user), user=to_user_response(user))


@router.post("/token", response_model=TokenResponse)
@limiter.limit(settings.rate_limit_auth_login)
def token(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = authenticate_user(form.username, form.password, db)
    return TokenResponse(access_token=create_access_token(user), user=to_user_response(user))


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    return to_user_response(current_user)
