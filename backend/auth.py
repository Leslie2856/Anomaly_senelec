from passlib.context import CryptContext
from sqlalchemy.orm import Session
from backend.database import SessionLocal, UserDB
from pydantic import BaseModel
from fastapi import Request, HTTPException
from fastapi.responses import RedirectResponse

class User(BaseModel):
    username: str
    role: str

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

def create_default_superadmin():
    db = SessionLocal()
    try:
        if not db.query(UserDB).first():
            super_admin = UserDB(
                username="superadmin",
                password_hash=hash_password("superadmin123"),
                role="super_admin",
                created_by="system"
            )
            db.add(super_admin)
            db.commit()
            print(">>> Super admin créé : superadmin / superadmin123")
    finally:
        db.close()

def require_auth(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user

create_default_superadmin()

def authenticate_user(username: str, password: str):
    db = SessionLocal()
    try:
        user = db.query(UserDB).filter(UserDB.username == username).first()
        if not user or not verify_password(password, user.password_hash):
            return None
        return user
    finally:
        db.close()

def create_user(db: Session, username: str, password: str, role: str, created_by: str):
    if db.query(UserDB).filter(UserDB.username == username).first():
        raise ValueError("Nom d'utilisateur déjà existant")
    new_user = UserDB(
        username=username,
        password_hash=hash_password(password),
        role=role,
        created_by=created_by
    )
    db.add(new_user)
    db.commit()

def delete_user(db: Session, username: str, current_user):
    user_to_delete = db.query(UserDB).filter(UserDB.username == username).first()
    if not user_to_delete:
        raise ValueError("Utilisateur introuvable")
    if current_user.role == "admin":
        if user_to_delete.role != "simple" or user_to_delete.created_by != current_user.username:
            raise PermissionError("Vous n'avez pas l'autorisation de supprimer cet utilisateur")
    db.delete(user_to_delete)
    db.commit()

def list_visible_users(db: Session, current_user):
    if current_user.role == "super_admin":
        return db.query(UserDB).all()
    elif current_user.role == "admin":
        return db.query(UserDB).filter(
            (UserDB.username == current_user.username) | 
            ((UserDB.role == "simple") & (UserDB.created_by == current_user.username))
        ).all()
    else:
        return db.query(UserDB).filter(UserDB.username == current_user.username).all()

def get_current_user(request: Request):
    user_data = request.session.get("user")
    session_id = request.cookies.get("session_id")
    print(f"Session user_data: {user_data}, session_id: {session_id}")  # Log pour débogage
    if not user_data:
        return None
    return type("User", (), user_data)

def change_password(db: Session, username: str, old_password: str, new_password: str):
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if not user:
        raise ValueError("Utilisateur introuvable")
    if not verify_password(old_password, user.password_hash):
        raise ValueError("Ancien mot de passe incorrect")
    user.password_hash = hash_password(new_password)
    db.commit()
    db.refresh(user)

def reset_password(db: Session, username: str, new_password: str, current_user: User):
    user_to_reset = db.query(UserDB).filter(UserDB.username == username).first()
    if not user_to_reset:
        raise ValueError("Utilisateur introuvable")

    if current_user.role == "super_admin":
        pass
    elif current_user.role == "admin":
        if user_to_reset.role != "simple" or user_to_reset.created_by != current_user.username:
            raise PermissionError("Vous n'avez pas l'autorisation de réinitialiser ce mot de passe")
    else:
        raise PermissionError("Vous n'avez pas l'autorisation de réinitialiser ce mot de passe")

    user_to_reset.password_hash = hash_password(new_password)
    db.commit()
    db.refresh(user_to_reset)