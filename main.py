import asyncio
import json
import time
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sse_starlette.sse import EventSourceResponse
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel

# =======================
# 1. 配置与数据库设置
# =======================
SECRET_KEY = "YOUR_SECRET_KEY_CHANGE_THIS"  # ⚠️ 请修改为随机字符串
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # Token 有效期 1 天

SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# =======================
# 2. 数据库模型 (Models)
# =======================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_admin = Column(Boolean, default=False)
    # 积分系统
    points = Column(Float, default=0.0)          # 通用积分
    student_points = Column(Float, default=0.0)  # 学生积分
    veteran_points = Column(Float, default=0.0)  # 老兵积分
    last_checkin = Column(String, default="")    # 上次签到日期 (YYYY-MM-DD)

class RedeemCode(Base):
    __tablename__ = "redeem_codes"
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, index=True)
    value = Column(Float, default=1.0)
    type = Column(String, default="general") # general, student, veteran
    is_used = Column(Boolean, default=False)

Base.metadata.create_all(bind=engine)

# =======================
# 3. Pydantic Schemas (数据验证)
# =======================
class UserRegister(BaseModel):
    username: str
    password: str
    email: str
    email_code: Optional[str] = None
    invite_code: Optional[str] = None
    cf_token: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str
    cf_token: Optional[str] = None

class VerifyRequest(BaseModel):
    verificationIds: List[str]
    cf_token: Optional[str] = None

class VeteranVerifyRequest(BaseModel):
    input: str # Token 或 链接
    cf_token: Optional[str] = None
    lang: Optional[str] = "zh"

class RedeemRequest(BaseModel):
    code: str

# =======================
# 4. 辅助工具 (Utils)
# =======================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise HTTPException(status_code=401)
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    user = db.query(User).filter(User.username == username).first()
    if user is None: raise HTTPException(status_code=401)
    return user

# =======================
# 5. 核心业务逻辑 (模拟 GitHub 工具调用)
# =======================
# ⚠️ 这里是连接 GitHub 项目 "one-verify-tool" 的关键位置
async def run_sheerid_engine(verification_id: str, mode: str = "student"):
    """
    模拟调用外部 Python 脚本的过程。
    实际部署时，你需要在这里 import 你的 GitHub 项目代码并调用。
    """
    # 阶段 1: 处理中
    yield {
        "verificationId": verification_id,
        "currentStep": "processing",
        "message": "正在启动自动化引擎 (Loading Engine)..."
    }
    await asyncio.sleep(1) # 模拟耗时

    yield {
        "verificationId": verification_id,
        "currentStep": "processing",
        "message": "正在提交基础信息 (Submitting Info)..."
    }
    await asyncio.sleep(1.5)

    # 阶段 2: 模拟 SSO 跳过 (这是 GitHub 项目的核心价值)
    yield {
        "verificationId": verification_id,
        "currentStep": "processing",
        "message": "正在绕过 SSO 登录 (Bypassing SSO)..."
    }
    await asyncio.sleep(2)

    # 阶段 3: 随机返回成功或失败 (模拟)
    import random
    success = random.choice([True, True, False]) # 2/3 概率成功

    if success:
        # 成功返回
        yield {
            "verificationId": verification_id,
            "currentStep": "success",
            "message": "认证成功 (Success)",
            "result": "验证通过！\nToken: xxxxx-mock-token-xxxxx\n(请点击下方按钮订阅)"
        }
    else:
        # 失败返回
        yield {
            "verificationId": verification_id,
            "currentStep": "error",
            "message": "认证失败: IP被拒绝 (IP Rejected)",
            "isRefunded": True # 告诉前端已退款
        }

# =======================
# 6. API 路由实现
# =======================
app = FastAPI(title="One.IDkey Backend")

# 跨域配置 (允许前端访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config API ---
@app.get("/api/config/public")
def get_public_config():
    return {
        "maintenance_mode": False,
        "announcement": "欢迎使用 One.IDkey！后端已连接成功。注册送积分测试中。",
        "announcement_en": "Welcome to One.IDkey! Backend connected.",
        "enable_buy_link": True,
        "buy_link": "https://your-shop-link.com",
        "enable_redeem_code": True,
        "captcha_required_login": False, # 为了测试方便先关掉验证码
        "captcha_required_verify": False,
        "free_mode": False,
        "student_free_mode": False,
        "veteran_free_mode": False,
        "res_enabled": True,
        "resources": [
            {"text": "加入 TG 频道", "url": "https://t.me/your_channel", "icon": "fa-paper-plane"}
        ]
    }

@app.get("/api/config/veteran")
def get_veteran_config():
    return {"veteran_enabled": True, "veteran_points_cost": 1.0}

# --- Auth API ---
@app.post("/api/auth/login")
def login(data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()
    if not user or not pwd_context.verify(data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    
    access_token = create_access_token(data={"sub": user.username})
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "username": user.username,
        "points": user.points,
        "student_points": user.student_points,
        "veteran_points": user.veteran_points,
        "is_admin": user.is_admin
    }

@app.post("/api/auth/register")
def register(data: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    hashed_pw = pwd_context.hash(data.password)
    # 注册赠送初始积分
    new_user = User(
        username=data.username, 
        email=data.email, 
        hashed_password=hashed_pw,
        points=1.0, 
        student_points=1.0
    )
    db.add(new_user)
    db.commit()
    return {"message": "注册成功，请登录"}

@app.get("/api/me")
def get_me(user: User = Depends(get_current_user)):
    return {
        "username": user.username,
        "points": user.points,
        "student_points": user.student_points,
        "veteran_points": user.veteran_points,
        "is_admin": user.is_admin
    }

# --- 核心验证 API (SSE) ---
@app.post("/api/verify")
async def verify_student(
    request: Request, 
    # db: Session = Depends(get_db) # 可以在这里加入 DB 依赖用于扣费
):
    """
    学生认证接口 (SSE 流式)
    """
    body = await request.json()
    ids = body.get("verificationIds", [])
    
    # 获取当前用户 (手动从 Header 获取 Token 以支持 SSE)
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        # 实际应返回 401，但在 SSE 中通常通过 data 消息通知
        return EventSourceResponse(iter([]))

    async def event_generator():
        # 1. 模拟扣费事件 (前端通过这个事件更新 UI 积分)
        yield {
            "event": "deducted", # 自定义事件名，前端未显式监听但会解析 message
            "data": json.dumps({
                "type": "deducted",
                "amount": len(ids),
                "deductedFromName": "学生",
                "allPoints": {"points": 99, "student_points": 99, "veteran_points": 99} # 模拟剩余积分
            })
        }

        # 2. 并行或串行处理每个 ID
        for vid in ids:
            # 调用上面定义的模拟引擎
            async for step_data in run_sheerid_engine(vid, "student"):
                yield {
                    "data": json.dumps(step_data)
                }

    return EventSourceResponse(event_generator())

@app.post("/api/veteran/verify")
async def verify_veteran(request: Request):
    """
    老兵认证接口 (SSE 流式)
    """
    body = await request.json()
    token_input = body.get("input", "")
    
    async def event_generator():
        # 模拟老兵认证流程
        yield {"data": json.dumps({"type": "deducted", "amount": 1, "deductedFromName": "老兵"})}
        
        # 老兵通常只有一个任务
        vid = "token_verify"
        async for step_data in run_sheerid_engine(vid, "veteran"):
            yield {"data": json.dumps(step_data)}

    return EventSourceResponse(event_generator())

# --- 其他辅助 API ---
@app.post("/api/user/checkin")
def checkin(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    today = datetime.now().strftime("%Y-%m-%d")
    if user.last_checkin == today:
        return {"status": "fail", "message": "今天已签到"}
    
    user.last_checkin = today
    user.student_points += 0.5 # 签到送 0.5
    db.commit()
    return {
        "status": "success", 
        "added": 0.5, 
        "points": user.points, 
        "student_points": user.student_points, 
        "veteran_points": user.veteran_points
    }

@app.post("/api/user/redeem")
def redeem(data: RedeemRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    code = db.query(RedeemCode).filter(RedeemCode.code == data.code, RedeemCode.is_used == False).first()
    if not code:
        return {"detail": "无效卡密"}
    
    code.is_used = True
    if code.type == "student": user.student_points += code.value
    elif code.type == "veteran": user.veteran_points += code.value
    else: user.points += code.value
    
    db.commit()
    return {
        "point_type": code.type,
        "added": code.value,
        "new_points": user.points,
        "student_points": user.student_points,
        "veteran_points": user.veteran_points
    }

if __name__ == "__main__":
    import uvicorn
    # 生成一些测试卡密
    db = SessionLocal()
    if not db.query(RedeemCode).first():
        db.add(RedeemCode(code="IDKEY-TEST-8888", value=10.0, type="student"))
        db.commit()
        print("✅ 生成测试卡密: IDKEY-TEST-8888")
    db.close()
    
    print("🚀 后端服务启动中: http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
