# ============================================================
# WonderWords AI Learning System — FastAPI Backend
# File: backend/main.py
# ============================================================
#
# REQUIREMENTS (requirements.txt):
#   fastapi==0.111.0
#   uvicorn[standard]==0.30.0
#   asyncpg==0.29.0
#   sqlalchemy[asyncio]==2.0.30
#   alembic==1.13.1
#   pydantic==2.7.1
#   pydantic-settings==2.3.0
#   python-jose[cryptography]==3.3.0
#   passlib[bcrypt]==1.7.4
#   python-multipart==0.0.9
#   openai==1.30.0
#   httpx==0.27.0
#   redis==5.0.4
#   celery==5.4.0
#   boto3==1.34.0          # S3 for audio uploads
#
# ============================================================

from __future__ import annotations
import uuid, os, json, math, time
from datetime import datetime, timedelta, date
from typing import Optional, List, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from pydantic_settings import BaseSettings
import asyncpg
from passlib.context import CryptContext
from jose import JWTError, jwt


# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────
class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost/wonderwords"
    DATABASE_URL_RAW: str = "postgresql://user:pass@localhost/wonderwords"
    SECRET_KEY: str = "change-me-in-production-use-256bit-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7   # 7 days
    OPENAI_API_KEY: str = ""
    REDIS_URL: str = "redis://localhost:6379/0"
    AWS_BUCKET: str = "wonderwords-audio"
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "https://wonderwords.app"]

    class Config:
        env_file = ".env"

settings = Settings()
pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2   = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ─────────────────────────────────────────
# DB POOL
# ─────────────────────────────────────────
db_pool: asyncpg.Pool | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    db_pool = await asyncpg.create_pool(settings.DATABASE_URL_RAW, min_size=5, max_size=20)
    yield
    await db_pool.close()

async def get_db():
    async with db_pool.acquire() as conn:
        yield conn


# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
app = FastAPI(title="WonderWords API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"

class UserOut(BaseModel):
    id: uuid.UUID
    role: str
    display_name: str
    avatar_emoji: str
    email: Optional[str]
    school_id: Optional[uuid.UUID]

class RegisterIn(BaseModel):
    role: Literal["child", "teacher", "parent"]
    display_name: str
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    pin_code: Optional[str] = Field(None, pattern=r"^\d{4,6}$")
    school_code: Optional[str] = None

class WordOut(BaseModel):
    id: int
    word: str
    category_id: int
    difficulty: str
    image_emoji: str
    phonetic: Optional[str]
    translation_zh: Optional[str]
    audio_url: Optional[str]

class CategoryOut(BaseModel):
    id: int
    name_en: str
    name_zh: str
    icon_emoji: str
    color_hex: str

class SessionStartIn(BaseModel):
    activity_type: Literal["pic_to_word", "audio_to_word", "repeat_speaking", "flashcard"]
    category_id: Optional[int] = None
    assignment_id: Optional[uuid.UUID] = None
    word_count: int = Field(default=10, ge=5, le=30)

class SessionStartOut(BaseModel):
    session_id: uuid.UUID
    words: List[WordOut]
    choices_map: dict    # word_id -> list of 3 distractor word_ids

class AnswerIn(BaseModel):
    session_id: uuid.UUID
    word_id: int
    is_correct: bool
    response_ms: Optional[int] = None
    speech_score: Optional[float] = None

class SpeakingSubmitIn(BaseModel):
    session_id: uuid.UUID
    word_id: int
    transcript: str
    confidence: float

class SessionEndOut(BaseModel):
    stars_earned: int
    accuracy_pct: float
    total_correct: int
    total_questions: int
    new_badges: List[dict]
    word_summary: List[dict]

class AssignmentCreateIn(BaseModel):
    title: str
    classroom_id: uuid.UUID
    word_ids: List[int]
    due_date: Optional[datetime] = None
    required_score: int = 70

class AssignmentOut(BaseModel):
    id: uuid.UUID
    title: str
    classroom_id: uuid.UUID
    due_date: Optional[datetime]
    required_score: int
    status: str
    word_count: int
    class_progress: Optional[dict] = None

class ChildReportOut(BaseModel):
    child_id: uuid.UUID
    display_name: str
    total_stars: int
    current_streak: int
    level: int
    weekly_accuracy: float
    words_mastered: int
    session_count_7d: int
    daily_activity: List[dict]
    weak_words: List[dict]
    strong_words: List[dict]
    badges: List[dict]

class ClassReportOut(BaseModel):
    classroom_id: uuid.UUID
    class_name: str
    student_count: int
    avg_accuracy: float
    avg_stars: float
    assignment_completion: float
    students: List[dict]
    top_categories: List[dict]


# ─────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────
def make_token(user_id: str) -> str:
    exp = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": user_id, "exp": exp}, settings.SECRET_KEY, settings.ALGORITHM)

async def current_user(token: str = Depends(oauth2), db=Depends(get_db)):
    exc = HTTPException(401, "Invalid credentials")
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        uid = payload.get("sub")
        if uid is None: raise exc
    except JWTError:
        raise exc
    user = await db.fetchrow("SELECT * FROM users WHERE id=$1 AND is_active=TRUE", uuid.UUID(uid))
    if user is None: raise exc
    return user

def require_role(*roles: str):
    async def inner(user=Depends(current_user)):
        if user["role"] not in roles:
            raise HTTPException(403, "Insufficient permissions")
        return user
    return inner


# ─────────────────────────────────────────
# ── ROUTER: AUTH
# ─────────────────────────────────────────
from fastapi import APIRouter
auth_router = APIRouter(prefix="/api/auth", tags=["auth"])

@auth_router.post("/register", response_model=Token)
async def register(body: RegisterIn, db=Depends(get_db)):
    if body.email:
        exists = await db.fetchval("SELECT 1 FROM users WHERE email=$1", body.email)
        if exists:
            raise HTTPException(400, "Email already registered")

    school_id = None
    if body.school_code:
        school_id = await db.fetchval("SELECT id FROM schools WHERE code=$1", body.school_code)
        if not school_id:
            raise HTTPException(404, "School code not found")

    pw_hash = pwd_ctx.hash(body.password) if body.password else None
    uid = await db.fetchval(
        """INSERT INTO users(role, display_name, email, password_hash, pin_code, school_id)
           VALUES($1,$2,$3,$4,$5,$6) RETURNING id""",
        body.role, body.display_name, body.email, pw_hash, body.pin_code, school_id
    )
    if body.role == "child":
        await db.execute("INSERT INTO child_profiles(user_id) VALUES($1)", uid)

    user = await db.fetchrow("SELECT * FROM users WHERE id=$1", uid)
    return Token(access_token=make_token(str(uid)), user=UserOut(**dict(user)))

@auth_router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    user = await db.fetchrow("SELECT * FROM users WHERE email=$1 AND is_active=TRUE", form.username)
    if not user or not pwd_ctx.verify(form.password, user["password_hash"] or ""):
        raise HTTPException(401, "Invalid email or password")
    await db.execute("UPDATE users SET last_login_at=NOW() WHERE id=$1", user["id"])
    return Token(access_token=make_token(str(user["id"])), user=UserOut(**dict(user)))

@auth_router.post("/pin-login", response_model=Token)
async def pin_login(display_name: str, pin: str, school_code: str, db=Depends(get_db)):
    """Quick PIN login for children — no email needed."""
    row = await db.fetchrow(
        """SELECT u.* FROM users u JOIN schools s ON s.id=u.school_id
           WHERE u.display_name=$1 AND u.pin_code=$2 AND s.code=$3
             AND u.role='child' AND u.is_active=TRUE""",
        display_name, pin, school_code
    )
    if not row:
        raise HTTPException(401, "Wrong name or PIN")
    return Token(access_token=make_token(str(row["id"])), user=UserOut(**dict(row)))

@auth_router.get("/me", response_model=UserOut)
async def me(user=Depends(current_user)):
    return UserOut(**dict(user))


# ─────────────────────────────────────────
# ── ROUTER: WORDS & CATEGORIES
# ─────────────────────────────────────────
word_router = APIRouter(prefix="/api/words", tags=["words"])

@word_router.get("/categories", response_model=List[CategoryOut])
async def get_categories(db=Depends(get_db)):
    rows = await db.fetch("SELECT * FROM categories ORDER BY sort_order")
    return [CategoryOut(**dict(r)) for r in rows]

@word_router.get("/", response_model=List[WordOut])
async def get_words(
    category_id: Optional[int] = None,
    difficulty: Optional[str] = None,
    limit: int = 50,
    db=Depends(get_db)
):
    q = "SELECT * FROM words WHERE is_active=TRUE"
    params = []
    if category_id:
        params.append(category_id); q += f" AND category_id=${len(params)}"
    if difficulty:
        params.append(difficulty); q += f" AND difficulty=${len(params)}"
    q += f" ORDER BY word LIMIT ${len(params)+1}"
    params.append(limit)
    rows = await db.fetch(q, *params)
    return [WordOut(**dict(r)) for r in rows]

@word_router.get("/due-review")
async def due_for_review(limit: int = 20, user=Depends(current_user), db=Depends(get_db)):
    """Spaced-repetition: words due for review today."""
    rows = await db.fetch(
        """SELECT w.*, wm.mastery_level, wm.accuracy
           FROM word_mastery wm JOIN words w ON w.id=wm.word_id
           WHERE wm.child_id=$1 AND wm.next_review_at <= NOW()
           ORDER BY wm.next_review_at LIMIT $2""",
        user["id"], limit
    )
    return [dict(r) for r in rows]


# ─────────────────────────────────────────
# ── ROUTER: GAME SESSIONS
# ─────────────────────────────────────────
game_router = APIRouter(prefix="/api/game", tags=["game"])

def _build_choices_map(words: list) -> dict:
    """For each word, pick 3 distractors from the same pool."""
    ids = [w["id"] for w in words]
    choices = {}
    import random
    for w in words:
        others = [i for i in ids if i != w["id"]]
        choices[str(w["id"])] = random.sample(others, min(3, len(others)))
    return choices

@game_router.post("/session/start", response_model=SessionStartOut)
async def start_session(body: SessionStartIn, user=Depends(current_user), db=Depends(get_db)):
    import random

    # If assignment, use assignment's words
    if body.assignment_id:
        rows = await db.fetch(
            """SELECT w.* FROM words w
               JOIN assignment_words aw ON aw.word_id=w.id
               WHERE aw.assignment_id=$1""", body.assignment_id
        )
        words = [dict(r) for r in rows]
    else:
        # Adaptive: mix due-review words + new words from category
        due = await db.fetch(
            """SELECT w.* FROM word_mastery wm JOIN words w ON w.id=wm.word_id
               WHERE wm.child_id=$1 AND wm.next_review_at<=NOW()
               AND ($2::int IS NULL OR w.category_id=$2) LIMIT $3""",
            user["id"], body.category_id, body.word_count // 2
        )
        extra_needed = body.word_count - len(due)
        due_ids = [r["id"] for r in due]
        fresh = await db.fetch(
            """SELECT * FROM words WHERE is_active=TRUE
               AND ($1::int IS NULL OR category_id=$1)
               AND id != ALL($2::int[])
               ORDER BY RANDOM() LIMIT $3""",
            body.category_id, due_ids or [0], extra_needed
        )
        words = [dict(r) for r in due] + [dict(r) for r in fresh]
        random.shuffle(words)

    if not words:
        raise HTTPException(400, "No words available")

    sess_id = await db.fetchval(
        """INSERT INTO sessions(child_id, assignment_id, activity_type, category_id, total_questions)
           VALUES($1,$2,$3,$4,$5) RETURNING id""",
        user["id"], body.assignment_id, body.activity_type, body.category_id, len(words)
    )
    return SessionStartOut(
        session_id=sess_id,
        words=[WordOut(**w) for w in words],
        choices_map=_build_choices_map(words)
    )

@game_router.post("/answer")
async def submit_answer(body: AnswerIn, user=Depends(current_user), db=Depends(get_db)):
    sess = await db.fetchrow(
        "SELECT * FROM sessions WHERE id=$1 AND child_id=$2 AND ended_at IS NULL",
        body.session_id, user["id"]
    )
    if not sess:
        raise HTTPException(404, "Session not found or already ended")

    await db.execute(
        """INSERT INTO answer_events(session_id, child_id, word_id, activity_type, is_correct, response_ms, speech_score)
           VALUES($1,$2,$3,$4,$5,$6,$7)""",
        body.session_id, user["id"], body.word_id,
        sess["activity_type"], body.is_correct, body.response_ms, body.speech_score
    )
    # increment correct_count in session
    if body.is_correct:
        await db.execute(
            "UPDATE sessions SET correct_count=correct_count+1 WHERE id=$1", body.session_id
        )
    return {"ok": True}

@game_router.post("/speaking/evaluate")
async def evaluate_speaking(body: SpeakingSubmitIn, user=Depends(current_user), db=Depends(get_db)):
    """Compare child's speech transcript to target word and give score."""
    word_row = await db.fetchrow("SELECT word FROM words WHERE id=$1", body.word_id)
    if not word_row:
        raise HTTPException(404, "Word not found")

    target   = word_row["word"].lower().strip()
    spoken   = body.transcript.lower().strip()

    # Levenshtein similarity (simple edit-distance)
    sim = _levenshtein_similarity(target, spoken)
    # Weighted: 70% similarity + 30% ASR confidence
    final_score = round(sim * 70 + body.confidence * 0.30, 1)
    passed = final_score >= 60

    await db.execute(
        """INSERT INTO speaking_attempts(child_id, word_id, session_id, transcript, confidence, similarity_score, passed)
           VALUES($1,$2,$3,$4,$5,$6,$7)""",
        user["id"], body.word_id, body.session_id,
        spoken, body.confidence, sim * 100, passed
    )

    # Encouraging feedback messages
    if final_score >= 90:
        feedback = "🌟 太棒了！發音超完美！"
    elif final_score >= 75:
        feedback = "✨ 說得很好！繼續練習！"
    elif final_score >= 60:
        feedback = "👍 不錯！再試一次會更好！"
    else:
        feedback = "💪 加油！聽聽看再試試！"

    return {"score": final_score, "passed": passed, "feedback": feedback, "similarity": sim * 100}

@game_router.post("/session/{session_id}/end", response_model=SessionEndOut)
async def end_session(session_id: uuid.UUID, user=Depends(current_user), db=Depends(get_db)):
    sess = await db.fetchrow(
        "SELECT * FROM sessions WHERE id=$1 AND child_id=$2 AND ended_at IS NULL",
        session_id, user["id"]
    )
    if not sess:
        raise HTTPException(404, "Session not found")

    total  = sess["total_questions"]
    correct = sess["correct_count"]
    accuracy = round(correct / max(total, 1) * 100, 1)

    # Stars: 10 per correct + streak bonus
    stars = correct * 10
    if accuracy >= 90: stars += 20   # accuracy bonus
    if accuracy >= 100: stars += 30  # perfect bonus

    duration = int((datetime.utcnow() - sess["started_at"]).total_seconds())

    await db.execute(
        """UPDATE sessions SET ended_at=NOW(), stars_earned=$1, duration_secs=$2 WHERE id=$3""",
        stars, duration, session_id
    )

    # Update streak
    await _update_streak(user["id"], db)

    # Check & award badges
    new_badges = await _check_badges(user["id"], db)

    # Word summary for this session
    word_rows = await db.fetch(
        """SELECT w.word, w.image_emoji, ae.is_correct, COUNT(*) as cnt
           FROM answer_events ae JOIN words w ON w.id=ae.word_id
           WHERE ae.session_id=$1 GROUP BY w.word, w.image_emoji, ae.is_correct""",
        session_id
    )
    word_summary = [dict(r) for r in word_rows]

    return SessionEndOut(
        stars_earned=stars,
        accuracy_pct=accuracy,
        total_correct=correct,
        total_questions=total,
        new_badges=new_badges,
        word_summary=word_summary
    )


# ─────────────────────────────────────────
# ── ROUTER: ASSIGNMENTS (Teacher)
# ─────────────────────────────────────────
assign_router = APIRouter(prefix="/api/assignments", tags=["assignments"])

@assign_router.post("/", response_model=AssignmentOut)
async def create_assignment(
    body: AssignmentCreateIn,
    user=Depends(require_role("teacher", "admin")),
    db=Depends(get_db)
):
    aid = await db.fetchval(
        """INSERT INTO assignments(title, teacher_id, classroom_id, due_date, required_score)
           VALUES($1,$2,$3,$4,$5) RETURNING id""",
        body.title, user["id"], body.classroom_id, body.due_date, body.required_score
    )
    for order, wid in enumerate(body.word_ids):
        await db.execute(
            "INSERT INTO assignment_words(assignment_id, word_id, sort_order) VALUES($1,$2,$3)",
            aid, wid, order
        )
    # Create student_assignment rows for every student in class
    students = await db.fetch(
        """SELECT u.id FROM users u JOIN child_profiles cp ON cp.user_id=u.id
           WHERE cp.classroom_id=$1""", body.classroom_id
    )
    for s in students:
        await db.execute(
            "INSERT INTO student_assignments(assignment_id, child_id) VALUES($1,$2) ON CONFLICT DO NOTHING",
            aid, s["id"]
        )
    return AssignmentOut(
        id=aid, title=body.title, classroom_id=body.classroom_id,
        due_date=body.due_date, required_score=body.required_score,
        status="pending", word_count=len(body.word_ids)
    )

@assign_router.get("/my", response_model=List[AssignmentOut])
async def my_assignments(user=Depends(current_user), db=Depends(get_db)):
    """Returns assignments for the current user (child, teacher)."""
    if user["role"] == "child":
        rows = await db.fetch(
            """SELECT a.*, sa.status as sa_status,
                      COUNT(aw.word_id) as word_count
               FROM student_assignments sa
               JOIN assignments a ON a.id=sa.assignment_id
               LEFT JOIN assignment_words aw ON aw.assignment_id=a.id
               WHERE sa.child_id=$1
               GROUP BY a.id, sa.status ORDER BY a.due_date NULLS LAST""",
            user["id"]
        )
    else:
        rows = await db.fetch(
            """SELECT a.*, COUNT(DISTINCT aw.word_id) as word_count
               FROM assignments a
               LEFT JOIN assignment_words aw ON aw.assignment_id=a.id
               WHERE a.teacher_id=$1
               GROUP BY a.id ORDER BY a.created_at DESC""",
            user["id"]
        )
    return [AssignmentOut(**{**dict(r), "status": r.get("sa_status", r["status"])}) for r in rows]

@assign_router.get("/{aid}/progress")
async def assignment_progress(
    aid: uuid.UUID,
    user=Depends(require_role("teacher", "admin")),
    db=Depends(get_db)
):
    rows = await db.fetch(
        """SELECT u.display_name, u.avatar_emoji,
                  sa.status, sa.score, sa.attempts, sa.completed_at
           FROM student_assignments sa JOIN users u ON u.id=sa.child_id
           WHERE sa.assignment_id=$1 ORDER BY u.display_name""",
        aid
    )
    total = len(rows)
    done  = sum(1 for r in rows if r["status"] == "completed")
    return {
        "total": total,
        "completed": done,
        "completion_pct": round(done / max(total, 1) * 100),
        "students": [dict(r) for r in rows]
    }


# ─────────────────────────────────────────
# ── ROUTER: REPORTS
# ─────────────────────────────────────────
report_router = APIRouter(prefix="/api/reports", tags=["reports"])

@report_router.get("/child/{child_id}", response_model=ChildReportOut)
async def child_report(child_id: uuid.UUID, user=Depends(current_user), db=Depends(get_db)):
    # Auth: parent can only view their own children, teachers see classmates
    if user["role"] == "parent":
        linked = await db.fetchval(
            "SELECT 1 FROM parent_children WHERE parent_id=$1 AND child_id=$2",
            user["id"], child_id
        )
        if not linked:
            raise HTTPException(403, "Not your child")
    elif user["role"] == "child" and user["id"] != child_id:
        raise HTTPException(403, "Forbidden")

    child = await db.fetchrow(
        "SELECT u.*, cp.* FROM users u JOIN child_profiles cp ON cp.user_id=u.id WHERE u.id=$1",
        child_id
    )
    if not child:
        raise HTTPException(404, "Child not found")

    # 7-day accuracy
    week_stats = await db.fetchrow(
        """SELECT SUM(correct_count) as c, SUM(total_questions) as t
           FROM sessions WHERE child_id=$1 AND started_at >= NOW()-INTERVAL '7 days'""",
        child_id
    )
    weekly_acc = round((week_stats["c"] or 0) / max(week_stats["t"] or 1, 1) * 100, 1)

    # Session count 7d
    sess_7d = await db.fetchval(
        "SELECT COUNT(*) FROM sessions WHERE child_id=$1 AND started_at>=NOW()-INTERVAL '7 days'",
        child_id
    )

    # Words mastered (mastery_level >= 4)
    mastered = await db.fetchval(
        "SELECT COUNT(*) FROM word_mastery WHERE child_id=$1 AND mastery_level>=4", child_id
    )

    # Daily activity (last 14 days)
    daily = await db.fetch(
        """SELECT DATE(started_at) as d, SUM(stars_earned) as stars,
                  ROUND(SUM(correct_count)::numeric/NULLIF(SUM(total_questions),0)*100,1) as acc
           FROM sessions WHERE child_id=$1 AND started_at>=NOW()-INTERVAL '14 days'
           GROUP BY DATE(started_at) ORDER BY d""",
        child_id
    )

    # Weak words (accuracy < 60, at least 3 attempts)
    weak = await db.fetch(
        """SELECT w.word, w.image_emoji, wm.accuracy, wm.attempts
           FROM word_mastery wm JOIN words w ON w.id=wm.word_id
           WHERE wm.child_id=$1 AND wm.accuracy<60 AND wm.attempts>=3
           ORDER BY wm.accuracy LIMIT 5""", child_id
    )
    # Strong words
    strong = await db.fetch(
        """SELECT w.word, w.image_emoji, wm.accuracy, wm.mastery_level
           FROM word_mastery wm JOIN words w ON w.id=wm.word_id
           WHERE wm.child_id=$1 AND wm.mastery_level>=4
           ORDER BY wm.accuracy DESC LIMIT 5""", child_id
    )
    # Badges
    badges = await db.fetch(
        """SELECT b.icon_emoji, b.name_zh, cb.earned_at
           FROM child_badges cb JOIN badges b ON b.id=cb.badge_id
           WHERE cb.child_id=$1 ORDER BY cb.earned_at DESC""", child_id
    )

    return ChildReportOut(
        child_id=child_id,
        display_name=child["display_name"],
        total_stars=child["total_stars"],
        current_streak=child["current_streak"],
        level=child["level"],
        weekly_accuracy=weekly_acc,
        words_mastered=mastered or 0,
        session_count_7d=sess_7d or 0,
        daily_activity=[dict(d) for d in daily],
        weak_words=[dict(w) for w in weak],
        strong_words=[dict(s) for s in strong],
        badges=[dict(b) for b in badges]
    )

@report_router.get("/class/{classroom_id}", response_model=ClassReportOut)
async def class_report(
    classroom_id: uuid.UUID,
    user=Depends(require_role("teacher", "admin")),
    db=Depends(get_db)
):
    cr = await db.fetchrow("SELECT * FROM classrooms WHERE id=$1", classroom_id)
    if not cr: raise HTTPException(404, "Classroom not found")

    students = await db.fetch(
        """SELECT u.id, u.display_name, u.avatar_emoji,
                  cp.total_stars, cp.current_streak, cp.level,
                  COALESCE(ROUND(AVG(wm.accuracy),1),0) as avg_acc,
                  COUNT(DISTINCT CASE WHEN wm.mastery_level>=4 THEN wm.word_id END) as mastered
           FROM users u
           JOIN child_profiles cp ON cp.user_id=u.id
           LEFT JOIN word_mastery wm ON wm.child_id=u.id
           WHERE cp.classroom_id=$1
           GROUP BY u.id, u.display_name, u.avatar_emoji, cp.total_stars, cp.current_streak, cp.level
           ORDER BY cp.total_stars DESC""",
        classroom_id
    )
    sc = len(students)
    avg_acc   = round(sum(s["avg_acc"] for s in students) / max(sc, 1), 1)
    avg_stars = round(sum(s["total_stars"] for s in students) / max(sc, 1), 1)

    # Assignment completion
    assign_stats = await db.fetchrow(
        """SELECT COUNT(*) as total,
                  SUM(CASE WHEN sa.status='completed' THEN 1 ELSE 0 END) as done
           FROM student_assignments sa
           JOIN assignments a ON a.id=sa.assignment_id
           WHERE a.classroom_id=$1""",
        classroom_id
    )
    completion = round((assign_stats["done"] or 0) / max(assign_stats["total"] or 1, 1) * 100, 1)

    # Top categories
    top_cats = await db.fetch(
        """SELECT c.name_zh, c.icon_emoji,
                  ROUND(AVG(wm.accuracy),1) as avg_acc
           FROM word_mastery wm
           JOIN words w ON w.id=wm.word_id
           JOIN categories c ON c.id=w.category_id
           JOIN child_profiles cp ON cp.user_id=wm.child_id AND cp.classroom_id=$1
           GROUP BY c.id ORDER BY avg_acc DESC LIMIT 5""",
        classroom_id
    )

    return ClassReportOut(
        classroom_id=classroom_id,
        class_name=cr["name"],
        student_count=sc,
        avg_accuracy=avg_acc,
        avg_stars=avg_stars,
        assignment_completion=completion,
        students=[dict(s) for s in students],
        top_categories=[dict(t) for t in top_cats]
    )


# ─────────────────────────────────────────
# ── ROUTER: NOTIFICATIONS
# ─────────────────────────────────────────
notif_router = APIRouter(prefix="/api/notifications", tags=["notifications"])

@notif_router.get("/")
async def get_notifications(user=Depends(current_user), db=Depends(get_db)):
    rows = await db.fetch(
        "SELECT * FROM notifications WHERE user_id=$1 ORDER BY created_at DESC LIMIT 30",
        user["id"]
    )
    return [dict(r) for r in rows]

@notif_router.post("/{nid}/read")
async def mark_read(nid: int, user=Depends(current_user), db=Depends(get_db)):
    await db.execute(
        "UPDATE notifications SET is_read=TRUE WHERE id=$1 AND user_id=$2", nid, user["id"]
    )
    return {"ok": True}


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def _levenshtein_similarity(a: str, b: str) -> float:
    """Returns 0.0-1.0 similarity between two strings."""
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return 1 - dp[n] / max(m, n)

async def _update_streak(child_id: uuid.UUID, db):
    today = date.today()
    cp = await db.fetchrow("SELECT last_study_date, current_streak FROM child_profiles WHERE user_id=$1", child_id)
    if not cp: return
    last = cp["last_study_date"]
    streak = cp["current_streak"]
    if last == today: return
    if last == today - timedelta(days=1):
        streak += 1
    else:
        streak = 1
    await db.execute(
        """UPDATE child_profiles
           SET last_study_date=$1, current_streak=$2,
               longest_streak=GREATEST(longest_streak,$2), updated_at=NOW()
           WHERE user_id=$3""",
        today, streak, child_id
    )

async def _check_badges(child_id: uuid.UUID, db) -> List[dict]:
    cp = await db.fetchrow(
        "SELECT total_stars, current_streak, longest_streak FROM child_profiles WHERE user_id=$1", child_id
    )
    total_correct = await db.fetchval("SELECT SUM(correct_count) FROM sessions WHERE child_id=$1", child_id) or 0
    existing = set(await db.fetchval(
        "SELECT ARRAY_AGG(badge_id) FROM child_badges WHERE child_id=$1", child_id
    ) or [])
    all_badges = await db.fetch("SELECT * FROM badges")
    new_earned = []
    for b in all_badges:
        if b["id"] in existing: continue
        earned = False
        if b["badge_type"] == "streak" and (cp["current_streak"] or 0) >= b["threshold"]:
            earned = True
        elif b["badge_type"] == "volume" and total_correct >= b["threshold"]:
            earned = True
        if earned:
            await db.execute(
                "INSERT INTO child_badges(child_id, badge_id) VALUES($1,$2) ON CONFLICT DO NOTHING",
                child_id, b["id"]
            )
            new_earned.append({"icon": b["icon_emoji"], "name": b["name_zh"]})
    return new_earned


# ─────────────────────────────────────────
# MOUNT ROUTERS
# ─────────────────────────────────────────
app.include_router(auth_router)
app.include_router(word_router)
app.include_router(game_router)
app.include_router(assign_router)
app.include_router(report_router)
app.include_router(notif_router)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
