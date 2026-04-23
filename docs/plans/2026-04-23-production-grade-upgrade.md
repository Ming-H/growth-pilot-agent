# Production-Grade AI Agent Upgrade Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 growth-pilot-agent 从货运特定的演示项目升级为通用的、可 SaaS 化部署的生产级用户增长 AI Agent 平台。

**Architecture:** Chief Agent (ReAct + Tool Calling) 编排 5 个 Expert Agent (Agent-as-Tool)，Pydantic 结构化通信，PostgreSQL 持久化，Redis 缓存，FastAPI + SSE 流式 API，JWT 认证。

**Tech Stack:** Python 3.12+ / FastAPI / LangGraph / Pydantic / PostgreSQL / SQLAlchemy / Redis / OpenTelemetry / Docker Compose / GitHub Actions

---

## Phase 1: 生产基础 — 数据库 + 认证 + 配置

### Task 1: 添加 PostgreSQL + SQLAlchemy 数据层

**Files:**
- Create: `src/db/__init__.py`
- Create: `src/db/database.py` — 数据库连接管理
- Create: `src/db/models.py` — SQLAlchemy ORM 模型
- Create: `src/db/migrations/` — Alembic 迁移目录
- Modify: `pyproject.toml` — 添加 sqlalchemy, asyncpg, alembic 依赖
- Modify: `src/core/config.py` — 添加数据库配置项

**Step 1: 添加依赖到 pyproject.toml**

在 dependencies 中添加:
```toml
"sqlalchemy[asyncio]>=2.0",
"asyncpg>=0.29",
"alembic>=1.13",
"greenlet>=3.0",
```

**Step 2: 创建数据库配置**

`src/core/config.py` 添加:
```python
# Database
db_url: str = "postgresql+asyncpg://gpa:gpa@localhost:5432/growth_pilot"
db_echo: bool = False
db_pool_size: int = 10
```

**Step 3: 创建 src/db/database.py**

```python
"""Database connection management."""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from src.core.config import get_settings

_engine = None
_session_factory = None

def get_engine():
    global _engine
    if _engine is None:
        s = get_settings()
        _engine = create_async_engine(
            s.db_url,
            echo=s.db_echo,
            pool_size=s.db_pool_size,
            pool_pre_ping=True,
        )
    return _engine

def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory

async def get_db() -> AsyncSession:
    """FastAPI dependency for database sessions."""
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

**Step 4: 创建 src/db/models.py**

```python
"""SQLAlchemy ORM models."""
import uuid
from datetime import datetime
from sqlalchemy import String, Text, Float, Integer, Boolean, DateTime, JSON, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(100), default="")
    org_id: Mapped[str] = mapped_column(String(36), index=True)
    role: Mapped[str] = mapped_column(String(20), default="member")  # admin / member
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    analyses: Mapped[list["Analysis"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(200))
    plan: Mapped[str] = mapped_column(String(20), default="free")  # free / pro / enterprise
    monthly_quota: Mapped[int] = mapped_column(Integer, default=100)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    users: Mapped[list["User"]] = relationship()


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    org_id: Mapped[str] = mapped_column(String(36), index=True)
    query: Mapped[str] = mapped_column(Text)
    scope: Mapped[str] = mapped_column(String(20), default="full")
    budget: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending / running / completed / failed
    result: Mapped[dict] = mapped_column(JSON, nullable=True)
    kpi_snapshot: Mapped[dict] = mapped_column(JSON, nullable=True)
    strategy_recommendation: Mapped[Text] = mapped_column(Text, nullable=True)
    token_usage: Mapped[dict] = mapped_column(JSON, default=dict)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[Text] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    user: Mapped["User"] = relationship(back_populates="analyses")

    __table_args__ = (
        Index("ix_analyses_org_created", "org_id", "created_at"),
    )


class MemoryEntry(Base):
    __tablename__ = "memory_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    org_id: Mapped[str] = mapped_column(String(36), index=True)
    query: Mapped[str] = mapped_column(Text)
    scope: Mapped[str] = mapped_column(String(20))
    summary: Mapped[str] = mapped_column(Text)
    results_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    relevance_tags: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_memory_org_scope", "org_id", "scope"),
    )
```

**Step 5: 初始化 Alembic**

```bash
cd /Users/z/Documents/work/my-projects/growth-pilot-agent
uv add sqlalchemy asyncpg alembic greenlet
uv run alembic init src/db/migrations
```

配置 `alembic.ini` 和 `src/db/migrations/env.py` 使用 async engine。

**Step 6: 生成初始迁移**

```bash
uv run alembic revision --autogenerate -m "initial tables"
uv run alembic upgrade head
```

**Step 7: Commit**

```bash
git add src/db/ pyproject.toml uv.lock alembic.ini
git commit -m "feat: add PostgreSQL database layer with SQLAlchemy ORM models"
```

---

### Task 2: JWT 认证 + 多租户

**Files:**
- Create: `src/auth/__init__.py`
- Create: `src/auth/jwt.py` — JWT token 生成/验证
- Create: `src/auth/security.py` — 密码哈希 + 验证
- Create: `src/auth/dependencies.py` — FastAPI auth dependencies
- Modify: `pyproject.toml` — 添加 python-jose, passlib 依赖
- Modify: `src/web.py` — 集成 auth middleware

**Step 1: 添加依赖**

```toml
"python-jose[cryptography]>=3.3",
"passlib[bcrypt]>=1.7",
```

**Step 2: 创建 src/auth/security.py**

```python
"""Password hashing and verification."""
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)
```

**Step 3: 创建 src/auth/jwt.py**

```python
"""JWT token management."""
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from src.core.config import get_settings

ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    s = get_settings()
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=s.jwt_expire_hours))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, s.jwt_secret, algorithm=ALGORITHM)

def decode_access_token(token: str) -> dict | None:
    s = get_settings()
    try:
        return jwt.decode(token, s.jwt_secret, algorithms=[ALGORITHM])
    except JWTError:
        return None
```

**Step 4: 创建 src/auth/dependencies.py**

```python
"""FastAPI authentication dependencies."""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.database import get_db
from src.db.models import User
from src.auth.jwt import decode_access_token
from sqlalchemy import select

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")
    return user
```

**Step 5: 修改 src/core/config.py 添加 JWT 配置**

```python
jwt_secret: str = "change-me-in-production"
jwt_expire_hours: int = 24
```

**Step 6: Commit**

```bash
git add src/auth/ src/core/config.py pyproject.toml uv.lock
git commit -m "feat: add JWT authentication with multi-tenant support"
```

---

### Task 3: 重构 Web API — 认证 + 分析持久化 + 组织隔离

**Files:**
- Modify: `src/web.py` — 添加认证路由、分析 CRUD、组织隔离
- Create: `src/api_models.py` — 更新 API request/response 模型（已有，需增强）

**Step 1: 添加认证路由到 web.py**

```python
# 新增注册和登录路由
@router.post("/auth/register")
@router.post("/auth/login")
@router.get("/auth/me")
```

**Step 2: 修改分析路由需要认证**

```python
@router.post("/analyze")
async def analyze(user: User = Depends(get_current_user), ...):
    # 创建 Analysis 记录
    # 执行分析（注入 org_id 隔离数据）
    # 更新 Analysis 记录结果
    # 返回结果
```

**Step 3: 添加分析历史查询路由**

```python
@router.get("/analyses")
@router.get("/analyses/{analysis_id}")
```

**Step 4: Commit**

```bash
git add src/web.py src/api_models.py
git commit -m "feat: add auth endpoints and analysis persistence to web API"
```

---

## Phase 2: Agent 引擎升级

### Task 4: 领域泛化 — 去除货运硬编码

**Files:**
- Modify: `src/prompts/templates/agents/*.py` — 移除货运特定描述
- Modify: `src/prompts/templates/components/*.py` — 泛化业务上下文
- Modify: `src/core/chief.py` — system prompt 泛化
- Modify: `src/agents/*.py` — 移除 `_build_sample_*` 中货运特定逻辑
- Modify: `docs/BRD.md` — 泛化行业背景（改为通用 O2O/电商增长）

**Step 1: 创建通用业务上下文**

在 prompt 中将:
- "货运" → "业务"
- "货运意向" → "转化意向"
- "搬家" → "核心使用场景"
- "滴滴集团用户池" → "平台用户池"
- "金刚位" → "核心资源位"

**Step 2: 将 sample data 生成改为行业无关**

每个 Expert 的 `_build_sample_*` 方法改为通用数据结构，通过 `industry_config` 参数注入行业特定逻辑。

**Step 3: Commit**

```bash
git add src/prompts/ src/agents/ docs/
git commit -m "refactor: generalize domain logic from freight to generic growth"
```

---

### Task 5: Evaluator-Optimizer 循环实现

**Files:**
- Modify: `src/core/chief.py` — 实现 Phase 3 (evaluate_and_refine)
- Create: `src/core/evaluator.py` — 质量评估器

**Step 1: 创建 src/core/evaluator.py**

```python
"""Quality evaluator for expert agent outputs."""
from pydantic import BaseModel
from src.core.llm_factory import create_llm

class QualityScore(BaseModel):
    completeness: float  # 完整性 0-1
    actionability: float  # 可操作性 0-1
    data_grounding: float  # 数据支撑度 0-1
    overall: float  # 综合评分
    reasoning: str  # 改进建议

EVALUATE_PROMPT = """You are a quality evaluator for growth analysis outputs.
Rate the following analysis on these dimensions (0.0-1.0):
- completeness: Does it cover all relevant aspects?
- actionability: Are the recommendations specific and implementable?
- data_grounding: Are claims backed by data/metrics?

Expert: {expert_name}
Query: {query}
Analysis: {analysis}

Respond in JSON: {{"completeness": ..., "actionability": ..., "data_grounding": ..., "overall": ..., "reasoning": "..."}}
"""

async def evaluate_expert_output(
    expert_name: str,
    query: str,
    analysis: str,
    *,
    tier: str = "fast",
) -> QualityScore:
    llm = create_llm(tier=tier)
    prompt = EVALUATE_PROMPT.format(expert_name=expert_name, query=query, analysis=analysis)
    response = await llm.ainvoke(prompt)
    # Parse JSON response into QualityScore
    ...
```

**Step 2: 在 chief.py 实现 _evaluate_and_refine**

```python
async def _evaluate_and_refine(
    self, expert_results: dict, query: str, *, max_rounds: int = 2, threshold: float = 0.7,
) -> dict:
    for round_num in range(max_rounds):
        scores = {}
        for name, result in expert_results.items():
            score = await evaluate_expert_output(name, query, str(result))
            scores[name] = score
        low_quality = {n: s for n, s in scores.items() if s.overall < threshold}
        if not low_quality:
            break
        for name, score in low_quality.items():
            expert_results[name] = await self._reinvoke_expert(name, query, feedback=score.reasoning)
    return expert_results
```

**Step 3: Commit**

```bash
git add src/core/evaluator.py src/core/chief.py
git commit -m "feat: implement evaluator-optimizer loop for expert output quality"
```

---

### Task 6: 结构化 Agent 通信 — Pydantic 模型替代 JSON 字符串

**Files:**
- Modify: `src/core/expert.py` — analyze() 返回 AgentResult
- Modify: `src/expert_tools.py` — tool 函数返回 Pydantic
- Modify: `src/core/chief.py` — context accumulation 用模型属性
- Modify: `src/core/models.py` — 确保所有 Result 模型完整

**Step 1: 修改 ExpertAgentBase.analyze() 签名**

```python
# Before: returns str
async def analyze(self, params: dict | str) -> str:
    ...
    return json.dumps(results, ensure_ascii=False, default=str)

# After: returns AgentResult
async def analyze(self, params: dict | str) -> AgentResult:
    results = self._execute_pipeline(params)
    llm_analysis = await self._synthesize(str(results), params)
    return self._build_result(results, llm_analysis)
```

**Step 2: 更新 expert_tools.py 的 @tool 函数**

每个 tool 函数返回对应的 Pydantic Result 模型，而不是 JSON 字符串。

**Step 3: 更新 Chief Agent 的 _accumulate_context**

直接读 Pydantic 模型属性，不再 json.loads()。

**Step 4: Commit**

```bash
git add src/core/expert.py src/core/models.py src/expert_tools.py src/core/chief.py
git commit -m "feat: replace JSON string communication with Pydantic models between agents"
```

---

## Phase 3: 可观测性 + 测试

### Task 7: OpenTelemetry 集成

**Files:**
- Create: `src/core/observability.py` — OTEL 设置 + CostTracker
- Modify: `src/middleware/__init__.py` — 添加 OTEL middleware
- Modify: `pyproject.toml` — 添加 opentelemetry 依赖
- Modify: `src/core/chief.py` — 集成 tracing spans

**Step 1: 添加依赖**

```toml
"opentelemetry-api>=1.20",
"opentelemetry-sdk>=1.20",
"opentelemetry-instrumentation-fastapi>=0.41b0",
```

**Step 2: 创建 src/core/observability.py**

```python
"""Observability: OpenTelemetry setup and cost tracking."""
import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

tracer = trace.get_tracer("growth-pilot-agent")

def setup_telemetry(service_name: str = "growth-pilot-agent"):
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

class CostTracker:
    """Track token usage and estimated cost per agent per run."""
    # GPT-4o pricing (per 1M tokens)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "deepseek-chat": {"input": 0.14, "output": 0.28},
    }

    def __init__(self):
        self._records: list[dict] = []

    def record(self, agent: str, model: str, input_tokens: int, output_tokens: int):
        prices = self.PRICING.get(model, {"input": 0.5, "output": 1.5})
        cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
        self._records.append({
            "agent": agent, "model": model,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        })

    def report(self) -> dict:
        total_cost = sum(r["cost_usd"] for r in self._records)
        by_agent = {}
        for r in self._records:
            by_agent.setdefault(r["agent"], {"cost": 0, "calls": 0})
            by_agent[r["agent"]]["cost"] += r["cost_usd"]
            by_agent[r["agent"]]["calls"] += 1
        return {"total_cost_usd": round(total_cost, 6), "by_agent": by_agent, "records": self._records}
```

**Step 3: 在 Chief Agent 中添加 tracing**

每个 Phase 用 `tracer.start_as_current_span()` 包裹，记录 agent 名称、耗时、token 使用量。

**Step 4: Commit**

```bash
git add src/core/observability.py src/middleware/ src/core/chief.py pyproject.toml uv.lock
git commit -m "feat: add OpenTelemetry tracing and per-agent cost tracking"
```

---

### Task 8: 综合测试套件

**Files:**
- Create: `tests/conftest.py` — 共享 fixtures (mock LLM, test DB)
- Create: `tests/test_api/` — API 集成测试
- Create: `tests/test_agents/` — Agent 单元测试
- Modify: 各 agent 和 tool 测试

**Step 1: 创建测试基础设施 tests/conftest.py**

```python
"""Shared test fixtures."""
import pytest
from unittest.mock import AsyncMock
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AsyncMock(content='{"summary": "test", "confidence": 0.8}'))
    return llm

@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()

@pytest.fixture
async def client(db_session):
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
```

**Step 2: 编写 API 测试**

```python
# tests/test_api/test_auth.py
async def test_register(client):
    response = await client.post("/api/v1/auth/register", json={...})
    assert response.status_code == 201

async def test_login(client):
    response = await client.post("/api/v1/auth/login", json={...})
    assert response.status_code == 200
    assert "access_token" in response.json()

# tests/test_api/test_analyze.py
async def test_analyze_requires_auth(client):
    response = await client.post("/api/v1/analyze", json={...})
    assert response.status_code == 401
```

**Step 3: 编写 Agent 测试**

```python
# tests/test_agents/test_prospect.py
async def test_prospect_pipeline(mock_llm):
    agent = ProspectExpert(llm=mock_llm)
    result = await agent.analyze({"query": "分析新客获取效果"})
    assert isinstance(result, ProspectResult)
    assert result.user_count > 0
    assert 0 <= result.analysis.confidence <= 1
```

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: add comprehensive test suite with API and agent tests"
```

---

## Phase 4: CI/CD + 部署

### Task 9: GitHub Actions CI Pipeline

**Files:**
- Create: `.github/workflows/ci.yml` — lint + test + build
- Create: `.github/workflows/deploy.yml` — Docker build + push (optional)

**Step 1: 创建 .github/workflows/ci.yml**

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --group dev
      - run: uv run ruff check src/ tests/

  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: gpa
          POSTGRES_PASSWORD: gpa
          POSTGRES_DB: growth_pilot_test
        ports: ["5432:5432"]
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --group dev
      - env:
          GPA_LLM_API_KEY: ${{ secrets.GPA_LLM_API_KEY }}
          GPA_DEMO_MODE: "true"
          DB_URL: postgresql+asyncpg://gpa:gpa@localhost:5432/growth_pilot_test
        run: uv run pytest tests/ -v --tb=short

  docker:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: deploy/Dockerfile
          push: false
          tags: growth-pilot-agent:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

**Step 2: Commit**

```bash
git add .github/
git commit -m "ci: add GitHub Actions pipeline with lint, test, and docker build"
```

---

### Task 10: Docker Compose 全栈部署

**Files:**
- Modify: `docker-compose.yml` — 添加 PostgreSQL + Redis 服务
- Modify: `deploy/Dockerfile` — 优化多阶段构建
- Create: `.env.docker` — Docker 环境变量模板

**Step 1: 重写 docker-compose.yml**

```yaml
services:
  app:
    build:
      context: .
      dockerfile: deploy/Dockerfile
    ports:
      - "8000:8000"
    env_file: .env.docker
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./data:/app/data

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${DB_USER:-gpa}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-gpa}
      POSTGRES_DB: ${DB_NAME:-growth_pilot}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-gpa}"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  pgdata:
```

**Step 2: 优化 Dockerfile**

多阶段构建，使用 uv 的 slim 镜像，安装系统依赖（LightGBM 等）。

**Step 3: Commit**

```bash
git add docker-compose.yml deploy/Dockerfile .env.docker
git commit -m "feat: add full-stack Docker Compose with PostgreSQL and Redis"
```

---

## Phase 5: 文档更新

### Task 11: 更新 README + docs

**Files:**
- Modify: `README.md` — 反映新架构和部署方式
- Modify: `docs/BRD.md` — 泛化行业背景
- Modify: `docs/TECHNICAL.md` — 添加数据库/认证/可观测性章节
- Modify: `docs/GUIDE.md` — 添加部署指南和 API 文档

**Step 1: 更新 README.md**

- 移除所有货运特定描述
- 更新架构图（添加 DB/Auth/OTEL 层）
- 更新 Quick Start（添加 Docker Compose 方式）
- 添加 API 文档链接

**Step 2: 更新 BRD.md**

- 行业背景泛化为"O2O/电商增长"
- 保留货运作为示例场景之一

**Step 3: Commit**

```bash
git add README.md docs/
git commit -m "docs: update documentation for production-grade SaaS platform"
```

---

## 实施优先级

| Phase | Tasks | 预计工作量 | 影响 |
|-------|-------|-----------|------|
| Phase 1 | Task 1-3 | 核心 | 数据库 + 认证是 SaaS 基石 |
| Phase 2 | Task 4-6 | 核心 | Agent 引擎升级，质量保障 |
| Phase 3 | Task 7-8 | 重要 | 可观测性和测试覆盖 |
| Phase 4 | Task 9-10 | 重要 | CI/CD 和部署自动化 |
| Phase 5 | Task 11 | 收尾 | 文档与 README |

**关键路径**: Task 1 → Task 2 → Task 3 → Task 6 (API 依赖 DB, Auth 依赖 DB, Agent 通信依赖 Pydantic)
