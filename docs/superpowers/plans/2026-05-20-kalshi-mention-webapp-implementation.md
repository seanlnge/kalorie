# Kalshi Mention Webapp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working web app (FastAPI + React/Tailwind/shadcn) that lists open earnings mention markets, runs background model jobs with EX-99 uploads, reuses event/company historical cache, and shows run results with default/latest run selection and run switching.

**Architecture:** Build a thin FastAPI orchestrator around existing `kalorie.app.cli` workflows and existing Kalshi/model modules. Persist cache at event/company scope and keep run-scoped artifacts separate. Add a React app that consumes API/WebSocket endpoints for jobs and run results.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, Pydantic v2, Typer CLI orchestration, React + Vite + TypeScript, Tailwind CSS, shadcn/ui, pytest, Ruff, Vitest.

---

### Task 1: Backend skeleton and run-store contracts

**Files:**
- Create: `src/kalorie/webapi/__init__.py`
- Create: `src/kalorie/webapi/schemas.py`
- Create: `src/kalorie/webapi/run_store.py`
- Create: `tests/unit/test_webapi_run_store.py`

- [ ] **Step 1: Write failing run-store tests**
- [ ] **Step 2: Run tests and confirm failure**
- [ ] **Step 3: Implement event/company cache + run path logic**
- [ ] **Step 4: Run tests and confirm pass**

### Task 2: Job registry, idempotency, and scheduler budgets

**Files:**
- Create: `src/kalorie/webapi/job_registry.py`
- Create: `src/kalorie/webapi/job_runner.py`
- Create: `tests/unit/test_webapi_job_registry.py`

- [ ] **Step 1: Write failing tests for idempotency and duplicate submit behavior**
- [ ] **Step 2: Run tests and confirm failure**
- [ ] **Step 3: Implement idempotency-key dedupe + budget-aware queueing**
- [ ] **Step 4: Run tests and confirm pass**

### Task 3: Kalshi market API service and normalization

**Files:**
- Create: `src/kalorie/webapi/kalshi_service.py`
- Create: `tests/unit/test_webapi_kalshi_service.py`

- [ ] **Step 1: Write failing tests for open mention market filtering and Decimal price normalization**
- [ ] **Step 2: Run tests and confirm failure**
- [ ] **Step 3: Implement service with fixed-point contract fields**
- [ ] **Step 4: Run tests and confirm pass**

### Task 4: FastAPI routes and WebSocket stream

**Files:**
- Create: `src/kalorie/webapi/main.py`
- Create: `tests/integration/test_webapi_routes.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing API tests for markets/runs/jobs endpoints**
- [ ] **Step 2: Run tests and confirm failure**
- [ ] **Step 3: Implement routes and websocket event fanout**
- [ ] **Step 4: Run tests and confirm pass**

### Task 5: Pipeline orchestration with leakage cutoff and cache reuse

**Files:**
- Modify: `src/kalorie/webapi/job_runner.py`
- Create: `tests/integration/test_webapi_job_flow.py`

- [ ] **Step 1: Write failing test for effective decision cutoff propagation**
- [ ] **Step 2: Run test and confirm failure**
- [ ] **Step 3: Implement cache-aware job execution and cutoff filtering contract**
- [ ] **Step 4: Run tests and confirm pass**

### Task 6: React app scaffold with Tailwind and shadcn

**Files:**
- Create: `web/*` (Vite React TypeScript scaffold)
- Create: `web/src/lib/api.ts`
- Create: `web/src/lib/types.ts`
- Create: `web/src/components/ui/*` (shadcn baseline)

- [ ] **Step 1: Scaffold frontend and install Tailwind/shadcn dependencies**
- [ ] **Step 2: Add API client and shared types**
- [ ] **Step 3: Verify app builds and tests run**

### Task 7: Markets pages, run selector, uploads, and job rail

**Files:**
- Create: `web/src/pages/HomePage.tsx`
- Create: `web/src/pages/MarketPage.tsx`
- Create: `web/src/components/markets/RunSelector.tsx`
- Create: `web/src/components/markets/Ex99Dropzone.tsx`
- Create: `web/src/components/jobs/RightRailJobs.tsx`
- Create: `web/src/components/markets/PredictionTable.tsx`
- Create: `web/src/hooks/useJobsStream.ts`
- Create: `web/src/tests/market-page.test.tsx`

- [ ] **Step 1: Write failing tests for default latest run + run dropdown switching**
- [ ] **Step 2: Run tests and confirm failure**
- [ ] **Step 3: Implement pages/components and shadcn completion toast deep-link behavior**
- [ ] **Step 4: Run tests and confirm pass**

### Task 8: End-to-end verification and polish

**Files:**
- Modify: backend/frontend files as needed from test failures

- [ ] **Step 1: Run backend test suite**
- [ ] **Step 2: Run frontend tests/build**
- [ ] **Step 3: Run smoke flow with two phrase markets in same event and verify shared cache**
- [ ] **Step 4: Run lint checks**
- [ ] **Step 5: Update plan checkboxes with final status**
