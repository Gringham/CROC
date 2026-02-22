# /srv/annotate/app/main.py
import os, hashlib, base64, mimetypes, sqlite3, json, re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from uvicorn import run

# --------- Paths & Settings ---------
IMAGE_ROOT   = Path(os.getenv("IMAGE_ROOT", "/data/images"))
MANIFEST_DIR = Path(os.getenv("MANIFEST_DIR", "/data/manifests"))
DB_PATH      = Path(os.getenv("DB_PATH", "/data/db/ratings.db"))
LOG_DIR      = Path(os.getenv("LOG_DIR", "/data/db/change_logs"))
# If set to "true", always rebuild the manifest even if a cached file exists
FORCE_REBUILD_MANIFEST = os.getenv("FORCE_REBUILD_MANIFEST", "false").lower() == "true"

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = FastAPI(
    title="Image–Text Match API",
    docs_url=None, redoc_url=None, openapi_url=None  # hide auto docs
)

# --------- Helpers ---------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def file_id_for_relpath(rel: str) -> str:
    # stable, opaque ID from relative path
    h = hashlib.sha1(rel.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(h).decode("ascii").rstrip("=")

def list_images(root: Path) -> List[str]:
    out: List[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            out.append(p.relative_to(root).as_posix())
    return sorted(out)

def group_by_folder(relpaths: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for rel in relpaths:
        folder = str(Path(rel).parent).replace("\\", "/")
        groups.setdefault(folder, []).append(rel)
    return groups

def round_robin(keys: List[str], B: int) -> List[List[str]]:
    buckets: List[List[str]] = [[] for _ in range(B)]
    for i, k in enumerate(keys):
        buckets[i % B].append(k)
    return buckets

def sanitize_seed(seed: str) -> str:
    """
    Limit seed used in filenames / shuffles to safe chars and length.
    This improves filesystem safety and keeps determinism.
    """
    s = (seed or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    return s[:64]

# --------- SQLite (schema & helpers) ---------
def ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("""
        CREATE TABLE IF NOT EXISTS ratings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            pid TEXT NOT NULL,
            image_id TEXT NOT NULL,
            rating REAL NOT NULL,
            idx_in_session INTEGER NOT NULL,
            total_in_session INTEGER NOT NULL,
            batch INTEGER NOT NULL,
            batches INTEGER NOT NULL,
            seed TEXT,
            ua TEXT,
            ip TEXT
        )""")
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_pid_image ON ratings(pid, image_id)")
        con.execute("""
        CREATE TABLE IF NOT EXISTS consents(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            pid TEXT NOT NULL,
            consented INTEGER NOT NULL,
            study_id TEXT,
            session_id TEXT,
            batch INTEGER,
            batches INTEGER,
            seed TEXT,
            cap INTEGER,
            raw_query TEXT,
            ua TEXT
        )""")
        con.commit()

def save_rating_upsert(rec: dict):
    with sqlite3.connect(DB_PATH, timeout=30) as con:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("""
        INSERT INTO ratings
            (ts,pid,image_id,rating,idx_in_session,total_in_session,batch,batches,seed,ua,ip)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(pid,image_id) DO UPDATE SET
            ts=excluded.ts,
            rating=excluded.rating,
            idx_in_session=excluded.idx_in_session,
            total_in_session=excluded.total_in_session,
            batch=excluded.batch,
            batches=excluded.batches,
            seed=excluded.seed,
            ua=excluded.ua,
            ip=excluded.ip
        """, (
            rec["ts"], rec["pid"], rec["image_id"], rec["rating"], rec["index"],
            rec["total"], rec["batch"], rec["batches"], rec["seed"], rec["ua"], rec["ip"]
        ))
        con.commit()

def save_consent(rec: dict):
    with sqlite3.connect(DB_PATH, timeout=30) as con:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("""
        INSERT INTO consents
            (ts,pid,consented,study_id,session_id,batch,batches,seed,cap,raw_query,ua)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            rec["ts"], rec["pid"], 1 if rec["consented"] else 0,
            rec.get("study_id",""), rec.get("session_id",""),
            rec.get("batch"), rec.get("batches"),
            rec.get("seed",""), rec.get("cap"),
            rec.get("raw_query",""), rec.get("ua","")
        ))
        con.commit()

def append_log_line(pid: str, event: dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{pid}.jsonl"
    event_with_ts = {"ts": now_iso(), "pid": pid, **event}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_with_ts, ensure_ascii=False) + "\n")

# --------- In-process image maps ---------
REL2ID: Dict[str, str] = {}
ID2PATH: Dict[str, Path] = {}

def build_id_maps():
    REL2ID.clear()
    ID2PATH.clear()
    rels = list_images(IMAGE_ROOT)
    for rel in rels:
        _id = file_id_for_relpath(rel)
        REL2ID[rel] = _id
        ID2PATH[_id] = (IMAGE_ROOT / rel)

# --------- Manifest (deterministic & cached) ---------
def get_manifest(B: int, seed: str):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    seed_clean = sanitize_seed(seed)
    fname = f"batch_manifest__B{B}__seed_{seed_clean or 'none'}.json"
    fpath = MANIFEST_DIR / fname

    if fpath.exists() and not FORCE_REBUILD_MANIFEST:
        try:
            return json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            # fall through to rebuild if cache is corrupt
            pass

    if not REL2ID:
        build_id_maps()

    groups = group_by_folder(list(REL2ID.keys()))
    group_keys = sorted(groups.keys())
    buckets = round_robin(group_keys, B)

    def seeded_shuffle(items, key):
        import random
        rnd = random.Random(key)
        rnd.shuffle(items)

    batches: List[List[str]] = []
    for b, gkeys in enumerate(buckets):
        items: List[str] = []
        for gk in gkeys:
            for rel in sorted(groups[gk]):
                items.append(REL2ID[rel])  # store only IDs
        # deterministic per-bucket shuffle
        seeded_shuffle(items, f"{seed_clean or 'no-seed'}::{b}")
        batches.append(items)

    m = {
        "version": 1,
        "createdAt": now_iso(),
        "numBatches": B,
        "seed": seed_clean or None,
        "totalFiles": len(REL2ID),
        "totalGroups": len(group_keys),
        "batches": batches
    }
    fpath.write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m

# --------- Pydantic models ---------
class RateIn(BaseModel):
    pid: str
    imageId: str
    rating: float
    index: int
    total: int
    batch: int
    batches: int
    seed: Optional[str] = None
    ua: Optional[str] = None

class LogIn(BaseModel):
    pid: str
    action: str
    index: Optional[int] = None
    extra: Optional[dict] = None

class ConsentIn(BaseModel):
    pid: str
    consented: bool
    studyId: Optional[str] = None
    sessionId: Optional[str] = None
    batch: Optional[int] = None
    batches: Optional[int] = None
    seed: Optional[str] = None
    cap: Optional[int] = None
    rawQuery: Optional[str] = None
    ua: Optional[str] = None

# --------- Friendly 500s ---------
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    import traceback
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(tb)  # visible via `docker compose logs app`
    return PlainTextResponse(str(exc), status_code=500)

# --------- API Endpoints ---------
@app.get("/api/batch-info")
def api_batch_info(
    batch: int = 1,
    batches: int = 1,
    cap: int = Query(0, alias="max"),
    seed: str = ""
):
    B  = max(1, int(batches))
    bi = max(1, min(B, int(batch)))
    m = get_manifest(B, seed)
    items = m["batches"][bi-1]
    total_all = len(items)
    total = min(cap, total_all) if cap and cap > 0 else total_all
    return {"ok": True, "numBatches": m["numBatches"], "seed": m["seed"], "totalAll": total_all, "total": total}

@app.get("/api/image-meta")
def api_image_meta(batch: int = 1, batches: int = 1, i: int = 0, seed: str = ""):
    B  = max(1, int(batches))
    bi = max(1, min(B, int(batch)))
    m = get_manifest(B, seed)
    items = m["batches"][bi-1]
    if i < 0 or i >= len(items):
        raise HTTPException(400, "Index out of range")
    return {"id": items[i]}

@app.get("/api/resolve-id")
def api_resolve_id(rel: str):
    if not rel:
        raise HTTPException(400, "Missing rel")
    if not REL2ID:
        build_id_maps()
    rel = rel.lstrip("/").replace("\\", "/")
    _id = REL2ID.get(rel)
    if not _id:
        build_id_maps()
        _id = REL2ID.get(rel)
        if not _id:
            raise HTTPException(404, f"Relpath not found: {rel}")
    return {"id": _id}

@app.get("/api/progress")
def api_progress(pid: str, batch: int = 1, batches: int = 1, cap: int = Query(0, alias="max"), seed: str = ""):
    if not pid:
        raise HTTPException(400, "Missing pid")

    B  = max(1, int(batches))
    bi = max(1, min(B, int(batch)))
    m = get_manifest(B, seed)
    items = m["batches"][bi-1]
    total_all = len(items)
    total = min(cap, total_all) if cap and cap > 0 else total_all
    items = items[:total]

    id2idx: Dict[str, int] = {img_id: i for i, img_id in enumerate(items)}

    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT image_id, rating FROM ratings WHERE pid=?", (pid,))
        rows = cur.fetchall()

    ratings_by_index: Dict[int, float] = {}
    for img_id, rating in rows:
        if img_id in id2idx:
            ratings_by_index[id2idx[img_id]] = float(rating)

    completed = sorted(ratings_by_index.keys())
    next_idx = total
    for i in range(total):
        if i not in ratings_by_index:
            next_idx = i
            break

    append_log_line(pid, {"action": "resume", "batch": bi, "batches": B, "seed": sanitize_seed(seed) or "", "next": next_idx, "done": len(completed), "total": total})
    return {"ok": True, "total": total, "completed": completed, "ratings": {str(k): v for k, v in ratings_by_index.items()}, "next": next_idx}

@app.post("/api/log")
def api_log(entry: LogIn):
    if not entry.pid or not entry.action:
        raise HTTPException(400, "Missing pid or action")
    append_log_line(entry.pid, {"action": entry.action, "index": entry.index, "extra": entry.extra or {}})
    return {"ok": True}

@app.get("/img/{image_id}")
def get_image(image_id: str):
    p = ID2PATH.get(image_id)
    if not p or not p.exists():
        build_id_maps()
        p = ID2PATH.get(image_id)
        if not p or not p.exists():
            raise HTTPException(404, "Image not found")
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"

    def file_iter(chunk=1024*256):
        with open(p, "rb") as f:
            while True:
                buf = f.read(chunk)
                if not buf: break
                yield buf

    headers = {"Cache-Control": "public, max-age=31536000, immutable"}
    return StreamingResponse(file_iter(), media_type=mime, headers=headers)

@app.post("/api/rate")
def api_rate(req: RateIn, request: Request):
    if not req.pid or not req.imageId:
        raise HTTPException(400, "Missing pid or imageId")
    if req.imageId not in ID2PATH:
        build_id_maps()
        if req.imageId not in ID2PATH:
            raise HTTPException(400, "Unknown imageId")

    rec = {
        "ts": now_iso(),
        "pid": req.pid.strip(),
        "image_id": req.imageId,
        "rating": float(req.rating),
        "index": int(req.index),
        "total": int(req.total),
        "batch": int(req.batch),
        "batches": int(req.batches),
        "seed": sanitize_seed(req.seed or ""),
        "ua": (req.ua or "")[:512],
        "ip": ""  # DO NOT COLLECT IPs
    }
    save_rating_upsert(rec)
    append_log_line(rec["pid"], {"action": "rate", "image_id": rec["image_id"], "index": rec["index"], "rating": rec["rating"]})
    return {"ok": True}

@app.post("/api/consent")
def api_consent(req: ConsentIn):
    if not req.pid:
        raise HTTPException(400, "Missing pid")
    rec = {
        "ts": now_iso(),
        "pid": req.pid.strip(),
        "consented": bool(req.consented),
        "study_id": (req.studyId or "")[:128],
        "session_id": (req.sessionId or "")[:128],
        "batch": int(req.batch) if req.batch is not None else None,
        "batches": int(req.batches) if req.batches is not None else None,
        "seed": sanitize_seed(req.seed or ""),
        "cap": int(req.cap) if req.cap is not None else None,
        "raw_query": (req.rawQuery or "")[:4096],
        "ua": (req.ua or "")[:512],
    }
    save_consent(rec)
    append_log_line(rec["pid"], {"action": "consent", "consented": rec["consented"], "study_id": rec["study_id"], "session_id": rec["session_id"]})
    return {"ok": True}

@app.get("/api/text-for-image")
def api_text_for_image(id: str):
    """
    Given an opaque image ID, locate sibling row.json and return the text to display.
    Mapping from filename suffix -> field:
      __fm.jpg, __fn.jpg -> 'prompts'
      __im.jpg, __in.jpg -> 'contrast_prompts'
    """
    p = ID2PATH.get(id)
    if not p or not p.exists():
        build_id_maps()
        p = ID2PATH.get(id)
        if not p or not p.exists():
            raise HTTPException(404, "Image not found")

    row_json = p.parent / "row.json"
    if not row_json.exists():
        raise HTTPException(404, "row.json not found for this image")

    try:
        meta = json.loads(row_json.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Failed to parse row.json: {e}")

    name = p.name
    if name.endswith("__fm.jpg") or name.endswith("__fn.jpg"):
        field = "prompts"
    elif name.endswith("__im.jpg") or name.endswith("__in.jpg"):
        field = "contrast_prompts"
    else:
        field = "prompts" if "prompts" in meta else "contrast_prompts"

    def take_text(v):
        if isinstance(v, list):
            for s in v:
                if isinstance(s, str) and s.strip():
                    return s.strip()
            return ""
        return (v or "").strip() if isinstance(v, str) else ""

    text = take_text(meta.get(field, ""))
    return {"ok": True, "id": id, "field": field, "text": text}

@app.get("/healthz")
def health():
    return {"ok": True, "images": len(ID2PATH)}

# --------- Entrypoint ---------
if __name__ == "__main__":
    ensure_db()
    build_id_maps()
    run("main:app", host="0.0.0.0", port=8000, reload=False)
