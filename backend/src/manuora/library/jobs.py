"""Upload jobs: place a PDF in data/sources, parse, chunk, embed and index it, one job at a time.

State lives in SQLite (data/library/jobs.sqlite) so progress survives page refreshes; a job that was
running when the process stopped is marked failed and can be retried from the stage it reached.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from manuora.ingestion.files import resolve_output_dir, resolve_source_path
from manuora.ingestion.naming import document_id as make_document_id, slug
from manuora.knowledge.manifests import load_manifest
from manuora.knowledge.schemas import ProductManifest
from manuora.library import catalog

STAGES = ("upload", "parse", "chunk", "embed", "index")
TYPES = ("user_manual", "administration_manual", "server_manual", "developer_manual",
         "getting_started_guide", "installation_manual", "reference_manual", "other")
MAX_BYTES = 1024 ** 3


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_filename(name: str) -> str:
    name = Path(name.replace("\\", "/")).name
    stem = re.sub(r"[^\w.\- ]+", "_", Path(name).stem).strip(" ._") or "manual"
    return f"{stem[:120]}.pdf"


class JobStore:
    COLUMNS = ("id", "created_at", "updated_at", "filename", "size", "upload_path", "product_id", "display_name",
               "version", "title", "type", "document_id", "replaces", "manifest", "stage", "status", "progress",
               "detail", "error", "timings")

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with self._connect() as connection:
            connection.execute(f"CREATE TABLE IF NOT EXISTS jobs ({', '.join(c + ' TEXT' for c in self.COLUMNS)},"
                               " PRIMARY KEY(id))")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _row(self, row) -> dict:
        job = dict(row)
        job["size"] = int(job["size"] or 0)
        job["progress"] = float(job["progress"] or 0)
        job["timings"] = json.loads(job["timings"] or "{}")
        return job

    def create(self, **fields) -> dict:
        job = {c: None for c in self.COLUMNS} | fields | {
            "id": uuid.uuid4().hex[:12], "created_at": now(), "updated_at": now(),
            "stage": "upload", "status": "queued", "progress": 0, "timings": "{}"}
        with self.lock, self._connect() as connection:
            connection.execute(f"INSERT INTO jobs VALUES ({','.join('?' * len(self.COLUMNS))})",
                               [job[c] for c in self.COLUMNS])
        return self.get(job["id"])

    def get(self, job_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, limit: int = 30) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def update(self, job_id: str, **fields) -> None:
        if "timings" in fields:
            fields["timings"] = json.dumps(fields["timings"])
        fields["updated_at"] = now()
        with self.lock, self._connect() as connection:
            connection.execute(f"UPDATE jobs SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
                               [*fields.values(), job_id])

    def next_queued(self) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1").fetchone()
        return self._row(row) if row else None

    def delete(self, job_id: str) -> None:
        with self.lock, self._connect() as connection:
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def interrupt_running(self) -> None:
        with self.lock, self._connect() as connection:
            connection.execute("UPDATE jobs SET status = 'failed', error = 'Interrupted: the app stopped while "
                               "this was running. Retry continues from this step.' WHERE status = 'running'")


class Library:
    """Uploads, the job worker and removal. `index_factory` builds a LibraryIndex; `on_change` runs after
    the indexes change (the app clears its caches there)."""

    def __init__(self, root: Path, source_root: Path, data_dir: Path, index_factory: Callable,
                 on_change: Callable[[], None] = lambda: None, gpu_busy: Callable[[], bool] = lambda: False):
        self.root = root
        self.source_root = source_root
        self.data_dir = data_dir
        self.uploads = data_dir / "uploads"
        self.logs = data_dir / "logs"
        self.trash = data_dir / "trash"
        self.store = JobStore(data_dir / "jobs.sqlite")
        self.index_factory = index_factory
        self.on_change = on_change
        self.gpu_busy = gpu_busy
        self.index_lock = threading.Lock()  # one writer for the indexes: the worker or a removal
        self._index = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.store.interrupt_running()
        self._thread = threading.Thread(target=self._run, name="library-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    @property
    def index(self):
        if self._index is None:
            self._index = self.index_factory()
        return self._index

    # ------------------------------------------------------------------ uploads
    def new_upload(self, filename: str) -> Path:
        """Where to stream an incoming file: data/library/uploads/<upload_id>/<safe name>.pdf."""
        target = self.uploads / uuid.uuid4().hex[:12] / safe_filename(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def finish_upload(self, target: Path) -> dict:
        size = target.stat().st_size if target.is_file() else 0
        with target.open("rb") as stream:
            head = stream.read(5)
        if head != b"%PDF-" or size > MAX_BYTES:
            shutil.rmtree(target.parent, ignore_errors=True)
            raise ValueError("not a PDF file" if head != b"%PDF-" else "file is larger than 1 GB")
        return {"upload_id": target.parent.name, "filename": target.name, "size": size,
                "suggestion": self.suggest(target.name)}

    def suggest(self, filename: str) -> dict:
        """Best guess of product, version, title and type from the filename and the existing library."""
        from manuora.ingestion.naming import inferred_type
        stem = Path(filename).stem
        key = slug(stem)
        manuals = catalog.manuals(self.source_root)
        products = {m.product_id: m.display_name for m in manuals}
        product_id = next((p for p, name in sorted(products.items(), key=lambda kv: -len(kv[0]))
                           if p in key or slug(name) in key), None)
        match = re.search(r"(?:^|[^\w.])v?(\d+(?:\.\d+){0,2})(?=$|[^\w.])", stem.replace("_", " ").replace("-", " "))
        version = match.group(1) if match else None
        kind = "reference_manual" if "reference" in key else inferred_type(stem)
        kind = kind if kind in TYPES else "user_manual"
        display_name = products.get(product_id)
        if product_id is None:  # unknown product: the filename's first word is the best guess
            first = re.split(r"[\s_\-.]+", stem.strip())[0]
            display_name = first if re.search(r"[A-Za-z]", first) else None
        # A new version of a known manual keeps that manual's title (latest version first).
        same_kind = sorted((m for m in manuals if m.product_id == product_id and m.type == kind),
                           key=lambda m: [int(x) if x.isdigit() else 0 for x in (m.version or "0").split(".")], reverse=True)
        if same_kind:
            title = same_kind[0].title
            if version and same_kind[0].version:
                title = re.sub(rf"\b{re.escape(same_kind[0].version)}\b", version, title)
        else:
            words = re.sub(r"[_\-]+", " ", stem)
            if display_name:
                words = re.sub(rf"^\s*{re.escape(display_name)}(?=\W|$)", "", words, flags=re.IGNORECASE)
            words = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", words)
            if version:
                words = re.sub(rf"\bv?{re.escape(version)}\b", "", words)
            title = re.sub(r"\s+", " ", re.sub(r"\b(en|EN)\s*$", "", words)).strip(" -") or stem
            title = title[:1].upper() + title[1:]
        return {"product_id": product_id, "display_name": display_name, "version": version,
                "title": title, "type": kind}

    def discard_upload(self, upload_id: str) -> None:
        if re.fullmatch(r"[0-9a-f]{12}", upload_id):
            shutil.rmtree(self.uploads / upload_id, ignore_errors=True)

    def submit(self, upload_id: str, *, display_name: str, version: str | None, title: str, type: str,
               product_id: str | None = None) -> dict:
        if not re.fullmatch(r"[0-9a-f]{12}", upload_id):
            raise ValueError("unknown upload")
        files = list((self.uploads / upload_id).glob("*.pdf"))
        if not files:
            raise ValueError("upload not found; drop the file again")
        display_name, title = display_name.strip(), title.strip()
        version = (version or "").strip() or None
        if not display_name or not title:
            raise ValueError("product and title are required")
        if version and not re.fullmatch(r"[\w.\-]{1,32}", version):
            raise ValueError("version may use letters, digits, '.', '-' and '_' only")
        if type not in TYPES:
            raise ValueError(f"type must be one of {', '.join(TYPES)}")
        manuals = catalog.manuals(self.source_root)
        known = {m.product_id: m.display_name for m in manuals}
        product_id = product_id if product_id in known else next(
            (p for p, name in known.items() if name.casefold() == display_name.casefold() or p == slug(display_name)),
            slug(display_name))
        if not product_id:
            raise ValueError("product name needs letters or digits")
        replaces = next((m.document_id for m in manuals if m.product_id == product_id and m.version == version
                         and (m.path.casefold() == files[0].name.casefold() or m.title.casefold() == title.casefold())),
                        None)
        busy = [j for j in self.store.list(200) if j["status"] in ("queued", "running") and replaces
                and j["replaces"] == replaces]
        if busy:
            raise ValueError("that manual is already being processed")
        job = self.store.create(filename=files[0].name, size=files[0].stat().st_size, upload_path=str(files[0]),
                                product_id=product_id, display_name=known.get(product_id, display_name),
                                version=version, title=title, type=type, replaces=replaces)
        self._wake.set()
        return job

    def retry(self, job_id: str) -> dict:
        job = self.store.get(job_id)
        if not job or job["status"] != "failed":
            raise ValueError("only failed jobs can be retried")
        self.store.update(job_id, status="queued", error=None, detail="Waiting to retry")
        self._wake.set()
        return self.store.get(job_id)

    def dismiss(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if not job:
            return
        if job["status"] == "running":
            raise ValueError("a running job cannot be dismissed")
        if job["status"] in ("queued", "failed") and job["stage"] == "upload" and job["upload_path"]:
            self.discard_upload(Path(job["upload_path"]).parent.name)
        self.store.delete(job_id)

    def log(self, job_id: str, tail: int = 20000) -> str:
        path = self.logs / f"{job_id}.log"
        if not re.fullmatch(r"[0-9a-f]{12}", job_id) or not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-tail:]

    # ------------------------------------------------------------------ removal
    def remove(self, document_id: str) -> dict:
        manual = next((m for m in catalog.manuals(self.source_root) if m.document_id == document_id), None)
        if manual is None:
            raise KeyError(document_id)
        if any(j["status"] in ("queued", "running") and document_id in (j["document_id"], j["replaces"])
               for j in self.store.list(200)):
            raise ValueError("this manual is being processed; wait for the job to finish")
        with self.index_lock:
            removed = self.index.remove_document(document_id, refresh=False)
            bin_dir = self.trash / f"{datetime.now():%Y%m%d_%H%M%S}_{document_id}"
            bin_dir.mkdir(parents=True, exist_ok=True)
            data = yaml.safe_load(manual.manifest.read_text(encoding="utf-8"))
            document = next(d for d in data["documents"] if d["id"] == document_id)
            pdf = manual.folder / document["path"]
            outputs = manual.folder / "processed" / document_id
            for path in (pdf, outputs):
                if path.exists():
                    shutil.move(str(path), str(bin_dir / path.name))
            shutil.copy2(manual.manifest, bin_dir / "manifest.yaml")
            data["documents"] = [d for d in data["documents"] if d["id"] != document_id]
            if data["documents"]:
                manual.manifest.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            else:
                manual.manifest.unlink()
                for folder in (manual.folder / "processed", manual.folder, manual.folder.parent):  # now-empty folders
                    if folder != self.source_root and folder.is_dir() and not any(folder.iterdir()):
                        folder.rmdir()
            self.index.refresh_identity()
        self.on_change()
        return {"document_id": document_id, "removed_chunks": removed, "moved_to": str(bin_dir.relative_to(self.root))}

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        while not self._stop.is_set():
            job = self.store.next_queued()
            if job is None:
                self._wake.wait(5)
                self._wake.clear()
                continue
            self._process(job)

    def _process(self, job: dict) -> None:
        job_id = job["id"]
        timings = job["timings"]
        self.logs.mkdir(parents=True, exist_ok=True)
        start = STAGES.index(job["stage"]) if job["stage"] in STAGES else 0
        self.store.update(job_id, status="running")
        try:
            for stage in STAGES[start:]:
                began = time.perf_counter()
                self.store.update(job_id, stage=stage, progress=0, detail=None)
                parts = getattr(self, f"_stage_{stage}")(self.store.get(job_id)) or {}
                timings.update(parts)  # a stage may report time spent on a sub-step (index reports embed)
                timings[stage] = round(time.perf_counter() - began - sum(parts.values()), 1)
                self.store.update(job_id, timings=timings)
            self.store.update(job_id, status="done", progress=1, detail="Ready to search in chat")
            self.on_change()
        except Exception as exc:  # the job shows the error; the worker keeps running
            self._log(job_id, f"\nERROR: {type(exc).__name__}: {exc}\n")
            self.store.update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:2000])

    def _log(self, job_id: str, text: str) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        with (self.logs / f"{job_id}.log").open("a", encoding="utf-8") as log:
            log.write(text)

    def _manifest_document(self, job: dict):
        manifest = load_manifest(Path(job["manifest"]))
        return manifest, next(d for d in manifest.documents if d.id == job["document_id"])

    def _stage_upload(self, job: dict) -> None:
        """Copy the PDF into data/sources/<Product>/<version>/ and add it to that folder's manifest."""
        if job["manifest"] and job["document_id"]:
            return  # placed on an earlier attempt
        upload = Path(job["upload_path"])
        manuals = catalog.manuals(self.source_root)
        same_release = [m for m in manuals if m.product_id == job["product_id"] and m.version == job["version"]]
        if same_release:
            folder = same_release[0].folder
        else:
            product_manuals = [m for m in manuals if m.product_id == job["product_id"]]
            if product_manuals:
                base = product_manuals[0].folder
                base = base.parent if product_manuals[0].version is not None else base
            else:
                base = self.source_root / (re.sub(r"[^\w.\- ]+", "_", job["display_name"]).strip(" ._") or job["product_id"])
            folder = base / job["version"] if job["version"] else base
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder / "manifest.yaml"
        data = (yaml.safe_load(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else
                {"product_id": job["product_id"], "display_name": job["display_name"], "version": job["version"],
                 "language": "en", "documents": []})
        if data.get("product_id") != job["product_id"] or (data.get("version") and str(data["version"]) != (job["version"] or "")):
            raise ValueError(f"{manifest_path} belongs to another product or version")
        target_name = upload.name
        if job["replaces"]:
            entry = next(d for d in data["documents"] if d["id"] == job["replaces"])
            old = folder / entry["path"]
            if old.exists() and old.name != target_name:
                old.unlink()
            entry.update(title=job["title"], type=job["type"], path=target_name, authority="primary")
            doc_id = job["replaces"]
        else:
            taken = {d["path"].casefold() for d in data["documents"]}
            stem, n = Path(target_name).stem, 2
            while target_name.casefold() in taken or (folder / target_name).exists():
                target_name, n = f"{stem}_{n}.pdf", n + 1
            used = {m.document_id for m in manuals}
            name = slug(Path(upload.name).stem)
            if job["version"]:  # the id ends with the version anyway: drop it from the filename part
                name = re.sub(rf"(^|_)v?{re.escape(slug(job['version']))}(?=_|$)", "", name).strip("_") or name
            doc_id = make_document_id(job["product_id"], name, job["version"], used)
            data["documents"].append({"id": doc_id, "type": job["type"], "title": job["title"],
                                      "path": target_name, "authority": "primary"})
        ProductManifest.model_validate(data)
        shutil.copy2(upload, folder / target_name)
        manifest_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        shutil.rmtree(upload.parent, ignore_errors=True)
        self._log(job["id"], f"placed {folder / target_name}\nmanifest {manifest_path} document {doc_id}\n")
        self.store.update(job["id"], manifest=str(manifest_path), document_id=doc_id, upload_path=None, progress=1)

    def _script(self, job: dict, script: str, label: str, env: dict | None = None) -> None:
        command = [sys.executable, str(self.root / "scripts" / script), "--manifest", job["manifest"],
                   "--document-id", job["document_id"]]
        self._log(job["id"], f"\n$ {' '.join(command)}\n")
        with (self.logs / f"{job['id']}.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=self.root, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})})
            began = time.perf_counter()
            while process.poll() is None:
                self.store.update(job["id"], detail=f"{label} · {int(time.perf_counter() - began)} s")
                time.sleep(2)
        if process.returncode:
            raise RuntimeError(f"{script} exited with code {process.returncode}; see the log")

    def _stage_parse(self, job: dict) -> None:
        cpu = self.gpu_busy()  # the chat model holds the GPU: parse on the CPU instead of competing for memory
        label = "Reading the PDF on CPU (chat model is loaded)" if cpu else "Reading the PDF"
        self._script(job, "parse_document.py", label, {"CUDA_VISIBLE_DEVICES": ""} if cpu else None)

    def _stage_chunk(self, job: dict) -> None:
        self._script(job, "assemble_retrieval_chunks.py", "Splitting into passages")

    def _chunks(self, job: dict) -> list[dict]:
        manifest, document = self._manifest_document(job)
        output = resolve_output_dir(resolve_source_path(Path(job["manifest"]), document), document)
        path = output / "retrieval_chunks.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _stage_embed(self, job: dict) -> None:
        pass  # embedding and storing run together in _stage_index (vectors are not kept between steps)

    def _stage_index(self, job: dict) -> None:
        chunks = self._chunks(job)
        if job["replaces"] is None and job["document_id"] in self.index.indexed_counts():
            raise ValueError(f"{job['document_id']} is already indexed")
        job_id = job["id"]

        began, embed_seconds = time.perf_counter(), 0.0

        def embedded(done: int, total: int) -> None:
            # Shown on the "embed" step; this callback runs while the job is on the index stage.
            nonlocal embed_seconds
            embed_seconds = time.perf_counter() - began
            self.store.update(job_id, stage="embed", progress=done / total, detail=f"{done} / {total} passages")

        def stored(done: int, total: int) -> None:
            self.store.update(job_id, stage="index", progress=done / total, detail="Adding to the search index")

        with self.index_lock:
            result = self.index.add_document(chunks, on_embed=embedded, on_store=stored)
        self._log(job_id, f"indexed {json.dumps(result)}\n")
        self.store.update(job_id, stage="index", detail=f"{result['chunks']} passages indexed")
        return {"embed": round(embed_seconds, 1)}
