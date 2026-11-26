
# ================================================================
# INSTITUTE360 - COMPLETE INTELLIGENT DATA INGESTION PLATFORM
# ONE PYTHON FILE | FIVE KAGGLE DATASETS | EIGHT PROCESSES
# ================================================================

import os
import re
import json
import uuid
import time
import shutil
import hashlib
import zipfile
import sqlite3
import logging

from pathlib import Path
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd
import streamlit as st

from qdrant_client import QdrantClient, models
from streamlit_autorefresh import st_autorefresh


# ================================================================
# CONFIGURATION
# ================================================================

APP_NAME = "Institute360 Intelligent Data Ingestion"

BASE_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = BASE_DIR / "extracted_data"
LOG_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "institute360_manifest.db"

# FIVE KAGGLE ZIP DATASETS - KEEP THESE FILES BESIDE THIS SCRIPT

ZIP_DATASETS = [
    BASE_DIR / "archive (2).zip",
    BASE_DIR / "archive (3).zip",
    BASE_DIR / "archive (4).zip",
    BASE_DIR / "archive (5).zip",
    BASE_DIR / "archive (6).zip"
]

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
VECTOR_SIZE = 768

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "institute360_knowledge"

BATCH_SIZE = 32
MAX_CHUNK_LENGTH = 1800
MONITOR_INTERVAL = 10

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# LOGGING
# ================================================================

logging.basicConfig(
    filename=str(LOG_DIR / "ingestion.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8"
)


def log(message, level="info"):
    print(message)
    getattr(logging, level, logging.info)(message)


# ================================================================
# SQLITE MANIFEST
# ================================================================

def get_db():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_db() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                source_file TEXT PRIMARY KEY,
                dataset TEXT,
                file_hash TEXT,
                record_count INTEGER,
                processed_at TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                point_id TEXT PRIMARY KEY,
                source_file TEXT,
                dataset TEXT,
                row_number INTEGER,
                chunk_number INTEGER,
                content_hash TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS archives (
                archive_path TEXT PRIMARY KEY,
                archive_hash TEXT,
                extracted_at TEXT
            )
        """)


def calculate_hash(file_path):
    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def hash_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def get_previous_file_hash(file_path):
    with get_db() as conn:
        result = conn.execute(
            "SELECT file_hash FROM files WHERE source_file=?",
            (str(file_path.resolve()),)
        ).fetchone()

    return result[0] if result else None


# ================================================================
# DATASET IDENTIFICATION
# ================================================================

DATASET_MAPPING = {
    "wa_fn-usec_-hr-employee-attrition.csv":
        "IBM_HR_Employee_Attrition",

    "hrdataset_v14.csv":
        "HR_Dataset_V14",

    "aug_train.csv":
        "Job_Change_Training",

    "aug_test.csv":
        "Job_Change_Test",

    "sample_submission.csv":
        "Job_Change_Sample_Submission",

    "ds_salaries.csv":
        "Data_Science_Salaries",

    "ai-job-skills-index-2026-11.csv":
        "AI_Job_Skills"
}


def identify_dataset(file_path):
    name = file_path.name.lower()

    if name in DATASET_MAPPING:
        return DATASET_MAPPING[name]

    return re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        file_path.stem
    )


# ================================================================
# PROCESS 1: ZIP EXTRACTION AND AUTOMATIC FILE DISCOVERY
# ================================================================

def extract_all_archives():
    extracted_count = 0

    for archive_path in ZIP_DATASETS:

        if not archive_path.exists():
            log(
                f"Dataset ZIP not found: {archive_path.name}",
                "warning"
            )
            continue

        try:
            archive_hash = calculate_hash(archive_path)
            target = EXTRACT_DIR / archive_path.stem

            with get_db() as conn:
                previous = conn.execute(
                    "SELECT archive_hash FROM archives "
                    "WHERE archive_path=?",
                    (str(archive_path.resolve()),)
                ).fetchone()

            if (
                previous
                and previous[0] == archive_hash
                and target.exists()
            ):
                continue

            log(f"Extracting {archive_path.name}")

            staging = EXTRACT_DIR / (
                archive_path.stem + "_staging"
            )

            if staging.exists():
                shutil.rmtree(staging)

            staging.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(
                archive_path, "r"
            ) as archive:

                for member in archive.infolist():

                    if member.is_dir():
                        continue

                    destination = (
                        staging / member.filename
                    ).resolve()

                    # Prevent ZIP path traversal.
                    if not destination.is_relative_to(
                        staging.resolve()
                    ):
                        log(
                            f"Unsafe ZIP entry skipped: "
                            f"{member.filename}",
                            "warning"
                        )
                        continue

                    if destination.suffix.lower() not in [
                        ".csv", ".json", ".txt"
                    ]:
                        continue

                    destination.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    with archive.open(member) as source:
                        destination.write_bytes(source.read())

            if target.exists():
                shutil.rmtree(target)

            staging.rename(target)

            with get_db() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO archives
                    (archive_path, archive_hash, extracted_at)
                    VALUES (?, ?, ?)
                """, (
                    str(archive_path.resolve()),
                    archive_hash,
                    datetime.now(timezone.utc).isoformat()
                ))

            extracted_count += 1
            log(f"Successfully extracted {archive_path.name}")

        except Exception as error:
            log(
                f"Archive extraction failed for "
                f"{archive_path.name}: {error}",
                "error"
            )

    return extracted_count


def discover_dataset_files():
    discovered = []

    for extension in ["*.csv", "*.json", "*.txt"]:
        discovered.extend(
            EXTRACT_DIR.rglob(extension)
        )

        discovered.extend(
            BASE_DIR.glob(extension)
        )

    return sorted(set(discovered))


# ================================================================
# PROCESS 2: DATA EXTRACTION
# ================================================================

def extract_data(file_path):
    try:
        extension = file_path.suffix.lower()

        if extension == ".csv":
            try:
                return pd.read_csv(
                    file_path,
                    encoding="utf-8-sig",
                    low_memory=False
                )
            except UnicodeDecodeError:
                return pd.read_csv(
                    file_path,
                    encoding="latin1",
                    low_memory=False
                )

        if extension == ".json":
            try:
                return pd.read_json(file_path)
            except ValueError:
                return pd.read_json(
                    file_path,
                    lines=True
                )

        if extension == ".txt":
            content = file_path.read_text(
                encoding="utf-8",
                errors="replace"
            )

            return pd.DataFrame({
                "content": [content]
            })

    except Exception as error:
        log(
            f"Data extraction error {file_path.name}: {error}",
            "error"
        )

    return pd.DataFrame()


# ================================================================
# PROCESS 3: DATA CLEANING AND PREPROCESSING
# ================================================================

SENSITIVE_COLUMNS = {
    "employeename",
    "empid",
    "employeenumber",
    "enrolleeid",
    "dob",
    "dateofbirth",
    "zip",
    "zipcode",
    "address",
    "phone",
    "email",
    "ssn",
    "nationalid",
    "managername",
    "managerid",
    "gender",
    "genderid",
    "sex",
    "racedesc",
    "hispaniclatino",
    "citizendesc",
    "maritaldesc"
}


def normalize_column_name(column):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(column).lower()
    )


def clean_data(df, dataset_name):
    if df.empty:
        return df

    df = df.copy()

    df = df.loc[:, ~df.columns.duplicated()]

    df.columns = [
        str(column).strip().replace("\ufeff", "")
        for column in df.columns
    ]

    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.drop_duplicates(inplace=True)

    for column in df.select_dtypes(
        include=["object", "string"]
    ).columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace({
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "NULL": pd.NA
            })
        )

    # Remove direct identifiers and sensitive demographic
    # attributes from HR embedding text.
    if "HR" in dataset_name:

        columns_to_remove = [
            column for column in df.columns
            if normalize_column_name(column)
            in SENSITIVE_COLUMNS
        ]

        df.drop(
            columns=columns_to_remove,
            inplace=True,
            errors="ignore"
        )

    # Remove recruitment candidate identifiers.
    if dataset_name in [
        "Job_Change_Training",
        "Job_Change_Test",
        "Job_Change_Sample_Submission"
    ]:
        df.drop(
            columns=["enrollee_id"],
            inplace=True,
            errors="ignore"
        )

    return df


# ================================================================
# PROCESS 4: INTELLIGENT CHUNKING
# ================================================================

def convert_record_to_text(row):
    fields = []

    for column, value in row.items():

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            fields.append(f"{column}: {value}")

    return " | ".join(fields)


def intelligent_chunking(df, dataset_name):
    chunks = []

    for row_number, (_, row) in enumerate(df.iterrows()):

        text = convert_record_to_text(row)

        if not text.strip():
            continue

        pieces = [
            text[index:index + MAX_CHUNK_LENGTH]
            for index in range(
                0, len(text), MAX_CHUNK_LENGTH
            )
        ]

        for chunk_number, piece in enumerate(pieces):
            chunks.append({
                "dataset": dataset_name,
                "row_number": row_number,
                "chunk_number": chunk_number,
                "text": piece
            })

    return chunks


# ================================================================
# PROCESS 5: METADATA GENERATION
# ================================================================

def generate_metadata(chunk, file_path):
    return {
        "dataset": chunk["dataset"],
        "source_file": str(file_path.resolve()),
        "file_name": file_path.name,
        "row_number": chunk["row_number"],
        "chunk_number": chunk["chunk_number"],
        "content": chunk["text"],
        "content_hash": hash_text(chunk["text"]),
        "file_type": file_path.suffix.lower(),
        "ingestion_timestamp":
            datetime.now(timezone.utc).isoformat()
    }


def generate_point_id(file_path, row_number, chunk_number):
    unique_key = (
        f"{file_path.resolve()}|"
        f"{row_number}|{chunk_number}"
    )

    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, unique_key)
    )


# ================================================================
# PROCESS 6: OLLAMA EMBEDDING GENERATION
# ================================================================

def check_ollama():
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=5
        )
        response.raise_for_status()

        available = [
            model["name"].split(":")[0]
            for model in response.json().get("models", [])
        ]

        return (
            EMBEDDING_MODEL.split(":")[0]
            in available
        )

    except Exception:
        return False


def generate_embeddings(texts):
    if not texts:
        return []

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": texts
        },
        timeout=300
    )

    response.raise_for_status()

    embeddings = response.json().get("embeddings", [])

    if len(embeddings) != len(texts):
        raise ValueError(
            "Ollama returned an incorrect number of embeddings."
        )

    for vector in embeddings:

        if len(vector) != VECTOR_SIZE:
            raise ValueError(
                f"Expected {VECTOR_SIZE} dimensions, "
                f"received {len(vector)}"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "Invalid values found in embedding."
            )

    return embeddings


# ================================================================
# PROCESS 7: QDRANT VECTOR STORAGE
# ================================================================

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        timeout=60
    )


def initialize_qdrant():
    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME):

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

        log("Qdrant collection created.")

    else:
        info = client.get_collection(COLLECTION_NAME)
        existing_size = info.config.params.vectors.size

        if existing_size != VECTOR_SIZE:
            raise ValueError(
                f"Qdrant vector size is {existing_size}, "
                f"but expected {VECTOR_SIZE}. "
                "Use a separate collection for a different model."
            )

    return client


def store_vectors(client, points):
    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )


def delete_old_points(client, point_ids):
    if not point_ids:
        return

    for start in range(0, len(point_ids), 500):
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(
                points=point_ids[start:start + 500]
            ),
            wait=True
        )


def get_existing_file_points(file_path):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT point_id FROM records WHERE source_file=?",
            (str(file_path.resolve()),)
        ).fetchall()

    return [row[0] for row in rows]


def save_file_results(
    file_path,
    dataset_name,
    file_hash,
    count,
    records
):
    with get_db() as conn:

        conn.execute("""
            INSERT OR REPLACE INTO files
            (source_file, dataset, file_hash,
             record_count, processed_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(file_path.resolve()),
            dataset_name,
            file_hash,
            count,
            datetime.now(timezone.utc).isoformat()
        ))

        conn.execute(
            "DELETE FROM records WHERE source_file=?",
            (str(file_path.resolve()),)
        )

        conn.executemany("""
            INSERT OR REPLACE INTO records
            (point_id, source_file, dataset,
             row_number, chunk_number, content_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """, records)


# ================================================================
# INGESTION OF ONE FILE
# ================================================================

def ingest_single_file(file_path, client, force=False):
    dataset_name = identify_dataset(file_path)
    file_hash = calculate_hash(file_path)

    previous_hash = get_previous_file_hash(file_path)

    if previous_hash == file_hash and not force:
        return {
            "dataset": dataset_name,
            "file": file_path.name,
            "status": "Unchanged",
            "vectors": 0
        }

    log(f"Starting ingestion: {file_path.name}")

    # PROCESS 2: Extraction
    df = extract_data(file_path)

    if df.empty:
        return {
            "dataset": dataset_name,
            "file": file_path.name,
            "status": "Empty or failed",
            "vectors": 0
        }

    # PROCESS 3: Cleaning
    df = clean_data(df, dataset_name)

    # PROCESS 4: Chunking
    chunks = intelligent_chunking(df, dataset_name)

    if not chunks:
        return {
            "dataset": dataset_name,
            "file": file_path.name,
            "status": "No valid records",
            "vectors": 0
        }

    new_points = []
    new_manifest = []

    # PROCESS 6: Generate embeddings in batches.
    for start in range(0, len(chunks), BATCH_SIZE):

        batch = chunks[start:start + BATCH_SIZE]

        texts = [
            chunk["text"] for chunk in batch
        ]

        embeddings = generate_embeddings(texts)

        for chunk, vector in zip(batch, embeddings):

            # PROCESS 5: Metadata generation.
            metadata = generate_metadata(
                chunk, file_path
            )

            point_id = generate_point_id(
                file_path,
                chunk["row_number"],
                chunk["chunk_number"]
            )

            new_points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=metadata
                )
            )

            new_manifest.append((
                point_id,
                str(file_path.resolve()),
                dataset_name,
                chunk["row_number"],
                chunk["chunk_number"],
                metadata["content_hash"]
            ))

    # PROCESS 7: Store new vectors before deleting old ones.
    for start in range(0, len(new_points), 500):
        store_vectors(
            client,
            new_points[start:start + 500]
        )

    old_ids = set(
        get_existing_file_points(file_path)
    )

    new_ids = {
        record[0] for record in new_manifest
    }

    obsolete_ids = list(old_ids - new_ids)

    delete_old_points(client, obsolete_ids)

    save_file_results(
        file_path,
        dataset_name,
        file_hash,
        len(new_points),
        new_manifest
    )

    log(
        f"Completed {file_path.name}: "
        f"{len(new_points)} vectors"
    )

    return {
        "dataset": dataset_name,
        "file": file_path.name,
        "status": "Completed",
        "vectors": len(new_points)
    }


# ================================================================
# PROCESS 8: DELETED-FILE SYNCHRONIZATION
# ================================================================

def remove_missing_files(client, discovered_files):
    current_files = {
        str(file.resolve())
        for file in discovered_files
    }

    with get_db() as conn:
        tracked = conn.execute(
            "SELECT source_file FROM files"
        ).fetchall()

    removed = []

    for (source_file,) in tracked:

        if source_file not in current_files:

            log(
                f"Synchronizing deleted file: {source_file}"
            )

            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_file",
                                match=models.MatchValue(
                                    value=source_file
                                )
                            )
                        ]
                    )
                ),
                wait=True
            )

            with get_db() as conn:
                conn.execute(
                    "DELETE FROM records WHERE source_file=?",
                    (source_file,)
                )

                conn.execute(
                    "DELETE FROM files WHERE source_file=?",
                    (source_file,)
                )

            removed.append(source_file)

    return removed


# ================================================================
# COMPLETE INGESTION WORKFLOW
# ================================================================

def run_ingestion(force=False):
    initialize_database()
    client = initialize_qdrant()

    # PROCESS 1
    extract_all_archives()

    files = discover_dataset_files()

    if not files:
        return []

    results = []

    progress = st.progress(0)
    status = st.empty()

    for index, file_path in enumerate(files):

        status.info(
            f"Processing {index + 1}/{len(files)}: "
            f"{file_path.name}"
        )

        try:
            result = ingest_single_file(
                file_path,
                client,
                force
            )

            results.append(result)

        except Exception as error:
            log(
                f"Processing failed for "
                f"{file_path.name}: {error}",
                "error"
            )

            results.append({
                "dataset": identify_dataset(file_path),
                "file": file_path.name,
                "status": f"Failed: {error}",
                "vectors": 0
            })

        progress.progress(
            (index + 1) / len(files)
        )

    removed = remove_missing_files(
        client,
        files
    )

    if removed:
        log(
            f"Removed {len(removed)} deleted files."
        )

    status.success("Ingestion cycle completed.")

    return results


# ================================================================
# REAL-TIME MONITORING
# ================================================================

def monitor_changes():
    client = initialize_qdrant()

    extract_all_archives()

    files = discover_dataset_files()

    results = []

    for file_path in files:

        try:
            current_hash = calculate_hash(file_path)
            previous_hash = get_previous_file_hash(file_path)

            if current_hash != previous_hash:

                result = ingest_single_file(
                    file_path,
                    client
                )

                results.append(result)

        except Exception as error:
            log(
                f"Monitoring error: {error}",
                "error"
            )

    removed = remove_missing_files(
        client,
        files
    )

    return results, removed


# ================================================================
# SEMANTIC SEARCH
# ================================================================

def semantic_search(query, limit=5, selected_dataset="All"):
    client = get_qdrant_client()

    vector = generate_embeddings([query])[0]

    query_filter = None

    if selected_dataset != "All":
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="dataset",
                    match=models.MatchValue(
                        value=selected_dataset
                    )
                )
            ]
        )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True
    )

    results = []

    for point in response.points:

        results.append({
            "score": round(point.score, 4),
            "dataset": point.payload.get("dataset"),
            "file": point.payload.get("file_name"),
            "row": point.payload.get("row_number"),
            "content": point.payload.get("content")
        })

    return results


# ================================================================
# DASHBOARD STATISTICS
# ================================================================

def dashboard_statistics():
    client = get_qdrant_client()

    info = client.get_collection(COLLECTION_NAME)

    with get_db() as conn:
        records = pd.read_sql_query(
            "SELECT * FROM records",
            conn
        )

        files = pd.read_sql_query(
            "SELECT * FROM files",
            conn
        )

    return {
        "vectors": info.points_count or 0,
        "files": len(files),
        "records": len(records),
        "datasets": (
            records["dataset"].nunique()
            if not records.empty else 0
        ),
        "records_df": records,
        "files_df": files
    }


# ================================================================
# DATASET DELETION
# ================================================================

def delete_dataset(dataset_name):
    client = get_qdrant_client()

    with get_db() as conn:
        source_files = conn.execute(
            "SELECT source_file FROM files WHERE dataset=?",
            (dataset_name,)
        ).fetchall()

    for (source_file,) in source_files:

        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchValue(
                                value=source_file
                            )
                        )
                    ]
                )
            ),
            wait=True
        )

    with get_db() as conn:
        conn.execute(
            "DELETE FROM records WHERE dataset=?",
            (dataset_name,)
        )

        conn.execute(
            "DELETE FROM files WHERE dataset=?",
            (dataset_name,)
        )

    log(f"Deleted dataset vectors: {dataset_name}")


# ================================================================
# STREAMLIT APPLICATION
# ================================================================

def main():

    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🎓",
        layout="wide"
    )

    initialize_database()

    st.title(
        "🎓 Institute360 Intelligent Data Ingestion"
    )

    st.caption(
        "Five Kaggle datasets | Eight ingestion processes | "
        "Ollama | Qdrant | Streamlit"
    )

    # CONNECTION STATUS
    col1, col2, col3 = st.columns(3)

    ollama_connected = check_ollama()

    col1.metric(
        "Ollama",
        "Connected" if ollama_connected else "Disconnected"
    )

    try:
        client = initialize_qdrant()
        qdrant_connected = True

    except Exception as error:
        client = None
        qdrant_connected = False
        st.error(f"Qdrant connection error: {error}")

    col2.metric(
        "Qdrant",
        "Connected" if qdrant_connected else "Disconnected"
    )

    col3.metric(
        "Embedding dimensions",
        VECTOR_SIZE
    )

    if not ollama_connected:
        st.error(
            "Ollama is unavailable or the embedding model "
            "is missing. Run: ollama pull nomic-embed-text"
        )

    if not qdrant_connected:
        st.warning(
            "Start your existing Qdrant Docker container."
        )
        return

    # NAVIGATION
    st.sidebar.title("Institute360 Navigation")

    page = st.sidebar.radio(
        "Select module",
        [
            "Dashboard",
            "Data Ingestion",
            "Semantic Search",
            "Dataset Management",
            "Real-Time Monitoring",
            "System Logs"
        ]
    )

    st.sidebar.divider()
    st.sidebar.caption("Institute360 Data Platform")

    # ============================================================
    # DASHBOARD
    # ============================================================

    if page == "Dashboard":

        st.header("Platform Overview")

        try:
            stats = dashboard_statistics()

            a, b, c, d = st.columns(4)

            a.metric("Total Vectors", stats["vectors"])
            b.metric("Processed Files", stats["files"])
            c.metric("Datasets", stats["datasets"])
            d.metric("Indexed Records", stats["records"])

            st.subheader("Dataset Distribution")

            if not stats["records_df"].empty:

                distribution = (
                    stats["records_df"]
                    .groupby("dataset")
                    .size()
                    .reset_index(name="Records")
                )

                st.bar_chart(
                    distribution.set_index("dataset")
                )

                st.dataframe(
                    distribution,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.info(
                    "No records indexed. Start ingestion."
                )

            st.subheader("Processed Files")

            st.dataframe(
                stats["files_df"],
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:
            st.warning(str(error))

    # ============================================================
    # DATA INGESTION
    # ============================================================

    elif page == "Data Ingestion":

        st.header("Complete Data Ingestion")

        st.write(
            "All five Kaggle ZIP archives are configured below."
        )

        st.subheader("Configured datasets")

        missing = []

        for archive in ZIP_DATASETS:

            exists = archive.exists()

            st.write(
                ("🟢 " if exists else "🔴 ")
                + archive.name
            )

            if not exists:
                missing.append(archive.name)

        if missing:
            st.error(
                "Missing ZIP files: " + ", ".join(missing)
            )

        else:
            st.success("All five ZIP files are available.")

        st.divider()

        force = st.checkbox(
            "Force reprocess all files",
            value=False
        )

        if st.button(
            "🚀 Start Complete Ingestion",
            type="primary",
            use_container_width=True
        ):

            if not ollama_connected:
                st.error(
                    "Connect Ollama and ensure the embedding "
                    "model is available before ingestion."
                )

            elif missing:
                st.error(
                    "Please place all five ZIP files beside "
                    "universal_ingestion.py."
                )

            else:
                with st.spinner(
                    "Running all eight ingestion processes..."
                ):

                    try:
                        results = run_ingestion(force=force)

                        if results:
                            st.success(
                                "Ingestion cycle finished."
                            )

                            st.dataframe(
                                pd.DataFrame(results),
                                use_container_width=True,
                                hide_index=True
                            )

                            completed = sum(
                                1 for item in results
                                if item["status"] == "Completed"
                            )

                            failed = sum(
                                1 for item in results
                                if item["status"].startswith("Failed")
                            )

                            st.write(
                                f"Completed files: {completed}"
                            )

                            st.write(
                                f"Failed files: {failed}"
                            )

                        else:
                            st.warning(
                                "No supported data files were discovered."
                            )

                        stats = dashboard_statistics()

                        st.metric(
                            "Current Qdrant vectors",
                            stats["vectors"]
                        )

                    except Exception as error:
                        st.error(str(error))

    # ============================================================
    # SEMANTIC SEARCH
    # ============================================================

    elif page == "Semantic Search":

        st.header("Semantic Search")

        st.write(
            "Search records across the five datasets "
            "using natural-language questions."
        )

        query = st.text_input(
            "Enter your question",
            placeholder=(
                "Example: What skills are required "
                "for machine learning jobs?"
            )
        )

        try:
            stats = dashboard_statistics()

            options = ["All"]

            if not stats["records_df"].empty:
                options += sorted(
                    stats["records_df"]["dataset"]
                    .dropna()
                    .unique()
                    .tolist()
                )

        except Exception:
            options = ["All"]

        dataset_filter = st.selectbox(
            "Dataset filter",
            options
        )

        result_limit = st.slider(
            "Number of search results",
            1, 20, 5
        )

        if st.button("🔍 Search", type="primary"):

            if not query.strip():
                st.warning("Please enter a search question.")

            else:
                try:
                    with st.spinner(
                        "Generating query embedding..."
                    ):
                        results = semantic_search(
                            query,
                            result_limit,
                            dataset_filter
                        )

                    if not results:
                        st.info(
                            "No relevant records found. "
                            "Check ingestion and Qdrant vector count."
                        )

                    for index, result in enumerate(results):

                        with st.expander(
                            f"Result {index + 1} | "
                            f"Similarity: {result['score']}"
                        ):

                            st.write(
                                "**Dataset:** "
                                + str(result["dataset"])
                            )

                            st.write(
                                "**Source:** "
                                + str(result["file"])
                            )

                            st.write(
                                "**Row:** "
                                + str(result["row"])
                            )

                            st.write(result["content"])

                except Exception as error:
                    st.error(f"Search error: {error}")

    # ============================================================
    # DATASET MANAGEMENT
    # ============================================================

    elif page == "Dataset Management":

        st.header("Dataset Management")

        try:
            stats = dashboard_statistics()
            records = stats["records_df"]

            if records.empty:
                st.info("No indexed datasets available.")

            else:
                datasets = sorted(
                    records["dataset"]
                    .dropna()
                    .unique()
                    .tolist()
                )

                for dataset in datasets:

                    count = len(
                        records[
                            records["dataset"] == dataset
                        ]
                    )

                    with st.expander(
                        f"{dataset} | {count} records"
                    ):

                        st.write(
                            f"Indexed records: {count}"
                        )

                        st.warning(
                            "Deleting vectors does not delete the "
                            "original ZIP or extracted source files. "
                            "A later ingestion can restore them."
                        )

                        confirm = st.checkbox(
                            f"Confirm deletion of {dataset}",
                            key=f"confirm_{dataset}"
                        )

                        if st.button(
                            f"Delete {dataset}",
                            key=f"delete_{dataset}",
                            disabled=not confirm
                        ):

                            delete_dataset(dataset)

                            st.success(
                                "Dataset vectors deleted."
                            )

                            st.rerun()

        except Exception as error:
            st.error(str(error))

    # ============================================================
    # REAL-TIME MONITORING
    # ============================================================

    elif page == "Real-Time Monitoring":

        st.header("Real-Time Dataset Monitoring")

        st.write(
            "Monitor the five ZIP archives and extracted "
            "files for new, modified or deleted records."
        )

        st.info(
            f"Monitoring interval: {MONITOR_INTERVAL} seconds"
        )

        auto_monitor = st.toggle(
            "Enable automatic monitoring",
            value=False
        )

        if auto_monitor:

            st_autorefresh(
                interval=MONITOR_INTERVAL * 1000,
                key="institute360_monitor"
            )

            try:
                updates, removed = monitor_changes()

                if updates:
                    st.dataframe(
                        pd.DataFrame(updates),
                        use_container_width=True,
                        hide_index=True
                    )

                else:
                    st.success(
                        "Monitoring active. No file changes detected."
                    )

                if removed:
                    st.warning(
                        f"{len(removed)} deleted files synchronized."
                    )

            except Exception as error:
                st.error(f"Monitoring failed: {error}")

        else:

            st.info(
                "Automatic monitoring is currently paused."
            )

            if st.button("Check for changes now"):

                try:
                    updates, removed = monitor_changes()

                    st.success("Monitoring check completed.")

                    if updates:
                        st.dataframe(
                            pd.DataFrame(updates),
                            use_container_width=True,
                            hide_index=True
                        )

                    st.write(
                        f"Deleted files synchronized: {len(removed)}"
                    )

                except Exception as error:
                    st.error(str(error))

    # ============================================================
    # SYSTEM LOGS
    # ============================================================

    elif page == "System Logs":

        st.header("System Activity Logs")

        log_file = LOG_DIR / "ingestion.log"

        if log_file.exists():

            content = log_file.read_text(
                encoding="utf-8",
                errors="replace"
            )

            st.text_area(
                "Latest activity",
                content[-20000:],
                height=450
            )

            if st.button("Refresh logs"):
                st.rerun()

        else:
            st.info("No logs generated yet.")

    # FOOTER

    st.divider()

    st.caption(
        "Institute360 | Python + Ollama + Qdrant + Streamlit"
    )


# ================================================================
# APPLICATION START
# ================================================================

if __name__ == "__main__":
    main()
