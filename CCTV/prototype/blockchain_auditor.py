"""
blockchain_auditor.py - IBVAP Blockchain Audit Trail Daemon
============================================================
Standalone microservice that polls the PostgreSQL events table,
hashes each new event with SHA-256, and anchors the hash on a local
Ganache Ethereum node.  Audit receipts are appended to audit_ledger.json.

CRITICAL: This file is 100% standalone.  It does NOT import or modify
          main.py, edge_pipeline.py, or any other existing project file.
"""

import time
import json
import hashlib
import psycopg2
from datetime import datetime
# pyrefly: ignore [missing-import]
from web3 import Web3

# ---------------------------------------------------------------------------
# 1. Ethereum / Web3 initialisation
# ---------------------------------------------------------------------------
GANACHE_URL = "http://127.0.0.1:8545"
w3 = Web3(Web3.HTTPProvider(GANACHE_URL))

try:
    w3.eth.default_account = w3.eth.accounts[0]
    print(f"[blockchain_auditor] Connected to Ganache. Default account: {w3.eth.default_account}")
except Exception as exc:
    print(f"[blockchain_auditor] WARNING - Ganache not reachable: {exc}")
    print("[blockchain_auditor] Blockchain anchoring will be skipped until the node is available.")

# ---------------------------------------------------------------------------
# 2. PostgreSQL connection parameters (must match docker-compose environment)
# ---------------------------------------------------------------------------
DB_PARAMS = {
    "dbname":   "ibvap_db",
    "user":     "ibvap_user",
    "password": "ibvap_secret",
    "host":     "localhost",
    "port":     5433,
}

AUDIT_LEDGER_PATH = "audit_ledger.json"
POLL_INTERVAL_SEC = 3


def get_db_connection():
    """Open a PostgreSQL connection. Returns the connection object or None."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except psycopg2.OperationalError as exc:
        print(f"[blockchain_auditor] DB connection failed: {exc}")
        return None


def hash_event(row_dict: dict) -> str:
    """Return a hex SHA-256 digest of the JSON-serialised event dict."""
    payload = json.dumps(row_dict, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def anchor_on_blockchain(master_hash: str):
    """
    Send a 0-ETH transaction whose data field carries the hash.
    Returns the transaction hash hex string, or None if unavailable.
    """
    if not w3.is_connected():
        return None
    try:
        tx_hash = w3.eth.send_transaction({
            "from":  w3.eth.default_account,
            "to":    w3.eth.default_account,
            "value": 0,
            "data":  Web3.to_bytes(text=master_hash),
        })
        return tx_hash.hex()
    except Exception as exc:
        print(f"[blockchain_auditor] Blockchain tx failed: {exc}")
        return None


def write_ledger_entry(event_id, blockchain_tx, hash_stored: str):
    """Append a single JSON audit record to audit_ledger.json."""
    record = {
        "event_id":      event_id,
        "blockchain_tx": blockchain_tx,
        "hash_stored":   hash_stored,
        "audited_at":    datetime.utcnow().isoformat() + "Z",
    }
    with open(AUDIT_LEDGER_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def main():
    print("[blockchain_auditor] Daemon starting ...")
    last_processed_time = datetime(2000, 1, 1)

    while True:
        conn = get_db_connection()

        if conn is not None:
            cur = None
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT id, camera_id, event_type, confidence, timestamp
                    FROM   events
                    WHERE  timestamp > %s
                    ORDER  BY timestamp ASC
                    """,
                    (last_processed_time,),
                )
                rows = cur.fetchall()

                for row in rows:
                    event_id, camera_id, event_type, confidence, timestamp = row

                    event_dict = {
                        "id":         event_id,
                        "camera_id":  camera_id,
                        "event_type": event_type,
                        "timestamp":  str(timestamp),
                    }

                    master_hash = hash_event(event_dict)
                    blockchain_tx = anchor_on_blockchain(master_hash)
                    write_ledger_entry(event_id, blockchain_tx, master_hash)

                    status = f"tx={blockchain_tx[:16]}..." if blockchain_tx else "NO BLOCKCHAIN TX"
                    print(
                        f"[blockchain_auditor] Audited event {event_id} | "
                        f"hash={master_hash[:12]}... | {status}"
                    )

                    last_processed_time = timestamp

            except Exception as exc:
                print(f"[blockchain_auditor] Error processing events: {exc}")
            finally:
                if cur is not None:
                    cur.close()
                conn.close()
        else:
            print("[blockchain_auditor] Skipping poll cycle - no DB connection.")

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
