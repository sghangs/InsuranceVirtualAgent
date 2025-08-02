from src.longmemory.pg_client import ConnectionManager
import os
from src.longmemory.embedding_utility import get_embedding
from src.constant import MIN_CONN, MAX_CONN
from psycopg import sql
from dotenv import load_dotenv
load_dotenv()

# Load environment variables for database connection
db_config = {
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
    "host": os.getenv("PG_HOST"),
    "database": os.getenv("PG_DB"),
    "port": os.getenv("PG_PORT", "5432"),  # Default to 5432 if not set
    "minconn": MIN_CONN,
    "maxconn": MAX_CONN
}

async def get_user_profile(user_id):
    """Retrieve user profile information from the database."""
    manager = ConnectionManager.instance(db_config)
    await manager.open_pool()
    conn = await manager.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT name, policy_number, policy_type, dob, phone, email, address FROM user_profiles WHERE user_id=%s", (user_id,))
            row = await cur.fetchone()
    finally:
        # Commit the transaction
        await conn.commit()
        await manager.put_conn(conn)

    if not row:
        return {"name": "", "policy_number": "", "policy_type": "", "dob": "", "phone": "", "email": "", "address": ""}

    return {
        "name": row[0],
        "policy_number": row[1],
        "policy_type": row[2],
        "dob": row[3],
        "phone": row[4],
        "email": row[5],
        "address": row[6]
    }

async def upsert_user_profile(user_id, profile_updates: dict):
    # Always include user_id
    fields = ["user_id"] + list(profile_updates.keys())
    values = [user_id] + list(profile_updates.values())

    # Build the columns and placeholders
    columns = sql.SQL(', ').join(map(sql.Identifier, fields))
    placeholders = sql.SQL(', ').join(sql.Placeholder() * len(fields))

    # Build the update assignments, skipping user_id
    update_assignments = sql.SQL(', ').join([
        sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(f), sql.Identifier(f))
        for f in profile_updates.keys()
    ] + [sql.SQL("updated_at = NOW()")])

    query = sql.SQL("""
        INSERT INTO user_profiles ({columns}, created_at, updated_at)
        VALUES ({placeholders}, NOW(), NOW())
        ON CONFLICT (user_id) DO UPDATE SET
            {update_assignments};
    """).format(
        columns=columns,
        placeholders=placeholders,
        update_assignments=update_assignments
    )

    
    manager = ConnectionManager.instance(db_config)
    await manager.open_pool()
    conn = await manager.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(query, values)
    finally:
        await conn.commit()
        await manager.put_conn(conn)