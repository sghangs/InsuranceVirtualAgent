from src.longmemory.pg_client import ConnectionManager
import os,sys
from src.exception.exception import InsuranceAgentException
from src.longmemory.embedding_utility import get_embedding
from src.constant import MIN_CONN, MAX_CONN, TOP_K_CONVERSATIONS, SIMILARITY_THRESHOLD
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

async def store_summary(policy_number, user_id, summary, thread_id):
    """Store conversation summary and embedding in the database."""
    embedding = get_embedding(summary)
    manager = ConnectionManager.instance(db_config)
    await manager.open_pool()
    conn = await manager.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO conversations (policy_number, user_id, message, embedding, thread_id) VALUES (%s, %s, %s, %s, %s)",
                (policy_number, user_id, summary, embedding, thread_id)
            )
    except Exception as e:
        # Handle exceptions (e.g., log them)
        raise InsuranceAgentException(sys, e)
    finally:
        # Commit the transaction
        await conn.commit()
        await manager.put_conn(conn)
        
        
# Retrieve similar conversations (summaries or key events)
async def retrieve_similar_conversations(user_id, query, policy_number=None, top_k=TOP_K_CONVERSATIONS, similarity_threshold=0.75):
    """
    Retrieve similar conversations based on the query and policy number,
    using pgvector for similarity search and applying a cosine similarity threshold.
    """
    embedding = get_embedding(query)
    manager = ConnectionManager.instance(db_config)
    await manager.open_pool()
    conn = await manager.get_conn()
    results = []
    try:
        async with conn.cursor() as cur:
            # Use pgvector cosine distance (<=>) for similarity
            await cur.execute(
                """
                SELECT user_id, message, created_at,
                       1 - (embedding <=> %s::vector) AS cosine_similarity
                FROM conversations
                WHERE policy_number = %s AND user_id = %s
                AND 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    embedding,           # For similarity calculation
                    policy_number,       # Filter
                    user_id,             # Filter
                    embedding,           # For threshold
                    similarity_threshold,# Threshold value
                    embedding,           # For ordering
                    top_k                # Limit
                )
            )
            rows = await cur.fetchall()
            results = [
                {"user_id": r[0], "message": r[1], "created_at": r[2], "cosine_similarity": float(r[3])}
                for r in rows
            ]
    except Exception as e:
        raise InsuranceAgentException(sys, e)
    finally:
        await conn.commit()
        await manager.put_conn(conn)

    return results

