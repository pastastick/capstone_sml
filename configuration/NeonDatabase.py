import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
import datetime
from PIL import Image
import io

# Load .env file
load_dotenv()

# Initialize connection pool
connection_pool = None

def init_db_pool():
    global connection_pool
    if connection_pool is None:
        connection_string = os.getenv('DATABASE_URL')
        connection_pool = pool.SimpleConnectionPool(
            2, 20, connection_string
        )
        print("Connection pool created successfully")

def get_connection():
    global connection_pool
    if connection_pool is None:
        init_db_pool()
    return connection_pool.getconn()

def release_connection(conn):
    global connection_pool
    connection_pool.putconn(conn)

def save_image_record(image_name, category, image_bytes, score, classification):
    """Save image and detection results to database"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        timestamp = datetime.datetime.now()
        
        # Convert numpy float to native Python float
        if hasattr(score, 'item'):  # For numpy floats
            score = float(score.item())
        else:
            score = float(score)
        
        # Convert image to optimized format
        img = Image.open(io.BytesIO(image_bytes))
        img_byte_arr = io.BytesIO()
        
        # Determine format based on original image
        format = 'JPEG' if img.mode == 'RGB' else 'PNG'
        img.save(img_byte_arr, format=format, quality=85)
        optimized_bytes = img_byte_arr.getvalue()
        
        # Perhatikan: sekarang ada 6 parameter (%s)
        cur.execute(
            "INSERT INTO images (name, category, image_data, score, classification, timestamp) VALUES (%s, %s, %s, %s, %s, %s)",
            (image_name, category, psycopg2.Binary(optimized_bytes), score, classification, timestamp)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()
        release_connection(conn)

def get_unique_categories():
    """Get list of unique categories from database"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT DISTINCT category FROM images")
        return [row[0] for row in cur.fetchall()]
    finally:
        cur.close()
        release_connection(conn)
        
def get_images_by_category_and_classification(category, classification, limit=5):
    """Get images by category and classification type"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT name, image_data, score FROM images "
            "WHERE category = %s AND classification = %s "
            "ORDER BY timestamp DESC LIMIT %s",
            (category, classification, limit))
        return cur.fetchall()
    finally:
        cur.close()
        release_connection(conn)

def create_table_if_not_exists():
    """Create images table if not exists"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                category VARCHAR(50) NOT NULL,  -- KOLOM BARU
                image_data BYTEA NOT NULL,
                score FLOAT NOT NULL,
                classification VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP NOT NULL
            )
        """)
        conn.commit()
        
        # Tambahkan kolom jika tabel sudah ada (untuk migrasi)
        try:
            cur.execute("ALTER TABLE images ADD COLUMN IF NOT EXISTS category VARCHAR(50)")
            conn.commit()
        except Exception as alter_error:
            print(f"Warning: {alter_error}")
            
    finally:
        cur.close()
        release_connection(conn)

# Create table on startup
create_table_if_not_exists()