# tasks.py
import os
import json
import sqlite3
import subprocess
import base64
from pathlib import Path
from datetime import datetime
from llm import call_llm
from glob import glob
import shutil
import re

# For local testing, you might set DATA_DIR = Path("./data")
DATA_DIR = Path("/data")

def ensure_in_data(file_path: Path):
    """Ensure file_path is under DATA_DIR."""
    if not file_path.resolve().as_posix().startswith(DATA_DIR.resolve().as_posix()):
        raise ValueError("Access denied. File must be under /data.")

#####################
# Phase A: Operations Tasks (A1–A10)
#####################

### A1. Install uv (if needed) and run datagen.py with an email argument.
def task_a1_run_datagen(user_email: str):
    if shutil.which("uv") is None:
        subprocess.run(["pip", "install", "uv"], check=True)
    datagen_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    datagen_script_path = DATA_DIR / "datagen.py"
    ensure_in_data(datagen_script_path)
    import requests
    response = requests.get(datagen_url)
    if response.status_code != 200:
        raise Exception("Failed to download datagen.py")
    datagen_script_path.write_text(response.text)
    subprocess.run(["python", str(datagen_script_path), user_email], check=True)
    return f"Ran datagen.py with argument {user_email}"

### A2. Format /data/format.md using prettier@3.4.2.
def task_a2_format_markdown():
    file_path = DATA_DIR / "format.md"
    ensure_in_data(file_path)
    if not file_path.exists():
        raise ValueError(f"{file_path} does not exist.")
    try:
        subprocess.run(["prettier", "--write", str(file_path)], check=True)
        return f"Formatted {file_path} using prettier@3.4.2"
    except subprocess.CalledProcessError as e:
        raise Exception("Error running prettier: " + str(e))

### A3. Count the number of Wednesdays in /data/dates.txt.
def task_a3_count_wednesdays():
    input_file = DATA_DIR / "dates.txt"
    output_file = DATA_DIR / "dates-wednesdays.txt"
    ensure_in_data(input_file)
    ensure_in_data(output_file)
    if not input_file.exists():
        raise ValueError(f"{input_file} does not exist.")
    
    count = 0
    for line in input_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            date_obj = datetime.strptime(line, "%Y-%m-%d")
            if date_obj.weekday() == 2:
                count += 1
        except Exception:
            raise ValueError(f"Invalid date format in line: {line}")
    
    output_file.write_text(str(count))
    return f"Counted Wednesdays: {count} written to {output_file}"

### A4. Sort contacts in /data/contacts.json by last_name then first_name.
def task_a4_sort_contacts():
    input_file = DATA_DIR / "contacts.json"
    output_file = DATA_DIR / "contacts-sorted.json"
    ensure_in_data(input_file)
    ensure_in_data(output_file)
    if not input_file.exists():
        raise ValueError(f"{input_file} does not exist.")
    contacts = json.loads(input_file.read_text())
    sorted_contacts = sorted(contacts, key=lambda x: (x.get("last_name", ""), x.get("first_name", "")))
    output_file.write_text(json.dumps(sorted_contacts, indent=2))
    return f"Sorted contacts written to {output_file}"

### A5. Write the first line of the 10 most recent .log files in /data/logs/.
def task_a5_recent_logs():
    logs_dir = DATA_DIR / "logs"
    output_file = DATA_DIR / "logs-recent.txt"
    ensure_in_data(logs_dir)
    ensure_in_data(output_file)
    if not logs_dir.exists():
        raise ValueError(f"{logs_dir} does not exist.")
    log_files = list(logs_dir.glob("*.log"))
    if not log_files:
        raise ValueError("No .log files found in /data/logs/")
    log_files_sorted = sorted(log_files, key=lambda f: f.stat().st_mtime, reverse=True)[:10]
    lines = []
    for file in log_files_sorted:
        with file.open("r") as f:
            first_line = f.readline().strip()
            lines.append(first_line)
    output_file.write_text("\n".join(lines))
    return f"Wrote first lines of 10 most recent log files to {output_file}"

### A6. Index Markdown files in /data/docs/ by extracting the first H1 from each.
def task_a6_index_docs():
    docs_dir = DATA_DIR / "docs"
    output_file = docs_dir / "index.json"
    ensure_in_data(docs_dir)
    ensure_in_data(output_file)
    if not docs_dir.exists():
        raise ValueError(f"{docs_dir} does not exist.")
    index = {}
    for filepath in docs_dir.rglob("*.md"):
        rel_path = filepath.relative_to(docs_dir).as_posix()
        title = None
        with filepath.open("r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    title = line.lstrip("#").strip()
                    break
        if title:
            index[rel_path] = title
    output_file.write_text(json.dumps(index, indent=2))
    return f"Created docs index file at {output_file}"

### A7. Extract sender’s email from /data/email.txt using LLM.
def task_a7_extract_email_sender():
    input_file = DATA_DIR / "email.txt"
    output_file = DATA_DIR / "email-sender.txt"
    ensure_in_data(input_file)
    ensure_in_data(output_file)
    if not input_file.exists():
        raise ValueError(f"{input_file} does not exist.")
    content = input_file.read_text()
    prompt = f"Extract the sender’s email address from the following email message:\n\n{content}"
    email_address = call_llm(prompt).strip()
    output_file.write_text(email_address)
    return f"Extracted sender email: {email_address} written to {output_file}"

### A8. Extract a credit card number from /data/credit-card.png using LLM.
def task_a8_extract_credit_card():
    input_file = DATA_DIR / "credit-card.png"
    output_file = DATA_DIR / "credit-card.txt"
    ensure_in_data(input_file)
    ensure_in_data(output_file)
    if not input_file.exists():
        raise ValueError(f"{input_file} does not exist.")
    with input_file.open("rb") as f:
        img_data = f.read()
    img_b64 = base64.b64encode(img_data).decode("utf-8")
    prompt = f"Extract the credit card number from the following image (base64 encoded). Provide the number without any spaces:\n\n{img_b64}"
    card_number = call_llm(prompt).strip().replace(" ", "")
    output_file.write_text(card_number)
    return f"Extracted credit card number written to {output_file}"

### A9. Find the most similar pair of comments in /data/comments.txt using embeddings.
def task_a9_similar_comments():
    input_file = DATA_DIR / "comments.txt"
    output_file = DATA_DIR / "comments-similar.txt"
    ensure_in_data(input_file)
    ensure_in_data(output_file)
    if not input_file.exists():
        raise ValueError(f"{input_file} does not exist.")
    comments = [line.strip() for line in input_file.read_text().splitlines() if line.strip()]
    if len(comments) < 2:
        raise ValueError("Not enough comments to compare.")
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(comments)
    max_sim = -1.0
    pair = (None, None)
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim > max_sim:
                max_sim = sim
                pair = (comments[i], comments[j])
    output_file.write_text(f"{pair[0]}\n{pair[1]}")
    return f"Wrote most similar comments (similarity {max_sim:.4f}) to {output_file}"

### A10. Calculate total Gold ticket sales from /data/ticket-sales.db.
def task_a10_ticket_sales_gold():
    db_file = DATA_DIR / "ticket-sales.db"
    output_file = DATA_DIR / "ticket-sales-gold.txt"
    ensure_in_data(db_file)
    ensure_in_data(output_file)
    if not db_file.exists():
        raise ValueError(f"{db_file} does not exist.")
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    result = cur.fetchone()[0]
    conn.close()
    if result is None:
        result = 0
    output_file.write_text(str(result))
    return f"Total Gold ticket sales ({result}) written to {output_file}"

#####################
# Phase B: Business Tasks (B3–B9)
#####################

### B3. Fetch data from an API and save it.
def task_b3_fetch_api_data():
    # Example: fetch JSON from a given API URL and save to /data/api_data.json
    api_url = "https://jsonplaceholder.typicode.com/todos/1"  # sample API
    output_file = DATA_DIR / "api_data.json"
    ensure_in_data(output_file)
    import requests
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception("Failed to fetch API data")
    output_file.write_text(response.text)
    return f"Fetched data from API and saved to {output_file}"

### B4. Clone a git repository and make a commit.
def task_b4_clone_repo():
    # Example: clone a repo into /data/clone_repo and create a dummy file and commit it.
    repo_url = "https://github.com/sanand0/tools-in-data-science-public.git"  # example repo
    clone_dir = DATA_DIR / "clone_repo"
    if clone_dir.exists():
        raise ValueError(f"Directory {clone_dir} already exists. Cannot clone into existing directory.")
    ensure_in_data(clone_dir)
    subprocess.run(["git", "clone", repo_url, str(clone_dir)], check=True)
    # Create a dummy file, add and commit.
    dummy_file = clone_dir / "dummy.txt"
    dummy_file.write_text("This is a dummy file for testing commits.")
    subprocess.run(["git", "-C", str(clone_dir), "add", "dummy.txt"], check=True)
    subprocess.run(["git", "-C", str(clone_dir), "commit", "-m", "Add dummy file"], check=True)
    return f"Cloned repo into {clone_dir} and committed dummy file."

### B5. Run a SQL query on a SQLite (or DuckDB) database.
def task_b5_run_sql_query():
    # Example: execute a query on a database file in /data and save the result.
    db_file = DATA_DIR / "sample.db"
    output_file = DATA_DIR / "sql_query_result.txt"
    ensure_in_data(db_file)
    ensure_in_data(output_file)
    if not db_file.exists():
        raise ValueError(f"{db_file} does not exist.")
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    # Example query: count rows in a table called "records"
    cur.execute("SELECT COUNT(*) FROM records")
    result = cur.fetchone()[0]
    conn.close()
    output_file.write_text(str(result))
    return f"Executed SQL query. Row count: {result} written to {output_file}"

### B6. Extract data from (scrape) a website.
def task_b6_scrape_website():
    # Example: scrape the title of a webpage and save it to /data/scraped.txt
    import requests
    from bs4 import BeautifulSoup
    url = "https://www.example.com"
    output_file = DATA_DIR / "scraped.txt"
    ensure_in_data(output_file)
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch webpage")
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    output_file.write_text(title)
    return f"Scraped webpage title '{title}' and saved to {output_file}"

### B7. Compress or resize an image.
def task_b7_resize_image():
    # Example: resize /data/input_image.png to 50% of its original size and save as /data/resized_image.png
    from PIL import Image
    input_file = DATA_DIR / "input_image.png"
    output_file = DATA_DIR / "resized_image.png"
    ensure_in_data(input_file)
    ensure_in_data(output_file)
    if not input_file.exists():
        raise ValueError(f"{input_file} does not exist.")
    with Image.open(input_file) as img:
        new_size = (img.width // 2, img.height // 2)
        resized_img = img.resize(new_size)
        resized_img.save(output_file)
    return f"Resized image saved to {output_file}"

### B8. Transcribe audio from an MP3 file.
def task_b8_transcribe_audio():
    # Example: transcribe /data/audio.mp3 using LLM (you could integrate with an actual speech-to-text API)
    input_file = DATA_DIR / "audio.mp3"
    output_file = DATA_DIR / "audio_transcription.txt"
    ensure_in_data(input_file)
    ensure_in_data(output_file)
    if not input_file.exists():
        raise ValueError(f"{input_file} does not exist.")
    # For simplicity, encode the MP3 and ask LLM to “transcribe”
    with input_file.open("rb") as f:
        audio_data = f.read()
    audio_b64 = base64.b64encode(audio_data).decode("utf-8")
    prompt = f"Transcribe the following MP3 audio (base64 encoded):\n\n{audio_b64}"
    transcription = call_llm(prompt).strip()
    output_file.write_text(transcription)
    return f"Audio transcription saved to {output_file}"

### B9. Convert Markdown to HTML.
def task_b9_md_to_html():
    # Convert /data/markdown.md to HTML and save as /data/markdown.html
    import markdown
    input_file = DATA_DIR / "markdown.md"
    output_file = DATA_DIR / "markdown.html"
    ensure_in_data(input_file)
    ensure_in_data(output_file)
    if not input_file.exists():
        raise ValueError(f"{input_file} does not exist.")
    md_text = input_file.read_text()
    html = markdown.markdown(md_text)
    output_file.write_text(html)
    return f"Converted Markdown to HTML at {output_file}"

#####################
# Task Router: Choose which task to run based on keywords.
def run_task(task_description: str):
    task_lower = task_description.lower()
    # --- Phase A tasks ---
    if "datagen" in task_lower or ("uv" in task_lower and "datagen.py" in task_lower):
        match = re.search(r"email:\s*([\w\.-]+@[\w\.-]+)", task_description, re.IGNORECASE)
        if not match:
            raise ValueError("No email address found in task description for A1.")
        email = match.group(1)
        return task_a1_run_datagen(email)
    elif "prettier" in task_lower and "format.md" in task_lower:
        return task_a2_format_markdown()
    elif "wednesday" in task_lower and "dates.txt" in task_lower:
        return task_a3_count_wednesdays()
    elif "contact" in task_lower:
        return task_a4_sort_contacts()
    elif "log" in task_lower and "most recent" in task_lower:
        return task_a5_recent_logs()
    elif "docs" in task_lower and "markdown" in task_lower:
        return task_a6_index_docs()
    elif "email" in task_lower and "sender" in task_lower:
        return task_a7_extract_email_sender()
    elif "credit" in task_lower and "card" in task_lower:
        return task_a8_extract_credit_card()
    elif "comment" in task_lower and "similar" in task_lower:
        return task_a9_similar_comments()
    elif "ticket" in task_lower and "gold" in task_lower:
        return task_a10_ticket_sales_gold()
    # --- Phase B tasks ---
    elif "fetch" in task_lower and "api" in task_lower:
        return task_b3_fetch_api_data()
    elif "clone" in task_lower and "repo" in task_lower:
        return task_b4_clone_repo()
    elif "sql" in task_lower and "query" in task_lower:
        return task_b5_run_sql_query()
    elif "scrape" in task_lower or "website" in task_lower:
        return task_b6_scrape_website()
    elif "resize" in task_lower or ("compress" in task_lower and "image" in task_lower):
        return task_b7_resize_image()
    elif "transcribe" in task_lower and ("audio" in task_lower or "mp3" in task_lower):
        return task_b8_transcribe_audio()
    elif "convert" in task_lower and "markdown" in task_lower and "html" in task_lower:
        return task_b9_md_to_html()
    else:
        raise ValueError("Task not recognized or not implemented yet.")
