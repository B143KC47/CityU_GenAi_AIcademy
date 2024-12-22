import os
import sqlite3
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
token = os.environ.get("GITHUB_TOKEN") 
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

DB_NAME = "vocab.db"


# -------------------------------------------------------------------
# Database Setup
# -------------------------------------------------------------------
def create_database(db_name=DB_NAME):
    """
    Create / connect to SQLite database.    
    We will create two tables:
      - vocabulary: stores 5 advanced-level vocabulary words
      - patterns: stores 2 sentence patterns for each vocabulary word
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create the vocabulary table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vocabulary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT UNIQUE NOT NULL
        )
    """)

    # Create the sentence patterns table 
    # Each pattern references the vocabulary table via word_id
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word_id INTEGER NOT NULL,
            pattern TEXT NOT NULL,
            FOREIGN KEY(word_id) REFERENCES vocabulary(id)
        )
    """)

    conn.commit()
    return conn


# -------------------------------------------------------------------
# GPT: Generate 5 advanced-level vocabulary words (Step 1)
# -------------------------------------------------------------------
def generate_vocabulary_words(client, model_name):
    """
    Ask GPT to generate ONLY 5 secondary 6 vocabulary words, one per line, 
    WITHOUT definitions or explanations.
    """
    prompt = (
        "Generate exactly 5 secondary 6 English vocabulary words. "
        "No definitions, no explanations—just the words, one per line."
    )
    messages = [
        SystemMessage(content="You are a helpful english vocabulary generator.You will generate useful words for exam writing that is secondary 6 level"),
        UserMessage(content=prompt),
    ]

    response = client.complete(
        messages=messages,
        model=model_name,
        temperature=0.7,
        max_tokens=100,
        top_p=1.0
    )
    return response.choices[0].message.content.strip()


# -------------------------------------------------------------------
# GPT: Generate 2 sentence patterns for each vocabulary word (Step 2)
# -------------------------------------------------------------------
def generate_sentence_patterns(client, model_name, word):
    """
    Generate exactly 2 sentence patterns for a given vocabulary word.
    Return them in a list, one pattern per line, no extra text.
    """
    prompt = (
        f"Provide exactly 2 descriptive sentence patterns using the word '{word}'. "
        "Output them on separate lines, and do not include explanations."
    )
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content=prompt),
    ]

    response = client.complete(
        messages=messages,
        model=model_name,
        temperature=0.7,
        max_tokens=200,
        top_p=1.0
    )
    patterns_text = response.choices[0].message.content.strip()
    # Split by lines to capture the two raw patterns
    patterns = [p.strip() for p in patterns_text.split('\n') if p.strip()]
    return patterns


# -------------------------------------------------------------------
# Store Vocabulary and Patterns in Database (Step 3)
# -------------------------------------------------------------------
def store_vocabulary_and_patterns(conn, words):
    """
    For each word in 'words':
      1) Insert the word into the vocabulary table (if not already present).
      2) Generate 2 sentence patterns for the word.
      3) Insert those 2 patterns into the patterns table referencing the word id.
    """
    cursor = conn.cursor()
    new_words = []

    for word in words:
        # Insert the word if it doesn't exist
        cursor.execute("SELECT id FROM vocabulary WHERE word = ?", (word,))
        row = cursor.fetchone()
        if not row:
            cursor.execute("INSERT INTO vocabulary (word) VALUES (?)", (word,))
            conn.commit()
            cursor.execute("SELECT id FROM vocabulary WHERE word = ?", (word,))
            row = cursor.fetchone()
            new_words.append(word)

        word_id = row[0]

        # Generate 2 sentence patterns for this word
        patterns = generate_sentence_patterns(client, model_name, word)

        # Store each pattern
        for pattern in patterns:
            cursor.execute("INSERT INTO patterns (word_id, pattern) VALUES (?, ?)", (word_id, pattern,))
        conn.commit()

    return new_words


# -------------------------------------------------------------------
# Fetch the 5 Most Recent Words and Their Patterns (Step 4 - partial)
# -------------------------------------------------------------------
def fetch_recent_vocab_and_patterns(conn, limit=5):
    """
    Fetch up to 'limit' most recently added vocabulary words
    and their associated sentence patterns from the database.
    Return a list of (word, [patterns]) tuples.
    """
    cursor = conn.cursor()
    
    # Fetch the most recent words by their ID descending
    cursor.execute("""
        SELECT id, word 
        FROM vocabulary 
        ORDER BY id DESC 
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()

    result = []
    for row in rows:
        word_id, word = row
        # Fetch the patterns for this word
        cursor.execute("SELECT pattern FROM patterns WHERE word_id = ?", (word_id,))
        patterns = [r[0] for r in cursor.fetchall()]
        result.append((word, patterns))

    return result


# -------------------------------------------------------------------
# GPT: Generate a small paragraph & 5 questions using the vocab/patterns (Step 4 cont.)
# -------------------------------------------------------------------
def generate_paragraph_and_questions(client, model_name, recent_data):
    """
    Use the recent 5 vocab words and their sentence patterns to create:
      - A small paragraph that includes the words and hints of patterns
      - 5 questions (one question about each word/pattern usage)
    """
    # Build a text input describing what we have
    words_and_patterns_str = ""
    for (word, patterns) in recent_data:
        pattern_bullets = "\n".join(f"- {pt}" for pt in patterns)
        words_and_patterns_str += f"Word: {word}\nPatterns:\n{pattern_bullets}\n\n"

    prompt = (
        "You have the following 5 vocabulary words and their 2 sentence patterns each:\n\n"
        f"{words_and_patterns_str}\n"
        "1) Write a short paragraph (no more than 300 words) that naturally incorporates these words or references their patterns.\n"
        "2) Then create exactly 5 questions to help a learner practice using these words and patterns.\n"
        "Do not include answers, just the questions.\n"
        "Format your response with the paragraph first, then a clear separation, then the questions listed.\n"
    )

    messages = [
        SystemMessage(content="You are a helpful assistant specialized in educational content."),
        UserMessage(content=prompt),
    ]

    response = client.complete(
        messages=messages,
        model=model_name,
        temperature=0.7,
        max_tokens=1500,
        top_p=1.0
    )

    return response.choices[0].message.content.strip()


# -------------------------------------------------------------------
# Generate an HTML Form (Step 5)
# -------------------------------------------------------------------
def generate_html_form(paragraph_and_questions, output_filename="lesson.html"):
    """
    Parse the output containing the paragraph and 5 questions.
    Generate a small HTML file with:
      - the paragraph
      - five question fields
      - a submit button
    This will allow the user to input answers to the 5 questions.
    """
    # Separate the paragraph from the questions.
    # Assume that GPT's response will have some marker like "Questions:" or a line break separation.
    # We'll do a naive split by newlines or use a marker.
    content_lines = paragraph_and_questions.strip().split('\n')
    
    # Find the line index that starts the questions
    # We'll do something simple: find the first line containing "Question" or "question".
    # If we don't find it, fallback to the entire text as paragraph.
    question_start_idx = None
    for i, line in enumerate(content_lines):
        if "question" in line.lower():
            question_start_idx = i
            break

    if question_start_idx is None:
        # No question found, treat entire text as paragraph
        paragraph = paragraph_and_questions
        questions = []
    else:
        paragraph_lines = content_lines[:question_start_idx]
        questions_lines = content_lines[question_start_idx:]
        paragraph = "\n".join(paragraph_lines).strip()
        questions = [q for q in questions_lines if q.strip()]

    # Build the HTML
    html_content = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        "    <meta charset='UTF-8'>\n"
        "    <title>Vocabulary & Sentence Patterns</title>\n"
        "</head>\n"
        "<body>\n"
        f"    <h1>Short Paragraph</h1>\n"
        f"    <p>{paragraph}</p>\n"
        "    <hr/>\n"
        "    <h2>Practice Questions</h2>\n"
        "    <form method='post' action='/check_answers'>\n"
    )

    for idx, question_text in enumerate(questions, start=1):
        html_content += (
            f"        <p><strong>{question_text}</strong></p>\n"
            f"        <textarea name='answer_{idx}' rows='2' cols='60'></textarea>\n"
            "        <br/>\n"
        )

    html_content += (
        "        <input type='submit' value='Submit Answers'/>\n"
        "    </form>\n"
        "</body>\n"
        "</html>"
    )

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML learning material has been generated: {output_filename}")


# -------------------------------------------------------------------
# (Optional) Step 6: Show the user the correct way of thinking
# -------------------------------------------------------------------


# Make sure to import necessary modules and define client and model_name before using this function
# For example:
# from your_ai_library import Client, SystemMessage, UserMessage
# client = Client(api_key='your_api_key')
# model_name = 'your_model_name'

def check_answers(user_answers, paragraph_questions):
    # Construct the prompt for the AI
    prompt = (
        "Here is a paragraph, a set of questions, and the student's answers to those questions.\n\n"
        "Paragraph and Questions:\n"
        f"{paragraph_questions}\n\n"
        "Student's Answers:\n"
    )
    
    # Append each answer with its corresponding question number
    for i, ans in enumerate(user_answers, start=1):
        prompt += f"Q{i}: {ans}\n"
    
    # Instruct the AI to provide feedback for each answer
    prompt += (
        "\nPlease review each answer and provide feedback on its correctness. "
        "If the answer is correct, acknowledge it. If not, provide guidance or hints to help improve the answer."
    )
    
    try:
        # Make a request to the OpenAI API
        response = client.complete(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in educational content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300,
            top_p=1.0
        )
        
        # Extract the AI's feedback from the response
        ai_feedback = response.choices[0].message['content'].strip()
        
        return ai_feedback
    
    except Exception as e:
        # Handle potential errors gracefully
        return f"An error occurred while generating feedback: {str(e)}"

# -------------------------------------------------------------------
# Main flow
# -------------------------------------------------------------------
def main():
    # 1) Generate exactly 5 advanced-level vocabulary words
    vocab_text = generate_vocabulary_words(client, model_name)
    print("Generated 5 advanced vocabulary words:\n", vocab_text, "\n")

    # Split the vocab_summary into lines
    words = [w.strip() for w in vocab_text.split('\n') if w.strip()]

    # 2 & 3) Create database, store words, and also store 2 sentence patterns for each word
    conn = create_database()
    new_words = store_vocabulary_and_patterns(conn, words)
    if not new_words:
        print("No new vocabulary words were added to the database.")
        conn.close()
        return
    else:
        print(f"New vocabulary words added to the database: {new_words}\n")

    # 4) Fetch the recent 5 vocabulary words and their patterns, 
    #    and ask GPT to create a small paragraph and 5 questions
    recent_data = fetch_recent_vocab_and_patterns(conn, limit=5)
    paragraph_and_questions = generate_paragraph_and_questions(client, model_name, recent_data)
    print("Paragraph and Questions:\n")
    print(paragraph_and_questions, "\n")

    # 5) Generate an HTML form with the paragraph and questions
    generate_html_form(paragraph_and_questions, "lesson.html")


    user_answers = [
        "My answer to question 1",
        "Another thought for Q2",
        "Q3 response here",
        "Q4 example usage",
        "Q5 final reply"
    ]
    print("Simulating user answers submission and checking...\n")
    feedback = check_answers(user_answers,paragraph_and_questions)
    print("Feedback to user:\n", feedback)

    conn.close()


if __name__ == "__main__":
    main()

