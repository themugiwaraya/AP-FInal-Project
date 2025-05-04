import logging
import os
import uuid
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from aiohttp import ClientSession
import chromadb
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from langchain.embeddings import OllamaEmbeddings
import numpy as np
from bs4 import BeautifulSoup
import aiohttp

TOKEN = "7418707631:AAGT9UGwC5lofsIMCj-ba8W4zjtoOr_ZdGk"  
OLLAMA_URL = "http://localhost:11434/api/generate"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot setup
app = Application.builder().token(TOKEN).build()

user_selected_docs = {}
chat_history = {}

def get_main_menu():
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("📎 Добавить URL")], 
            [KeyboardButton("📂 Добавить документ")],
            [KeyboardButton("📜 Мои документы")]  
        ],
        resize_keyboard=True
    )

async def handle_my_documents(update: Update, context):
    if not documents:
        await update.message.reply_text("📂 У вас пока нет сохраненных документов.")
        return
    
    keyboard = []
    for doc_id, title in documents.items():
        keyboard.append([InlineKeyboardButton(title, callback_data=f"open_{doc_id}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("📜 Ваши документы:", reply_markup=reply_markup)
   
def get_chat_history(user_id, doc_id):
    """Get chat history for specific user and document"""
    return chat_history.get(user_id, {}).get(doc_id, [])   

def add_to_chat_history(user_id, doc_id, query, response):
    """Add a new entry to chat history for specific user and document"""
    if user_id not in chat_history:
        chat_history[user_id] = {}
    if doc_id not in chat_history[user_id]:
        chat_history[user_id][doc_id] = []
    chat_history[user_id][doc_id].append((query, response))
    
async def generate_ollama_response(context, query):
    """Generate a response using Ollama"""
    prompt = f"""Analyze the provided context and answer the question.  
    If there is not enough information, state that explicitly.

Context:
{context}

Question: {query}

Answer:"""
    
    logger.info("Sending request to Ollama")

    async with ClientSession() as session:
        async with session.post(OLLAMA_URL, json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }) as response:
            if response.status == 200:
                result = await response.json()
                return result.get('response', 'Failed to generate a response')
            else:
                error_text = await response.text()
                logger.error(f"Ollama error: {error_text}")
                return "An error occurred while generating the response"

chroma_client = chromadb.PersistentClient(path="chroma_db")
try:
    chroma_client.delete_collection("documentsv2")
except:
    pass
collection = chroma_client.create_collection(name="documentsv2", metadata={"hnsw:space": "cosine"})

embedder = OllamaEmbeddings(model="llama3.2", base_url="http://localhost:11434")

documents = {}

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def create_wordcloud(text, file_path):
    wordcloud = WordCloud(font_path="arial.ttf", width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(file_path)
    plt.close()

def save_document_with_embedding(doc_id, text):
    try:
        embedding = embedder.embed_query(text)
        collection.add(ids=[doc_id], documents=[text], embeddings=[embedding])
        logger.info(f"Документ {doc_id} добавлен в ChromaDB")
    except Exception as e:
        logger.error(f"Ошибка при сохранении документа: {e}")
        raise
def create_doc_kb(doc_id):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📄 Открыть", callback_data=f"open_{doc_id}")],
        [InlineKeyboardButton("❌ Удалить", callback_data=f"delete_{doc_id}")]
    ])

def create_chat_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚪 Выйти из чата", callback_data="exit_chat")]
    ])

async def start(update: Update, context):
    welcome_text = (
        "Привет!  Я — ваш бот для работы с документами и URL-ссылками.\n\n"
        "Вот что я умею:\n"
        "1. 📎 Добавить URL — можете отправить мне ссылку, и я извлеку текст с веб-страницы.\n"
        "2. 📂 Добавить документ — отправьте мне документ, и я извлеку текст, создам облако слов и сохраню его для дальнейшей работы.\n"
        "3. 📜 Мои документы — покажу вам все сохранённые документы и позволю их открыть, удалить или взаимодействовать с ними.\n\n"
        "Просто выберите нужное действие, и я помогу вам!"
    )
    
    await update.message.reply_text(welcome_text, reply_markup=get_main_menu())


async def fetch_url_content(url: str) -> str:
    """Extracts text from webpage URL"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                    return text
                else:
                    return f"Ошибка получения данных с URL: {response.status}"
    except Exception as e:
        return f"Ошибка при запросе URL: {str(e)}"

async def handle_document(update: Update, context):
    document = update.message.document
    file_id = document.file_id
    file_name = document.file_name
    doc_id = str(uuid.uuid4())
    file_path = f"uploads/{doc_id}_{file_name}"

    new_file = await context.bot.get_file(file_id)
    await new_file.download_to_drive(file_path)
    text = extract_text(file_path)
    if text:
        try:
            save_document_with_embedding(doc_id, text)
            wordcloud_path = f"uploads/{doc_id}_wordcloud.png"
            create_wordcloud(text, wordcloud_path)
            documents[doc_id] = file_name

            with open(wordcloud_path, "rb") as photo:
                await update.message.reply_photo(photo, caption=f"Документ сохранён: {file_name}", reply_markup=create_doc_kb(doc_id))
        except Exception as e:
            await update.message.reply_text(f"Ошибка при сохранении документа: {str(e)}")
    else:
        await update.message.reply_text("Не удалось извлечь текст из документа.")

async def handle_url(update: Update, context):
    url = update.message.text
    doc_id = str(uuid.uuid4())

    logger.info(f"Processing URL: {url}")

    try:
        text = await fetch_url_content(url)
        if text:
            save_document_with_embedding(doc_id, text)
            documents[doc_id] = url
            
            wordcloud_path = f"uploads/{doc_id}_wordcloud.png"
            create_wordcloud(text, wordcloud_path)

            with open(wordcloud_path, "rb") as photo:
                await update.message.reply_photo(photo, caption=f"Ссылка сохранена: {url}", reply_markup=create_doc_kb(doc_id))
        else:
            await update.message.reply_text("Не удалось извлечь текст с указанного URL.")
    except Exception as e:
        await update.message.reply_text(f"Ошибка при сохранении ссылки: {str(e)}")

async def callback_handler(update: Update, context):
    query = update.callback_query
    await query.answer()

    data = query.data
    user_id = query.from_user.id

    if data == "exit_chat":
        user_selected_docs.pop(user_id, None)
        await query.message.reply_text("Вы вышли из чата", reply_markup=get_main_menu())
        return

    if data.startswith("delete_"):
        doc_id = data.split("_")[1]
        if doc_id in documents:
            # Delete document and its chat history
            wordcloud_path = f"uploads/{doc_id}_wordcloud.png"
            if os.path.exists(wordcloud_path):
                os.remove(wordcloud_path)
            documents.pop(doc_id)
            # Remove chat history for this document for all users
            for user_histories in chat_history.values():
                user_histories.pop(doc_id, None)
            await query.message.reply_text("Документ удален", reply_markup=get_main_menu())
        return

    doc_id = data.split("_")[1]
    if data.startswith("open_"):
        if doc_id not in documents:
            await query.message.reply_text("Ошибка: документ не найден. Возможно, он был удален.")
            return
        
        user_selected_docs[user_id] = doc_id
        wordcloud_path = f"uploads/{doc_id}_wordcloud.png"

        # Get history specific to this user and document
        doc_history = get_chat_history(user_id, doc_id)
        history_text = "\n\n".join([f"❓ {q}\n💬 {a}" for q, a in doc_history]) or "История запросов пуста."
        
        if os.path.exists(wordcloud_path):
            with open(wordcloud_path, "rb") as photo:
                await query.message.reply_photo(
                    photo,
                    caption=f"Вы выбрали: {documents[doc_id]}\nОтправьте запрос:\n\n📜 История:\n{history_text}",
                    reply_markup=create_chat_kb()
                )
        else:
            await query.message.reply_text(f"Файл отсутствует или был удален.\n\n📜 История:\n{history_text}")

async def handle_query(update: Update, context):
    user_id = update.message.from_user.id
    doc_id = user_selected_docs.get(user_id)

    if not doc_id or doc_id not in documents:
        await update.message.reply_text("Сначала выберите документ")
        return

    query_text = update.message.text
    try:
        query_embedding = embedder.embed_query(query_text)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)

        if results and results['documents']:
            context = "\n".join(results['documents'][0])
            processing_msg = await update.message.reply_text("⏳ Генерирую ответ...")

            response = await generate_ollama_response(context, query_text)
            await processing_msg.delete()
            await update.message.reply_text(response)

            add_to_chat_history(user_id, doc_id, query_text, response)

        else:
            await update.message.reply_text("Не найдено релевантной информации в документе.")
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        await update.message.reply_text(f"Ошибка поиска: {str(e)}")
        
async def handle_text(update: Update, context):
    text = update.message.text
    if text == "📜 Мои документы":
        await handle_my_documents(update, context)
    elif text.startswith(('http://', 'https://')):
        await handle_url(update, context)
    else:
        await handle_query(update, context)

def main():
    os.makedirs("uploads", exist_ok=True)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT, handle_text))  # Modified this line
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.run_polling()

if __name__ == "__main__":
    main()