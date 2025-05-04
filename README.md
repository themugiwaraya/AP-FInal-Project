# AP-FInal-Project
Document Analysis and Chat System
Overview
A Streamlit-based application that allows users to upload, analyze, and chat with PDF and text documents using local AI models.
Features

Document Upload: PDF and text file support
Web URL Content Scraping
Document Visualization (Word Cloud, Frequency Analysis)
Local AI-Powered Document Chat
Persistent Document and Chat History

Prerequisites

Python 3.8+
Ollama (for local AI models)
Llama3.2 model installed

Installation

Clone the repository:

bashCopygit clone https://github.com/yourusername/document-chat-system.git
cd document-chat-system

Create a virtual environment:

bashCopypython -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install dependencies:

bashCopypip install -r requirements.txt

Install Ollama and Llama3.2 model:

bashCopy# Follow Ollama installation instructions at https://ollama.com
ollama pull llama3.2
Running the Application
bashCopystreamlit run app.py
Usage
Document Management

Upload PDFs or text files via sidebar
Add web content by entering a URL
Chat with uploaded documents
Delete documents as needed

Chatting with Documents

Select a document from the sidebar
Ask questions in the chat interface
AI generates context-aware responses

Configuration
Modify key parameters in app.py:

Change embedding/chat models
Adjust document chunk sizes
Configure logging levels

Dependencies

Streamlit
ChromaDB
LangChain
Ollama
WordCloud
Matplotlib

Troubleshooting

Ensure Ollama is running
Check model availability
Verify Python and library versions
