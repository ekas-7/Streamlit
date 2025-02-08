import streamlit as st
import os
import warnings
import speech_recognition as sr
import tempfile
import time
from gtts import gTTS  # ✅ Google Text-to-Speech
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Suppress deprecated warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

# Load environment variables
load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("GEMINI_API_KEY is not set in environment variables.")
    st.stop()

# Initialize embeddings
embeddings = OllamaEmbeddings(model="llama3")

# Initialize session state for vector_store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Initialize Google Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=gemini_api_key,
    temperature=0.7
)

# Set up a ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a helpful and informative educational chatbot designed to assist students, educators, and anyone interested in learning.  Adhere to the following guidelines:

•⁠  ⁠*Understand the User's Learning Needs:* Carefully analyze the user's input to understand their learning goals, current knowledge level, and the specific information they are seeking. Be prepared to ask clarifying questions if the request is ambiguous, incomplete, or too broad.  Don't assume anything the user hasn't explicitly stated.  Are they looking for a definition, an explanation, help with a specific problem, or resources for further study?

•⁠  ⁠*Provide Accurate and Relevant Educational Content:* Base your responses on reliable and up-to-date educational sources.  If you are unsure of the answer, acknowledge that you don't know rather than providing potentially incorrect information.  If appropriate, suggest relevant educational resources, such as textbooks, websites, articles, or videos. Cite your sources when possible.

•⁠  ⁠*Explain Concepts Clearly and Concisely:* Keep your explanations clear, concise, and easy to understand. Avoid unnecessary jargon or technical terms unless specifically requested by the user or appropriate for the user's level.  Use clear and simple language. Organize your information logically, using bullet points, lists, diagrams, or examples when appropriate.  Tailor your explanations to the user's level of understanding.

•⁠  ⁠*Be Helpful and Encouraging:* Maintain a polite, patient, and encouraging tone.  Address the user respectfully and avoid being condescending or dismissive.  Offer additional assistance or suggest related topics if you think it would be helpful.  Express gratitude for the user's question and encourage their learning.

•⁠  ⁠*Handle Ambiguity Gracefully:* If the user's request is unclear or open to interpretation, ask clarifying questions to ensure you understand what they are looking for.  Provide multiple possible interpretations if necessary.

•⁠  ⁠*Follow Instructions:* Pay close attention to any specific instructions given by the user, such as the desired format of the response, the length of the response, or the target audience (e.g., elementary school student, college student, professional).

•⁠  ⁠*Acknowledge Limitations:* Be transparent about your limitations as a language model.  If you are unable to fulfill the user's request (e.g., provide personalized tutoring, grade an assignment), explain why and suggest alternative solutions.  For instance, if the user asks for help with a complex problem that requires visual aids, you might suggest finding a relevant video tutorial or consulting a textbook.

•⁠  ⁠*Example Interaction:*

    *User:* What is photosynthesis?

    *Chatbot:* Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy in the form of sugars.  Essentially, they use sunlight, water, and carbon dioxide to create their own food.  This process is crucial for life on Earth, as it produces the oxygen we breathe.  Would you like me to explain the different stages of photosynthesis or provide some examples of organisms that use it?

Now, respond to the following user context and input : 
    
    Context: {context}
    """),
    ("user", "{input}")
])


# Create the document chain
document_chain = create_stuff_documents_chain(llm, prompt)

st.title("Zing Sensei")

# Sidebar for text addition
st.sidebar.header("Add Text to Knowledge Base")
text_input = st.sidebar.text_area("Enter text to add to the vector store:")
if st.sidebar.button("Add Text"):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        final_documents = text_splitter.split_text(text_input)
        
        new_vectors = FAISS.from_texts(final_documents, embeddings)
        
        if st.session_state.vector_store is None:
            st.session_state.vector_store = new_vectors
        else:
            st.session_state.vector_store.merge_from(new_vectors)
        
        st.sidebar.success("Text added successfully!")
    except Exception as e:
        st.sidebar.error(f"Error adding text: {e}")

# Function to convert audio to text
def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio."
        except sr.RequestError as e:
            return f"Speech Recognition API error: {e}"

# Main section for querying the chatbot
st.header("Ask a Question")
query = st.text_input("Enter your question:")
audio_value = st.audio_input("Record a voice message")

if audio_value:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_value.read())  # ✅ Read bytes before writing
        temp_audio_path = temp_audio.name

    query = audio_to_text(temp_audio_path)
    st.write(f"Transcribed Text: {query}")

response_text = ""

if st.button("Get Response & Read Aloud"):
    if not query:
        st.error("Please enter a text or audio query.")
    elif st.session_state.vector_store is None:
        st.error("No data available in vector store. Please add text first.")
    else:
        try:
            retriever = st.session_state.vector_store.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            start = time.time()
            response = retrieval_chain.invoke({"input": query})
            end = time.time()
            
            response_text = response['answer']
            st.success(response_text)
            st.write(f"Response time: {end - start:.2f} seconds")

            # Convert response text to speech using gTTS
            if response_text:
                tts = gTTS(response_text)
                tts_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                tts.save(tts_audio_path)

                # Play the generated speech in Streamlit
                st.audio(tts_audio_path, format="audio/mp3")

        except Exception as e:
            st.error(f"Error processing request: {e}")
