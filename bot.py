import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import os
from mem0 import Memory
from google import genai
from groq import Groq  
import logging

# -------------------- SETUP LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8')  
    ]
)
logger = logging.getLogger(__name__)

# Add a simple console handler without emojis for Windows compatibility
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [SIMPLIFIED] %(message)s')
console_handler.setFormatter(console_formatter)

# Create a filter to remove emojis from console output
class EmojiFilter(logging.Filter):
    def filter(self, record):
        # Replace emojis with text equivalents for console
        msg = record.getMessage()
        emoji_replacements = {
            '🚀': '[START]', '✅': '[OK]', '❌': '[ERROR]', '🔑': '[API]',
            '🧠': '[MEMORY]', '📋': '[SETUP]', '🤖': '[AI]', '💾': '[SAVE]',
            '🔍': '[SEARCH]', '📚': '[FOUND]', '📖': '[RESTORED]', '📝': '[INFO]',
            '💬': '[CHAT]', '🎉': '[SUCCESS]'
        }
        for emoji, replacement in emoji_replacements.items():
            msg = msg.replace(emoji, replacement)
        record.msg = msg
        record.args = ()
        return True

console_handler.addFilter(EmojiFilter())
logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# -------------------- CONFIG --------------------
load_dotenv()
if 'config_loaded' not in st.session_state:
    logger.info("[OK] Environment variables loaded")
    st.session_state.config_loaded = True

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if 'config_loaded' in st.session_state and st.session_state.config_loaded:
    logger.info(f"[API] Keys loaded: Pinecone={'OK' if PINECONE_API_KEY else 'MISSING'}, Gemini={'OK' if GEMINI_API_KEY else 'MISSING'}")
    st.session_state.config_loaded = False  # Prevent repeated logging


# -------------------- MEM0 CONFIG --------------------
MEM0_CONFIG = {
    "llm": {
        "provider": "groq",
        "config": {
        "model": "llama-3.1-8b-instant",  
        "temperature": 0.7,
        "api_key": GROQ_API_KEY
        }
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/embedding-001",
            "api_key": GEMINI_API_KEY
        }
    },
    "vector_store": {
        "provider": "pinecone",
        "config": {
            "collection_name": "chat",
            "embedding_model_dims": 768,
            "api_key": PINECONE_API_KEY,
            "serverless_config": {
                "cloud": "aws",
                "region": "us-east-1"
            },
            "metric": "cosine"
        }
    }
}

# -------------------- INITIALIZE MEMORY --------------------
# Use session state to prevent multiple initializations
if 'memory_initialized' not in st.session_state:
    logger.info("[MEMORY] Initializing Mem0 memory manager...")
    memory_manager = Memory.from_config(MEM0_CONFIG)
    logger.info("[OK] Memory manager initialized successfully")

    # -------------------- INITIALIZE GEMINI --------------------
    logger.info("[AI] Initializing Gemini client...")
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("[OK] Groq client initialized successfully")
    st.session_state.groq_client = groq_client
    
    # Store in session state to prevent re-initialization
    st.session_state.memory_initialized = True
    st.session_state.memory_manager = memory_manager
    st.session_state.groq_client = groq_client
    logger.info("[SESSION] Components stored in session state")
else:
    # Retrieve from session state
    memory_manager = st.session_state.memory_manager
    groq_client = st.session_state.groq_client

# -------------------- STREAMLIT UI --------------------
if 'app_started' not in st.session_state:
    logger.info("[APP] Starting E-commerce Bot Streamlit App...")
    st.session_state.app_started = True

st.title("E-commerce Bot")
st.sidebar.header("User Session Management")

# User selection
user_options = ["user1", "user2", "new_user"]
selected_user = st.sidebar.selectbox("Select User", user_options)

if selected_user == "new_user":
    if 'temp_user_id' not in st.session_state:
        st.session_state['temp_user_id'] = f"guest_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    user_id = st.session_state['temp_user_id']
else:
    user_id = selected_user
    if 'temp_user_id' in st.session_state:
        del st.session_state['temp_user_id']

st.sidebar.write(f"**Current User:** {user_id}")

# Initialize session state for chat histories
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if user_id not in st.session_state.chat_histories:
    logger.info(f"🔍 Retrieving past memories for user: {user_id}")
    # Search for all past messages from this user (not just "past conversations")
    past_memories = memory_manager.search(user_id=user_id, query="", limit=10)
    print(f"🔍 Past memories for user {user_id}: {past_memories}")
    st.session_state.chat_histories[user_id] = []
    if past_memories:
        # Extract results from search response
        past_results = past_memories.get('results', []) if isinstance(past_memories, dict) else past_memories
        logger.info(f"📚 Found {len(past_results)} past memories for user {user_id}")
        print(f"🔍 Past memories: {past_results}")

        # Reconstruct conversation history from memories
        user_messages = []
        bot_messages = []
        
        for memory in past_results:
            if 'content' in memory:
                content = memory['content']
                # Try to identify if it's a user or assistant message based on content
                if any(word in content.lower() for word in ['user:', 'question:', 'asked:']):
                    user_messages.append(content)
                elif any(word in content.lower() for word in ['assistant:', 'bot:', 'response:', 'replied:']):
                    bot_messages.append(content)
                else:
                    # Default to user message if unclear
                    user_messages.append(content)
        
        # Create conversation pairs from the most recent messages
        recent_pairs = min(5, len(user_messages), len(bot_messages))
        for i in range(recent_pairs):
            st.session_state.chat_histories[user_id].append({
                "user_msg": user_messages[-(i+1)],  # Most recent first
                "assistant_msg": bot_messages[-(i+1)] if i < len(bot_messages) else "I remember your message but not my response."
            })
        
        # Reverse to show chronological order
        st.session_state.chat_histories[user_id].reverse()
        
        if st.session_state.chat_histories[user_id]:
            logger.info(f"📖 Restored {len(st.session_state.chat_histories[user_id])} conversation pairs from memory")
        else:
            logger.info(f"📝 No conversation history found for user {user_id}")
    else:
        logger.info(f"📝 No past memories found for user {user_id}")
        st.session_state.chat_histories[user_id] = []

# Display chat history
for chat in st.session_state.chat_histories[user_id]:
    st.markdown(f"**User:** {chat['user_msg']}")
    st.markdown(f"**Bot:** {chat['assistant_msg']}")
    st.markdown("---")

# User input
user_input = st.text_input("Ask Query:", key=f"input_{user_id}")

# -------------------- HELPER FUNCTION --------------------
def generate_groq_response(messages):
    """Generate response using Gemini API with memory-enhanced messages"""
    try:
        logger.info("🤖 Generating Gemini response...")
        # Convert messages to the format expected by google-genai
        system_prompt = """You are an intelligent e-commerce assistant with memory capabilities. always remeber users input messages.
Follow all memory operations (ADD, SEARCH, UPDATE, DELETE) and classify user inputs.
Use past memory context to personalize your responses. 
Keep responses concise (2-3 lines). Reference past interactions when relevant.

**EXAMPLES OF MEMORY OPERATIONS**
ADD Example:
User: "Looking for headphones under $200"
Memory Add: Store budget ($200) and product category (headphones)
Response: Based on your previous Sony preference, I recommend the WH-CH720N at $180.

SEARCH Example:
User: "What did I buy last month?"
Memory Search: Retrieve purchase history and spending patterns
Response: Last month you purchased Sony WH-1000XM5 headphones ($299) and an iPhone case ($25).

UPDATE Example:
User: "Actually, I prefer wireless earbuds now"
Memory Update: Change headphone preference to wireless earbuds
Response: Updated your preference to wireless earbuds. The Sony WF-1000XM4 at $179 matches your budget.

DELETE Example:
User: "I'm no longer interested in gaming accessories"
Memory Delete: Remove gaming-related preferences
Response: Removed gaming accessories. Current interests are audio equipment and home decor.
"""

        user_content = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
        
        # Combine system prompt with user content
        system_prompt = f"{system_prompt}\n\nUser: {user_content}\nAssistant:"
        logger.info(f"📝 Prompt prepared for user input: '{user_content[:50]}{'...' if len(user_content) > 50 else ''}'")
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # or "llama-3.1-70b-versatile"
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_content}],
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"✅  Groq response received: '{response_text[:100]}{'...' if len(response_text) > 100 else ''}'")
        return response_text
    except Exception as e:
        logger.error(f"❌  Groq API error: {str(e)}")
        return f"Sorry, something went wrong with Gemini: {str(e)}"

def detect_category(user_input):
    """Simple category detection based on keywords"""
    categories = {
        "shipping": ["delivery", "shipping", "arrive", "track", "ship"],
        "home decor": ["vase", "fountain", "furniture", "cushion", "decor", "home decor", "interior"],
        "products": ["product", "item", "size", "color", "specs"],
        "payment": ["pay", "payment", "card", "price", "cost"],
        "Electronic": ["headphones", "laptop", "phone", "tablet", "gadget", "accessory"],
    }
    
    user_input_lower = user_input.lower()
    for category, keywords in categories.items():
        if any(keyword in user_input_lower for keyword in keywords):
            return category
    return "general"

# -------------------- CHAT LOGIC --------------------
if user_input:
    logger.info(f"💬 User '{user_id}' sent message: '{user_input}'")
    
    #=======================================================================
     # --- DELETE LOGIC  ---
    if "forget" in user_input.lower():
        brand_name = user_input.lower().split("forget")[-1].strip()
        search_results = memory_manager.search(user_id=user_id, query="brand_name", limit=1)
        matches = search_results.get("results", []) if isinstance(search_results, dict) else (search_results or [])
        if matches:
        # Delete all matching memories
            for mem in matches:
                memory_id = mem["id"]
                memory_manager.delete(memory_id)
                logger.info(f"🗑️ Deleted memory ID {memory_id} containing {brand_name}")
            st.markdown(f"✅ All memories about {brand_name.upper()} have been deleted.")
        else:
            st.markdown("⚠️ No memory found to delete.")
        st.stop()  # Stop further execution in Streamlit
        
    #=======================================================================
    # Add user's message to memory with better metadata
    logger.info("💾 Adding user message to memory...")
    timestamp = datetime.now().isoformat()
    user_memory_data = [{"role": "user", "content": f"User message at {timestamp}: {user_input}"}]
    memory_manager.add(user_memory_data, user_id=user_id, metadata={"type": "user_message", "timestamp": timestamp})
    logger.info("✅ User message added to memory")

    # Extract the results from the search response
    logger.info("🔍 Searching for relevant memories...")
    user_memories = memory_manager.search(user_id=user_id, query=user_input, limit=5)
    
# ======================================================================
    # ------ UPDATE LOGIC----------
    negation_keywords = ["don't", "do not", "no longer", "not", "never"]
    is_negation = any(word in user_input.lower() for word in negation_keywords)

    # Search old memory
    search_results = memory_manager.search(user_id=user_id, query=user_input, limit=1)
    matches = search_results.get("results", []) if isinstance(search_results, dict) else (search_results or [])

    if matches:
        memory_id = matches[0]["id"]   
        print(f"Memory found! ID: {memory_id}, content: {matches[0]['memory']}")
        
        if is_negation:
        # Remove or mark as negative preference
            memory_manager.delete(memory_id)
            logger.info(f"🗑️ User no longer prefers this. Memory ID {memory_id} deleted.")
            st.markdown("✅ Preference removed from memory.")
        else:
            # Normal update
            memory_manager.update(memory_id, data=user_input.strip())
            logger.info(f"♻️ Memory updated for user {user_id}: '{user_input}'")
    else:
        if not is_negation:
            # Only add if it's not a negative preference
            memory_manager.add(
                [{"role": "user", "content": user_input}],
                user_id=user_id,
                metadata={"type": "preference", "timestamp": datetime.now().isoformat()}
            )
    #============================================================================
    category_memories = memory_manager.search(user_id="system_categories", query="", limit=1)

    # Handle both dict and list responses consistently
    user_results = user_memories.get("results", []) if isinstance(user_memories, dict) else user_memories or []
    category_results = category_memories.get("results", []) if isinstance(category_memories, dict) else category_memories or []

    logger.info(f"📚 Found {len(user_results)} user memories and {len(category_results)} category memories")
    
    # Build context string from 'memory' field
    memory_context = "\n".join(
        f"- {m['memory']}" for m in (category_results + user_results) if "memory" in m
    )

    messages_for_groq = [
        {"role": "system", "content": f"You are a helpful, concise e-commerce assistant.\n{memory_context}"},
        {"role": "user", "content": user_input}
    ]
    logger.info("📝 Messages prepared for Gemini API")

    # Generate bot response
    response = generate_groq_response(messages_for_groq)

    # Update chat history
    logger.info("💾 Updating chat history...")
    chat_data = {"user_msg": user_input, "assistant_msg": response}
    st.session_state.chat_histories[user_id].append(chat_data)

    # Display bot response
    st.markdown(f"**Bot:** {response}")


    # Add bot response to memory with better metadata
    logger.info("💾 Adding bot response to memory...")
    response_timestamp = datetime.now().isoformat()
    category = detect_category(user_input)
    bot_memory_data = [{"role": "assistant", "content": f"Bot response at {response_timestamp}: {response}"}]
    memory_manager.add(
        bot_memory_data,
        user_id=user_id,
        metadata={
            "type": "assistant_message",
            "timestamp": response_timestamp,
            "category": category
        }
    )

    conversation_memory = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response},
        {"role": "system", "content": f"metadata | timestamp={datetime.now().isoformat()} | category={category}"}
    ]
    memory_manager.add(conversation_memory, user_id=user_id, metadata={"type": "conversation", "category": category})
    logger.info("✅ Conversation pair added to memory")
    
    logger.info(f"✅ Conversation completed for user '{user_id}'")

    




