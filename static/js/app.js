const API_BASE = '/api';

const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const clearButton = document.getElementById('clearButton');
const ingestButton = document.getElementById('ingestButton');
const status = document.getElementById('status');

// Send message on button click
sendButton.addEventListener('click', sendMessage);

// Send message on Enter key
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Clear chat history
clearButton.addEventListener('click', () => {
    chatMessages.innerHTML = `
        <div class="message bot-message">
            <div class="message-content">
                <p>Chat history cleared. How can I help you today?</p>
            </div>
        </div>
    `;
    fetch(`${API_BASE}/clear`, { method: 'POST' })
        .then(() => updateStatus('Chat history cleared'));
});

// Ingest PDFs
ingestButton.addEventListener('click', async () => {
    ingestButton.disabled = true;
    updateStatus('Ingesting PDFs...');
    
    try {
        const response = await fetch(`${API_BASE}/ingest`, { method: 'POST' });
        const data = await response.json();
        
        if (response.ok) {
            updateStatus(`✓ ${data.message} - ${data.chunks_processed} chunks processed`);
        } else {
            updateStatus(`✗ Error: ${data.detail || 'Unknown error'}`);
        }
    } catch (error) {
        updateStatus(`✗ Error: ${error.message}`);
    } finally {
        ingestButton.disabled = false;
    }
});

// Send message function
async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addMessage(message, 'user');
    messageInput.value = '';
    sendButton.disabled = true;
    
    // Show loading indicator
    const loadingId = addLoadingMessage();
    
    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message }),
        });
        
        const data = await response.json();
        
        // Remove loading indicator
        removeLoadingMessage(loadingId);
        
        if (response.ok) {
            // Debug: log sources
            console.log('Sources received:', data.sources);
            
            // Add bot response with sources
            addMessage(data.response, 'bot', data.sources);
            
            // Show context info if available (but don't override sources status)
            if (data.graph_context && (!data.sources || data.sources.length === 0)) {
                updateStatus('✓ Used graph context from Neo4j');
            }
            
            // Sources are now permanently displayed in message bubble, 
            // so we don't need to show them in status bar
            if (data.sources && data.sources.length > 0) {
                updateStatus(`✓ Found ${data.sources.length} source(s) - see message below`);
            }
        } else {
            addMessage(`Error: ${data.detail || 'Unknown error occurred'}`, 'bot');
            updateStatus('✗ Error occurred');
        }
    } catch (error) {
        removeLoadingMessage(loadingId);
        addMessage(`Error: ${error.message}`, 'bot');
        updateStatus('✗ Connection error');
    } finally {
        sendButton.disabled = false;
        messageInput.focus();
    }
}

// Add message to chat
function addMessage(text, type, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Split text by newlines and create paragraphs
    const paragraphs = text.split('\n').filter(p => p.trim());
    paragraphs.forEach(p => {
        const pTag = document.createElement('p');
        pTag.textContent = p;
        contentDiv.appendChild(pTag);
    });
    
    // Add sources component for bot messages
    if (type === 'bot' && sources && sources.length > 0) {
        console.log('Adding sources to message:', sources);
        
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        
        const sourcesLabel = document.createElement('span');
        sourcesLabel.className = 'sources-label';
        sourcesLabel.innerHTML = '<i class="fas fa-book"></i> Sources:';
        sourcesDiv.appendChild(sourcesLabel);
        
        const sourcesList = document.createElement('div');
        sourcesList.className = 'sources-list';
        
        sources.forEach(source => {
            const sourceItem = document.createElement('span');
            sourceItem.className = 'source-item';
            
            // Handle different source types
            if (source.type === 'graph') {
                // Graph citation - show plant name and related info
                let citationText = source.name || source.source || 'Knowledge Graph';
                if (source.illnesses && source.illnesses.length > 0) {
                    citationText += ` (treats: ${source.illnesses.join(', ')})`;
                } else if (source.symptoms && source.symptoms.length > 0) {
                    citationText += ` (symptoms: ${source.symptoms.join(', ')})`;
                }
                sourceItem.textContent = citationText;
                sourceItem.title = `From Knowledge Graph: ${source.name}`;
            } else {
                // PDF source - show filename and page
                const pageNum = source.page || source.page_number || '?';
                sourceItem.textContent = `${source.filename}, page ${pageNum}`;
                sourceItem.title = `From PDF: ${source.filename}, page ${pageNum}`;
            }
            
            sourcesList.appendChild(sourceItem);
        });
        
        sourcesDiv.appendChild(sourcesList);
        contentDiv.appendChild(sourcesDiv);
        
        console.log('Sources component added to message');
    } else if (type === 'bot') {
        console.log('No sources provided or empty sources array');
    }
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Add loading message
function addLoadingMessage() {
    const loadingId = 'loading-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.id = loadingId;
    messageDiv.className = 'message bot-message';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<div class="loading"></div> Thinking...';
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return loadingId;
}

// Remove loading message
function removeLoadingMessage(loadingId) {
    const loadingElement = document.getElementById(loadingId);
    if (loadingElement) {
        loadingElement.remove();
    }
}

// Update status
function updateStatus(text) {
    status.textContent = text;
    setTimeout(() => {
        status.textContent = '';
    }, 5000);
}

// Check health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        console.log("data", data);
        
        if (data.neo4j && data.weaviate && data.chatbot) {
            console.log("chatbot", data.chatbot);
            console.log("neo4j", data.neo4j);
            console.log("weaviate", data.weaviate);
            updateStatus('✓ All systems ready');
        } else {
            const issues = [];
            if (!data.neo4j) issues.push('Neo4j');
            if (!data.weaviate) issues.push('Weaviate');
            if (!data.chatbot) issues.push('Chatbot');
            updateStatus(`⚠ Missing: ${issues.join(', ')}`);
        }
    } catch (error) {
        updateStatus('⚠ Could not check system status');
    }
});

