from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from complaint_x_bot.retrieval_generation import generation
from complaint_x_bot.data_ingestion import data_ingestion
from hallucination_detection import HallucinationDetector
import uuid
import json
from datetime import datetime

vstore = data_ingestion("done")
chain = generation(vstore)
HALLUCINATION_THRESHOLD = 0.2
hallucination_detector = HallucinationDetector(threshold=HALLUCINATION_THRESHOLD)
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

chat_sessions = {}
active_sessions = {}  
agent_connections = {} 
waiting_customers = [] 
agents_online = []  


@app.route("/")
def index():
    """Customer chat interface"""
    return render_template("index.html", HALLUCINATION_THRESHOLD=HALLUCINATION_THRESHOLD)

@app.route("/agent")
def agent():
    """Agent interface"""
    return render_template("agent.html")


@app.route("/get", methods=["POST", "GET"])
def chat():
    """Handle AI responses for customer queries"""
    if request.method == "POST":
        msg = request.form["msg"]
        session_id = request.form.get("session_id", "default_session")
        input_query = msg
        if session_id in agent_connections and agent_connections[session_id]["active"]:
            return json.dumps({
                "answer": "This conversation is currently handled by an agent. Please continue there.",
                "is_hallucination": False,
                "hallucination_score": 1.0,
                "threshold": HALLUCINATION_THRESHOLD
            })
        retrieved_docs = vstore.similarity_search(input_query)
        retrieved_texts = [doc.page_content for doc in retrieved_docs]
        result = chain.invoke(
            {"input": input_query},
            config={
                "configurable": {"session_id": session_id}
            },
        )["answer"]
        is_hallucination, similarity_score = hallucination_detector.is_hallucination(
            retrieved_texts, result
        )
        print(f"Query: {input_query}")
        print(f"Similarity score: {similarity_score}")
        print(f"Threshold: {HALLUCINATION_THRESHOLD}")
        print(f"Is hallucination: {is_hallucination}")

        
        final_verdict = True

        if similarity_score > 0.1:
            final_verdict = True
        else:
            final_verdict = False


        response = {
            "answer": result,
            "is_hallucination": final_verdict , 
            "hallucination_score": similarity_score,
            "threshold": HALLUCINATION_THRESHOLD
        }
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_sessions[session_id].append({
            'sender': 'user',
            'message': input_query,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        chat_sessions[session_id].append({
            'sender': 'bot',
            'message': result,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'is_hallucination': is_hallucination
        })
        
        return json.dumps(response)



@app.route("/feedback", methods=["POST"])
def handle_feedback():
    """Handle customer feedback and agent connection requests"""
    if request.method == "POST":
        message_id = request.form.get("messageId")
        is_satisfied = request.form.get("isSatisfied") == "true"
        session_id = request.form.get("session_id", "default_session")
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        feedback = {
            "message_id": message_id,
            "is_satisfied": is_satisfied,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        if not is_satisfied:
            if session_id not in waiting_customers:
                waiting_customers.append(session_id)
                chat_sessions[session_id].append({
                    "sender": "system",
                    "message": "Customer requested agent assistance",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                
                socketio.emit('new_waiting_customer', {
                    'session_id': session_id,
                    'chat_history': chat_sessions[session_id][-5:] if len(chat_sessions[session_id]) > 5 else chat_sessions[session_id]
                }, room='agents_room')
        
        return jsonify({
            "success": True, 
            "session_id": session_id,
            "waiting_position": waiting_customers.index(session_id) + 1 if session_id in waiting_customers else 0
        })

@app.route("/get_waiting_customers", methods=["GET"])
def get_waiting_customers():
    """Get list of customers waiting for agent assistance"""
    waiting_data = []
    
    for session_id in waiting_customers:
        
        last_messages = []
        if session_id in chat_sessions:
        
            user_messages = [msg for msg in chat_sessions[session_id] if msg['sender'] == 'user']
            last_messages = user_messages[-3:] if len(user_messages) > 3 else user_messages
        
        waiting_data.append({
            'session_id': session_id,
            'waiting_since': datetime.now().strftime("%H:%M:%S"),  
            'last_messages': last_messages
        })
    
    return jsonify({"waiting_customers": waiting_data})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print("Client disconnected")
    
@socketio.on('agent_login')
def handle_agent_login(data):
    """Handle agent login"""
    agent_id = data.get('agent_id')
    if agent_id:

        if agent_id not in agents_online:
            agents_online.append(agent_id)
        

        join_room('agents_room')
        

        emit('waiting_customers_update', {
            'waiting_customers': waiting_customers,
            'count': len(waiting_customers)
        })
        
        print(f"Agent {agent_id} logged in")

@socketio.on('agent_accept_customer')
def handle_agent_accept_customer(data):
    """Handle agent accepting a customer from waiting list"""
    agent_id = data.get('agent_id')
    session_id = data.get('session_id')
    
    if not agent_id or not session_id:
        emit('error', {'message': 'Missing agent_id or session_id'})
        return
    
    
    if session_id in waiting_customers:
        waiting_customers.remove(session_id)
    
    
    agent_connections[session_id] = {
        'agent_id': agent_id,
        'started_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'active': True
    }
    
    
    join_room(session_id)
    
    
    history = chat_sessions.get(session_id, [])
    emit('chat_history', {'history': history}, room=session_id)
    
    emit('agent_connected', {
        'agent_id': agent_id,
        'message': 'An agent has connected to assist you'
    }, room=session_id)
    
    emit('waiting_customers_update', {
        'waiting_customers': waiting_customers,
        'count': len(waiting_customers)
    }, room='agents_room')
    
    print(f"Agent {agent_id} accepted customer {session_id}")

@socketio.on('join_session')
def handle_join_session(data):
    """Handle customer or agent joining a specific chat session"""
    session_id = data.get('session_id')
    is_agent = data.get('is_agent', False)
    
    if not session_id:
        return
    
    
    join_room(session_id)
    
    if not is_agent and session_id in agent_connections and agent_connections[session_id]['active']:
        
        emit('user_status', {
            'status': 'connected',
            'message': 'Customer has connected to the chat'
        }, room=session_id)
    
    history = chat_sessions.get(session_id, [])
    emit('chat_history', {'history': history})
    
    if is_agent and session_id in chat_sessions and len(chat_sessions[session_id]) > 0:
        emit('chat_status', {
            'has_history': True,
            'message_count': len(chat_sessions[session_id])
        })

@socketio.on('send_message')
def handle_message(data):
    """Handle message sending between customer and agent"""
    session_id = data.get('session_id')
    sender = data.get('sender')  
    message = data.get('message')
    
    if not session_id or not sender or not message:
        return
    
    
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    msg_obj = {
        'sender': sender,
        'message': message,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    
    chat_sessions[session_id].append(msg_obj)
    
    
    emit('new_message', msg_obj, room=session_id)
    
    if sender == 'user' and session_id not in agent_connections:
        active_sessions[session_id] = {'ai_connected': True}

@socketio.on('agent_end_conversation')
def handle_agent_end_conversation(data):
    """Handle agent ending a conversation with a customer"""
    session_id = data.get('session_id')
    agent_id = data.get('agent_id')
    
    if not session_id or not agent_id:
        return
    
    if session_id in agent_connections and agent_connections[session_id]['agent_id'] == agent_id:
        agent_connections[session_id]['active'] = False
        agent_connections[session_id]['ended_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        

        if session_id in chat_sessions:
            chat_sessions[session_id].append({
                'sender': 'system',
                'message': 'Agent has ended the conversation. You are now chatting with the AI assistant again.',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        
        emit('agent_disconnected', {
            'message': 'Agent has ended the conversation. You are now chatting with the AI assistant again.'
        }, room=session_id)
        
        
        leave_room(session_id)
        
        print(f"Agent {agent_id} ended conversation with customer {session_id}")

@socketio.on('get_agent_status')
def handle_get_agent_status(data):
    """Check if a customer is connected to an agent"""
    session_id = data.get('session_id')
    
    if not session_id:
        emit('agent_status', {'connected': False})
        return
    

    is_connected = (
        session_id in agent_connections and 
        agent_connections[session_id]['active']
    )
    
    emit('agent_status', {
        'connected': is_connected,
        'agent_id': agent_connections[session_id]['agent_id'] if is_connected else None
    })

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)