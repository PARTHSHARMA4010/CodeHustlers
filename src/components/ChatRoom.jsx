import React, { useState, useEffect } from "react";
import io from "socket.io-client";
import { motion } from "framer-motion";

const socket = io("http://localhost:3000");

const ChatApp = () => {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [message, setMessage] = useState("");
    const [messages, setMessages] = useState([]);
    const [joined, setJoined] = useState(false);
    const [isDoctor, setIsDoctor] = useState(false);

    useEffect(() => {
        const handleMessage = (data) => {
            setMessages((prevMessages) => [...prevMessages, data]);
        };
        
        socket.on("chat message", handleMessage);
        return () => {
            socket.off("chat message", handleMessage);
        };
    }, []);

    const handleJoin = () => {
        if (username.trim()) {
            if (password === "p") setIsDoctor(true);
            setJoined(true);
        }
    };

    const handleSendMessage = () => {
        if (message.trim()) {
            socket.emit("chat message", { user: username, message, isDoctor });
            setMessage("");
        }
    };

    const styles = {
        root: { fontFamily: "Inter, sans-serif", background: "#f4f4f4", display: "flex", justifyContent: "center", alignItems: "center", height: "100vh", margin: 0 },
        container: { width: "90%", maxWidth: "450px", background: "#fff", padding: "25px", borderRadius: "12px", boxShadow: "0px 4px 15px rgba(0, 0, 0, 0.1)" },
        header: { textAlign: "center", color: "#00796b", marginBottom: "20px", fontSize: "24px", fontWeight: "600" },
        input: { padding: "12px", border: "1px solid #ddd", borderRadius: "12px", outline: "none", fontSize: "14px", transition: "border-color 0.3s ease", width: "100%" },
        button: { padding: "12px", border: "none", background: "#00796b", color: "white", fontWeight: "bold", borderRadius: "12px", cursor: "pointer", transition: "background 0.3s ease, transform 0.2s ease" },
        buttonHover: { background: "#004d40", transform: "translateY(-2px)" },
        chat: { display: "flex", flexDirection: "column", gap: "15px" },
        messages: { maxHeight: "300px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "10px", padding: "15px", border: "1px solid #ddd", borderRadius: "12px", background: "#fafafa" },
        message: { padding: "10px", borderRadius: "12px", background: "#fff", boxShadow: "0px 4px 15px rgba(0, 0, 0, 0.1)" },
        doctorMessage: { background: "#00796b", color: "#ffffff" },
        inputContainer: { display: "flex", gap: "10px" },
    };

    return (
        <div style={styles.root}>
            <div style={styles.container}>
                <motion.h2 initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} style={styles.header}>
                    Anonymous Community Forum
                </motion.h2>

                {!joined ? (
                    <div style={{ display: "flex", flexDirection: "column", gap: "15px" }}>
                        <input type="text" placeholder="Enter display name" onChange={(e) => setUsername(e.target.value)} style={styles.input} />
                        <input type="password" placeholder="Enter password (if doctor)" onChange={(e) => setPassword(e.target.value)} style={styles.input} />
                        <button onClick={handleJoin} style={styles.button}>Join Chat</button>
                    </div>
                ) : (
                    <div style={styles.chat}>
                        <div style={styles.messages}>
                            {messages.map((msg, index) => (
                                <motion.div key={index} style={{ ...styles.message, ...(msg.isDoctor ? styles.doctorMessage : {}) }} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
                                    <strong>{msg.user}{msg.isDoctor ? " (Doctor)" : ""}:</strong> {msg.message}
                                </motion.div>
                            ))}
                        </div>
                        <div style={styles.inputContainer}>
                            <input type="text" placeholder="Type your message" value={message} onChange={(e) => setMessage(e.target.value)} style={styles.input} />
                            <button onClick={handleSendMessage} style={styles.button}>Send</button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ChatApp;
