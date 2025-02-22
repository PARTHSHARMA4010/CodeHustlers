import React from "react";
import Chatbot from "react-chatbot-kit";
import "react-chatbot-kit/build/main.css";
import MessageParser from "./MessageParser.jsx";
import { createChatBotMessage } from "react-chatbot-kit";
import ActionProvider from "./ActionProvider.jsx";
import { useNavigate } from "react-router-dom";

function ChatBot(props) {
  const navigate = useNavigate();

  // Define color theme
  const primaryColor = "#006953";
  const secondaryColor = "#A9C7B8";
  const backgroundColor = "#F7F6F2";

  // Chatbot Header Styles
  const chatHeader = {
    background: `linear-gradient(45deg, ${primaryColor}, ${secondaryColor})`,
    borderRadius: "15px 15px 0 0",
    fontFamily: "Roboto, sans-serif",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    fontSize: "1.2rem",
    color: "#FFFFFF",
    padding: "10px 20px",
    fontWeight: "600",
  };

  const closeButton = {
    background: "transparent",
    color: "#FFFFFF",
    fontSize: "1.2rem",
    border: "none",
    cursor: "pointer",
  };

  const handleClick = () => {
    props.setChatBot(false);
    navigate("/");
  };

  const botName = "Your Assistant";

  // Chatbot configuration
  const config = {
    initialMessages: [
      createChatBotMessage("Hello! How can I assist you today with your mental health?"),
    ],
    botName: botName,
    customComponents: {
      header: () => (
        <div style={chatHeader}>
          {botName}
          <button style={closeButton} onClick={handleClick}>✖️</button>
        </div>
      ),
    },
    customStyles: {
      botMessageBox: {
        backgroundColor: `linear-gradient(45deg, ${secondaryColor}, ${primaryColor})`,
        color: "#FFFFFF",
        borderRadius: "12px",
        padding: "10px 15px",
        maxWidth: "95%",
        margin: "10px",
        boxShadow: "0 2px 10px rgba(0, 0, 0, 0.1)",
      },
      userMessageBox: {
        backgroundColor: primaryColor,
        color: "#FFFFFF",
        borderRadius: "12px",
        padding: "10px 15px",
        maxWidth: "750px",
        margin: "10px",
        alignSelf: "flex-end",
        boxShadow: "0 2px 10px rgba(0, 0, 0, 0.2)",
      },
      chatButton: {
        background: `linear-gradient(45deg, ${primaryColor}, ${secondaryColor})`,
        color: "#FFFFFF",
        borderRadius: "50%",
        fontSize: "1.5rem",
        padding: "10px",
        border: "none",
        cursor: "pointer",
        boxShadow: "0 4px 12px rgba(0, 0, 0, 0.2)",
        transition: "background 0.3s ease-in-out",
      },
    },
  };

  return (
    <div
      style={{
        height: "100vh",
        backgroundColor: backgroundColor,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: "275px",
          height: "80%",
          maxHeight: "520px",
          borderRadius: "20px",
          overflow: "hidden",
          boxShadow: "0 5px 30px rgba(0, 0, 0, 0.1)",
          backgroundColor: "#FFFFFF",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Chatbot
          config={config}
          messageParser={MessageParser}
          actionProvider={ActionProvider}
        />
      </div>

      <button
        style={{
          position: "absolute",
          top: "20px",
          right: "20px",
          backgroundColor: primaryColor,
          color: "#FFFFFF",
          fontSize: "1.5rem",
          borderRadius: "50%",
          padding: "10px",
          border: "none",
          cursor: "pointer",
          boxShadow: "0 2px 10px rgba(0, 0, 0, 0.2)",
        }}
        onClick={handleClick}
      >
        ✖️
      </button>
    </div>
  );
}

export default ChatBot;
