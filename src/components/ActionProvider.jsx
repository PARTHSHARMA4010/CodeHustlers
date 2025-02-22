import React from 'react';
import { GoogleGenerativeAI } from "@google/generative-ai";

// Hardcoded API key
console.log(import.meta.env.API_KEY)
const genAI = new GoogleGenerativeAI(import.meta.env.VITE_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-pro" });

const ActionProvider = ({ createChatBotMessage, setState, children }) => {
    const geminiAction = async(query) => {
        let message = "";
        try {
            const result = await model.generateContent(query);
            const response = result.response;
            const text = response.text();
            message = createChatBotMessage(text);
        } catch (error) {
            console.log("Gemini Error: ", error);
            message = createChatBotMessage("Try after some time");
        }
        updateState(message);
    }

    const updateState = (message, checker) => {
        setState((prev) => ({
            ...prev,
            messages: [...prev.messages, message]
        }))
    }

    return (
        <div>
            {React.Children.map(children, (child) => {
                return React.cloneElement(child, {
                    actions: {
                        geminiAction
                    },
                });
            })}
        </div>
    );
};

export default ActionProvider;
