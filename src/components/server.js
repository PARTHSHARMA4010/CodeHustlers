import express from "express";
import { createServer } from "http";
import { Server } from "socket.io";
import cors from "cors";

const app = express();
const server = createServer(app);
const io = new Server(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

app.use(cors());

io.on("connection", (socket) => {
    console.log("A user connected");

    socket.on("chat message", (data) => {
        if (data.isDoctor) {
            console.log("👨‍⚕️ Doctor Message Received:", data);
        } else {
            console.log("💬 User Message Received:", data);
        }

        io.emit("chat message", data); // Broadcast to all clients
    });

    socket.on("disconnect", () => {
        console.log("A user disconnected");
    });
});

server.listen(3000, () => {
    console.log("🚀 Server running on http://localhost:3000");
});
