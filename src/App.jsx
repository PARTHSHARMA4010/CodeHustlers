// Import necessary modules
import React, { useEffect } from 'react';
import { LanguageProvider } from './LanguageContext';
import Navbar from './components/Navbar';
import MentalHealthSupport from './components/home1';
import StatisticsImage from './components/StatisticsImage';
import Mission from './components/Mission';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import Calendar from './components/Calender';
import Part3 from './components/part3';
import Mood from './components/mood';
import HealConnectPage from './components/healconnect';
import ChatBot from './components/ChatBot';
import Login from './components/Login';
import FeedbackComponent from './components/feedback';
import MainLayout from './components/MainLayout';
import Question from './components/Question';
import Questionnaire from './components/Quetionerrie';
import Parth from './components/Comfort';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import BuyNowModal from './components/paymentGateway';
import ChatApp from './components/ChatRoom';

// Component to handle redirection
const RedirectToGoogle = () => {
    useEffect(() => {
        window.location.href = 'https://hack-the-chain-ayurveda.vercel.app/';
    }, []);

    return null; // Render nothing as this component is just for redirection
};

function App() {
    return (
        <LanguageProvider>
            <Router>
                <div style={{ backgroundColor: '#F7F6F2' }}>
                    <Navbar />
                    <Routes>
                        {/* Define routes for components */}
                        <Route path="/" element={
                            <>
                                <MentalHealthSupport />
                                <section className="flex justify-between mt-8">
                                    <StatisticsImage />
                                    <Mission />
                                </section>
                                <Mood />
                                <Part3 />
                            </>
                        } />
                        <Route path="/calendar" element={<Calendar />} />
                        <Route path="/chatbot" element={<HealConnectPage />} />
                        <Route path="/chat" element={<ChatBot />} />
                        <Route path="/feedback" element={<FeedbackComponent />} />
                        <Route path="/planner" element={<MainLayout />} />
                        <Route path="/symptom-checker" element={<Questionnaire />} />
                        <Route path="/login-signup" element={<Login />} />
                        <Route path="/interactive-healing" element={<Parth />} />
                        <Route path="/payment" element={<BuyNowModal />} />
                        {/* Redirect to Google on /parth */}
                        <Route path="/ayurveda" element={<RedirectToGoogle />} />
                        {/* <Route path="/room" element={<ChatApp />} /> */}
                        <Route path="/community" element={<ChatApp />} />

                    </Routes>
                </div>
            </Router>
        </LanguageProvider>
    );
}

export default App;