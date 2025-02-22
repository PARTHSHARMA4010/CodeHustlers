import React, { useContext } from 'react';
import { LanguageContext } from '../LanguageContext';
import { useNavigate } from 'react-router-dom';

const translations = {
  en: {
    title: 'Mental health support for everyone.',
    description: 'Empowering your journey to mental well-being with accessible tools, supportive communities, expert guidance, and personalized resources that help you navigate life\'s challenges, build resilience, and foster a sense of hope and strength.',
    quote: 'Taking care of your mental health is not a luxury, it\'s a necessity. Prioritize your well-being.',
    button: 'Test Your Stress Level',
  },
  hi: {
    title: 'हर किसी के लिए मानसिक स्वास्थ्य सहायता।',
    description: 'सुलभ उपकरणों, सहायक समुदायों, विशेषज्ञ मार्गदर्शन और व्यक्तिगत संसाधनों के साथ आपको जीवन की चुनौतियों का सामना करने, लचीलापन बनाने और आशा व शक्ति की भावना को बढ़ावा देने में सहायता प्रदान करना।',
    quote: 'अपने मानसिक स्वास्थ्य का ख्याल रखना एक विलासिता नहीं, बल्कि एक आवश्यकता है। अपनी भलाई को प्राथमिकता दें।',
    button: 'शुरू करें',
  },
  mr: {
    title: 'प्रत्येकासाठी मानसिक आरोग्य सहाय्य.',
    description: 'सुलभ साधनं, सहायक समुदाय, तज्ञ मार्गदर्शन आणि वैयक्तिकृत संसाधनांसह तुमच्या मानसिक आरोग्याच्या प्रवासाला शक्ती देणे, जीवनातील आव्हाने हाताळणे, लवचिकता निर्माण करणे आणि आशा व सामर्थ्याची भावना वाढवणे.',
    quote: 'तुमच्या मानसिक आरोग्याची काळजी घेणे हे विलास नाही, तर गरज आहे. तुमच्या भल्याला प्राधान्य द्या.',
    button: 'सुरु करा',
  },
};

const MentalHealthSupport = () => {
  const { language } = useContext(LanguageContext);
  const { title, description, quote, button } = translations[language];
  const navigate = useNavigate();

  const handleButtonClick = () => {
    navigate('/symptom-checker');
  };

  return (
    <div className="flex flex-col min-h-screen bg-[#F7F6F2] p-10 space-y-10">
      <div className="flex flex-col md:flex-row items-center justify-center space-y-10 md:space-y-0 md:space-x-10">
        <div className="md:w-1/2 p-10 flex flex-col justify-center text-center md:text-left">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-6">{title}</h1>
          <p className="text-gray-600 mb-8 mt-4 leading-relaxed">{description}</p>
          <button
            onClick={handleButtonClick}
            className="bg-[#006953] hover:bg-[#004c43] text-white font-semibold py-3 px-6 rounded-lg shadow-md transform transition-transform duration-300 hover:scale-105 mb-16 mt-4"
          >
            {button}
          </button>
        </div>
        <div className="md:w-1/2 h-80 md:h-auto">
          <img
            src="https://images.squarespace-cdn.com/content/v1/5f3b3b1a9b02081fa4eff770/c39cd82f-8e62-4268-a4b3-a06710b98593/iStock-1189088293.jpg?format=2500w"
            alt="Family Mental Health Support"
            className="object-cover h-full w-full rounded-3xl shadow-lg transition-shadow duration-300 hover:shadow-xl"
          />
        </div>
      </div>

      <div className="w-full bg-[#004c43] py-6 text-center mt-auto">
        <p className="text-white text-xl font-semibold">{quote}</p>
      </div>
    </div>
  );
};

export default MentalHealthSupport;
