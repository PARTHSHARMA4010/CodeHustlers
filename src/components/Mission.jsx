import React, { useContext } from 'react';
import { LanguageContext } from '../LanguageContext';

const translations = {
  en: {
    title: 'At HealConnect, We believe mental wellness should be accessible to everyone, anytime.',
    mission: 'Our mission is to empower individuals to take control of their mental health with a range of tools designed to provide support, guidance, and personalized care.',
    button: 'Get Started'
  },
  hi: {
    title: 'HealConnect में, हम मानते हैं कि मानसिक स्वास्थ्य सभी के लिए सुलभ होना चाहिए, किसी भी समय।',
    mission: 'हमारा मिशन लोगों को उनकी मानसिक स्वास्थ्य का नियंत्रण लेने के लिए प्रेरित करना है, जिसमें सहायता, मार्गदर्शन और व्यक्तिगत देखभाल के लिए उपकरण शामिल हैं।',
    button: 'शुरू करें'
  },
  mr: {
    title: 'HealConnect येथे, आम्ही विश्वास ठेवतो की मानसिक स्वास्थ्य सर्वांसाठी कोणत्याही वेळी सुलभ असावे.',
    mission: 'आमचे ध्येय म्हणजे लोकांना त्यांच्या मानसिक आरोग्यावर नियंत्रण ठेवण्यासाठी प्रेरित करणे, ज्यात समर्थन, मार्गदर्शन आणि वैयक्तिक काळजीसाठी साधने समाविष्ट आहेत.',
    button: 'सुरू करा'
  }
};

const Mission = () => {
  const { language } = useContext(LanguageContext);
  const { title, mission, button } = translations[language];

  return (
    <div className="w-3/5 p-20 ml-5 mt-50 rounded-none mb-12" style={{ backgroundColor: '#F7F6F2' }}>
      <h2 className="text-3xl text-black font-serif font-playfair">{title}</h2>
      <p className="mt-5 mb-5 text-gray-700 w-3/4">{mission}</p>
      <button className="bg-green-800 text-white px-4 py-2 rounded-full text-lg cursor-pointer">{button}</button>
    </div>
  );
};

export default Mission;
