import React, { useState } from 'react';
import Question from './Question';

const questions = [
  "Been upset because of something that happened unexpectedly?",
  "Felt that you were unable to control important things in your life?",
  "Felt nervous and 'stressed'?",
  "Felt underconfident about your ability to handle your personal problems?",
  "Felt that things were not going your way?",
];

const options = [
  { label: 'Never', emoji: '😊', color: 'bg-green-400', score: 1 },
  { label: 'Almost Never', emoji: '🙂', color: 'bg-orange-400', score: 2 },
  { label: 'Sometimes', emoji: '😶', color: 'bg-gray-400', score: 3 },
  { label: 'Fairly Often', emoji: '😐', color: 'bg-yellow-400', score: 4 },
  { label: 'Very Often', emoji: '😢', color: 'bg-red-400', score: 5 }
];

const getResultContent = (stressLevel) => {
  switch (stressLevel) {
    case "Highly Stressed":
      return {
        paragraph: "You are experiencing a high level of stress, which can significantly impact your overall well-being. It is crucial to take immediate action to manage this stress. We strongly recommend consulting a healthcare professional to discuss your symptoms and explore potential treatment options. Our virtual health camp, DocDial, provides an excellent platform for you to connect with medical experts and get the support you need.",
        buttonText: "DocDial",
        buttonLink: "/login-signup" // Replace with actual DocDial link
      };    
    case "Moderately Stressed":
      return {
        paragraph: "You are experiencing a moderate level of stress, which, if left unmanaged, can affect your health and daily life. It is advisable to consult with a healthcare provider to address your stress and prevent it from escalating. Our virtual health camp, DocDial, allows you to connect with medical professionals who can offer guidance and support tailored to your needs.",
        buttonText: "DocDial",
        buttonLink: "/login-signup" // Replace with actual DocDial link
      };
    case "Slightly Stressed":
      return {
        paragraph: "You are experiencing a low level of stress. While this is manageable, it's essential to adopt strategies to keep your stress levels in check and prevent them from rising. We recommend exploring Comfort Cloud, our self-help resource platform, which offers various tools and techniques to help you relax and stay calm. Additionally, using our Daily Planner can assist you in setting and achieving small goals, further reducing stress.",
        buttonText: "Comfort Cloud & Daily Planner",
        buttonLink: "/comfortcloud" // Replace with actual Comfort Cloud link
      };
    case "Stress is NOT for you!!":
      return {
        paragraph: "Congratulations! You are currently stress-free, which is fantastic for your mental and physical health. To maintain this state, we suggest utilizing Comfort Cloud, our self-help resource platform. Here, you will find a variety of resources to help you continue practicing good mental health habits and ensure you stay stress-free.",
        buttonText: "Comfort Cloud",
        buttonLink: "/comfortcloud" // Replace with actual Comfort Cloud link
      };
    default:
      return { paragraph: "", buttonText: "", buttonLink: "" };
  }
};

const Questionnaire = () => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedOptions, setSelectedOptions] = useState(Array(questions.length).fill(null));
  const [result, setResult] = useState(null);

  const handleNext = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const handlePrev = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleSubmit = () => {
    const score = selectedOptions.reduce((acc, option) => {
      const selectedOption = options.find(o => o.label === option);
      return acc + (selectedOption ? selectedOption.score : 0);
    }, 0);

    let stressLevel;
    let imageUrl;

    if (score >= 15) {
      stressLevel = "Highly Stressed";
      imageUrl = 'highlystress-removebg-preview.png'; // Replace with actual image path
    } else if (score > 10) {
      stressLevel = "Moderately Stressed";
      imageUrl = 'moderately-removebg-preview.png'; // Replace with actual image path
    } else if (score > 8) {
      stressLevel = "Slightly Stressed";
      imageUrl = 'less-removebg-preview.png'; // Replace with actual image path
    } else {
      stressLevel = "Stress is NOT for you!!";
      imageUrl = 'nostress-removebg-preview.png'; // Replace with actual image path
    }

    setResult({ stressLevel, imageUrl });
  };

  const setSelectedOption = (option) => {
    const newSelectedOptions = [...selectedOptions];
    newSelectedOptions[currentQuestionIndex] = option;
    setSelectedOptions(newSelectedOptions);
  };

  if (result) {
    const { paragraph, buttonText, buttonLink } = getResultContent(result.stressLevel);

    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-background">
        <h2 className="text-2xl font-bold mb-6 text-primary">Your Stress Level: {result.stressLevel}</h2>
        <img src={result.imageUrl} alt={result.stressLevel} className="w-128 h-64 mb-4" />
        <p className="text-lg mb-4 text-center px-8" style={{ color: '#006953' }}>{paragraph}</p>
        <a href={buttonLink} className="bg-primary text-white py-2 px-4 rounded">
          {buttonText}
        </a>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-background">
      <h1 className="text-3xl font-bold mb-6 text-primary">Stress Tracker: Answer the Questions Below to Measure Your Stress</h1>
      <Question
        question={questions[currentQuestionIndex]}
        options={options}
        selectedOption={selectedOptions[currentQuestionIndex]}
        setSelectedOption={setSelectedOption}
      />
      <div className="mt-2 flex justify-between w-full max-w-xs">
        <button
          onClick={handlePrev}
          className="bg-secondary text-primary py-2 px-8 rounded disabled:opacity-50"
          disabled={currentQuestionIndex === 0}
        >
          Previous
        </button>
        {currentQuestionIndex === questions.length - 1 ? (
          <button
            onClick={handleSubmit}
            className="bg-primary text-white py-2 px-8 rounded"
          >
            Submit
          </button>
        ) : (
          <button
            onClick={handleNext}
            className="bg-primary text-white py-2 px-4 rounded"
            disabled={!selectedOptions[currentQuestionIndex]}
          >
            Next
          </button>
        )}
      </div>
    </div>
  );
};

export default Questionnaire;