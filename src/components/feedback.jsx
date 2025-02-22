import React, { useState } from 'react';

const FeedbackComponent = () => {
  const [rating, setRating] = useState(0);
  const [feedback, setFeedback] = useState("");

  const handleRatingClick = (index) => {
    setRating(index + 1);
  };

  const handleFeedbackChange = (event) => {
    setFeedback(event.target.value);
  };

  const handleSubmit = () => {
    alert(`Rating: ${rating}\nFeedback: ${feedback}`);
    // You can add functionality to submit the data to a server here.
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="w-96 h-auto bg-[#F7F6F2] rounded-lg p-6 shadow-lg text-center">
        <h2 className="text-lg font-bold text-gray-800 mb-2">We appreciate your feedback.</h2>
        <p className="text-sm text-gray-600 mb-4">
          We are always looking for ways to improve your experience. Please take a moment to evaluate and tell us what you think.
        </p>
        <div className="flex justify-center mb-4">
          {[...Array(5)].map((_, index) => (
            <span
              key={index}
              className={`text-2xl cursor-pointer ${index < rating ? 'text-yellow-400' : 'text-gray-300'}`}
              onClick={() => handleRatingClick(index)}
            >
              ★
            </span>
          ))}
        </div>
        <textarea
          className="w-full h-60 p-2 border border-gray-300 rounded-lg resize-none mb-4"
          placeholder="What can we do to improve your experience?"
          value={feedback}
          onChange={handleFeedbackChange}
        />
        <button 
          className="bg-green-700 text-white py-2 px-4 rounded-lg mt-6 hover:bg-green-800 transition"
          onClick={handleSubmit}
        >
          Submit My Feedback
        </button>
      </div>
    </div>
  );
};

export default FeedbackComponent;
