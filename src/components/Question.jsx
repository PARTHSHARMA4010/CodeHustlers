import React from 'react';

const Question = ({ question, options, selectedOption, setSelectedOption }) => {
  return (
    <div className="flex flex-col items-center justify-center bg-background p-4">
      <h2 className="text-2xl font-bold mb-6 text-black">{question}</h2>
      <div className="w-full max-w-xs">
        {options.map((option, index) => (
          <div key={index} className="flex items-center mb-3"> {/* Adjusted margin between options */}
            <input
              id={`option-${index}`}
              type="radio"
              name="option"
              value={option.label}
              checked={selectedOption === option.label}
              onChange={() => setSelectedOption(option.label)}
              className="mr-4 scale-125"
            />
            <label htmlFor={`option-${index}`} className="flex items-center cursor-pointer">
              <span className={`w-12 h-12 flex items-center justify-center rounded-full ${option.color} text-white mr-3 text-3xl`}>
                {option.emoji}
              </span>
              <span className="text-lg text-black">{option.label}</span>
            </label>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Question;
