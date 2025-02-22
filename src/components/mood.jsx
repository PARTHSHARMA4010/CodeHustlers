import React, { useState } from 'react';

const moods = [
  { label: 'Sad', emoji: '😢', color: 'bg-red-400' },
  { label: 'Neutral', emoji: '😐', color: 'bg-yellow-400' },
  { label: 'Happy', emoji: '😊', color: 'bg-green-400' },
];

function Mood() {
  const [selectedMood, setSelectedMood] = useState(moods[1]);

  return (
    <div className="flex flex-col items-center justify-center m-12 p-12 bg-[#F7F6F2]">
      <h1 className="text-2xl font-bold mb-4">How would you describe your mood?</h1>
      <div className="text-6xl">{selectedMood.emoji}</div>
      <p className="text-lg mt-2">I Feel {selectedMood.label}.</p>
      <div className="flex mt-4">
        {moods.map((mood, index) => (
          <div
            key={index}
            className={`${mood.color} w-16 h-16 flex items-center justify-center rounded-full mx-2 cursor-pointer ${selectedMood.label === mood.label ? 'opacity-100' : 'opacity-50'}`}
            onClick={() => setSelectedMood(mood)}
          >
            <span className="text-3xl">{mood.emoji}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Mood;
