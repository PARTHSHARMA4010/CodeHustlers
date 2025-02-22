import React, { useState } from 'react';

function ProgressTracker() {
  const goals = { steps: 10000, water: 2.5, sleep: 8 }; // Mock goals
  const [progress, setProgress] = useState({ steps: 3000, water: 1.5, sleep: 6 });

  const getProgressPercent = (goal, achieved) => (achieved / goal) * 100;

  const handleInputChange = (event, category) => {
    const value = parseFloat(event.target.value) || 0; // Parse input as float; default to 0 if empty
    setProgress((prevProgress) => ({
      ...prevProgress,
      [category]: value,
    }));
  };

  return (
    <div className="bg-secondary p-8 rounded-lg shadow-md">
      <h2 className="text-lg font-bold text-primary mb-2">Progress Tracker</h2>
      <div className="space-y-4">
        {Object.keys(goals).map((goal) => (
          <div key={goal} className="flex items-center space-x-4">
            <label className="font-semibold capitalize w-1/3">{goal}</label>
            <div className="flex-grow bg-white rounded-md overflow-hidden">
              <div
                className="bg-primary h-2"
                style={{ width: `${getProgressPercent(goals[goal], progress[goal])}%` }}
              ></div>
            </div>
            <span className="text-gray-700">
              {progress[goal]} / {goals[goal]}
            </span>
            <input
              type="number"
              className="border border-gray-300 rounded px-2 py-1 w-20"
              placeholder="Enter value"
              value={progress[goal]}
              onChange={(event) => handleInputChange(event, goal)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export default ProgressTracker;
