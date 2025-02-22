import React, { useState } from 'react';

function CustomHealthGoals() {
  const [goals, setGoals] = useState({
    steps: '',
    water: '',
    sleep: '',
    customGoal: '',
  });
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [customGoalsList, setCustomGoalsList] = useState([]);
  const [goalsSet, setGoalsSet] = useState(false); // Track if goals are set

  const updateGoal = (type, value) => {
    setGoals({ ...goals, [type]: value });
  };

  const addCustomGoal = () => {
    if (goals.customGoal) {
      setCustomGoalsList([...customGoalsList, { goal: goals.customGoal, completed: false }]);
      setGoals({ ...goals, customGoal: '' });
    }
  };

  const toggleCompletion = (index) => {
    const updatedGoalsList = [...customGoalsList];
    updatedGoalsList[index].completed = !updatedGoalsList[index].completed;
    setCustomGoalsList(updatedGoalsList);
  };

  const handleModalOpen = () => {
    setIsModalOpen(true);
  };

  const handleModalClose = () => {
    setIsModalOpen(false);
    setGoalsSet(true); // Mark the goals as set once the modal is closed
  };

  const handleUpdateGoals = () => {
    setGoalsSet(false); // Reset goalsSet to false to allow editing
    setIsModalOpen(true); // Open modal to update goals
  };

  return (
    <div className="bg-secondary p-8 rounded-lg shadow-md text-gray-800">
      <h2 className="text-xl font-semibold text-primary mb-4">Custom Health Goals</h2>

      {!goalsSet ? (
        <div>
          {/* Set Your Goals Section */}
          <button
            onClick={handleModalOpen}
            className="bg-primary text-white px-4 py-2 rounded-lg mb-6 shadow hover:bg-primary-dark"
          >
            Set Your Goals
          </button>
          <p className="text-md text-gray-600 mb-4 leading-relaxed">
            Set your goals today and take the first step towards a healthier, stronger, and more energetic you!
            Every small effort counts, and with each step you take, you're building a better version of yourself.
          </p>
        </div>
      ) : (
        <div>
          {/* Goals Set Display Section */}
          <div className="mt-4">
            <h3 className="text-lg font-bold mb-3">Goals Set:</h3>
            <ul className="list-disc pl-5 text-gray-700 space-y-2">
              <li>Steps Goal: {goals.steps}</li>
              <li>Water Intake: {goals.water} liters</li>
              <li>Sleep Hours: {goals.sleep} hours</li>
              {customGoalsList.map((goal, index) => (
                <li key={index} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={goal.completed}
                    onChange={() => toggleCompletion(index)}
                    className="mr-2"
                  />
                  <span className={goal.completed ? 'line-through text-gray-500' : ''}>{goal.goal}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Update Your Goals Button */}
          <button
            onClick={handleUpdateGoals}
            className="bg-primary text-black px-4 py-2 rounded-lg mt-6 shadow hover:bg-primary-dark"
          >
            Update Your Goals
          </button>
        </div>
      )}

      {/* Modal to Set/Update Goals */}
      {isModalOpen && (
        <div className="fixed top-0 left-0 w-full h-full bg-black bg-opacity-50 flex justify-center items-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-lg w-11/12 max-w-md relative">
            <button
              onClick={handleModalClose}
              className="absolute top-2 right-2 text-gray-500 text-lg"
            >
              &times;
            </button>

            <h3 className="text-lg font-bold mb-4 text-primary">Set Your Health Goals</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-gray-700 font-semibold">Steps Goal:</label>
                <input
                  type="number"
                  className="w-full p-2 mt-1 rounded-md border border-gray-300 focus:border-primary"
                  placeholder="Enter steps goal"
                  value={goals.steps}
                  onChange={(e) => updateGoal('steps', e.target.value)}
                />
              </div>

              <div>
                <label className="block text-gray-700 font-semibold">Water Intake (liters):</label>
                <input
                  type="number"
                  className="w-full p-2 mt-1 rounded-md border border-gray-300 focus:border-primary"
                  placeholder="Enter water intake goal"
                  value={goals.water}
                  onChange={(e) => updateGoal('water', e.target.value)}
                />
              </div>

              <div>
                <label className="block text-gray-700 font-semibold">Sleep Hours:</label>
                <input
                  type="number"
                  className="w-full p-2 mt-1 rounded-md border border-gray-300 focus:border-primary"
                  placeholder="Enter sleep goal"
                  value={goals.sleep}
                  onChange={(e) => updateGoal('sleep', e.target.value)}
                />
              </div>

              <div>
                <label className="block text-gray-700 font-semibold">Custom Goal:</label>
                <input
                  type="text"
                  className="w-full p-2 mt-1 rounded-md border border-gray-300 focus:border-primary"
                  placeholder="Enter a custom goal"
                  value={goals.customGoal}
                  onChange={(e) => updateGoal('customGoal', e.target.value)}
                />
                <button
                  onClick={addCustomGoal}
                  className="bg-primary text-black px-4 py-2 mt-2 rounded-lg shadow hover:bg-primary-dark"
                >
                  Add Custom Goal
                </button>
              </div>
            </div>

            <button
              onClick={handleModalClose}
              className="bg-primary text-black px-4 py-2 rounded-lg mt-6 w-full shadow hover:bg-primary-dark"
            >
              Save Goals
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default CustomHealthGoals;
