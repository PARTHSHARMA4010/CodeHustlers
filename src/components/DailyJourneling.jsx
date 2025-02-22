import React, { useState } from 'react';

function DailyJournaling() {
  const [entry, setEntry] = useState('');
  const [journalEntries, setJournalEntries] = useState([]);

  const saveEntry = () => {
    if (entry) {
      setJournalEntries([...journalEntries, { text: entry, date: new Date().toLocaleDateString() }]);
      setEntry('');
    }
  };

  return (
    <div className="bg-primary text-white p-8 rounded-lg shadow-md">
      <h2 className="text-lg font-bold mb-2 text-white">Daily Journaling</h2>
      <textarea
        className="w-full p-2 text-black rounded-md"
        placeholder="Write about your day..."
        value={entry}
        onChange={(e) => setEntry(e.target.value)}
      ></textarea>
      <button className="bg-secondary text-primary p-2 mt-2 rounded-md" onClick={saveEntry}>
        Save Entry
      </button>
      <div className="mt-4">
        <h3 className="font-semibold">Past Entries:</h3>
        <ul>
          {journalEntries.map((entry, index) => (
            <li key={index} className="text-sm">
              {entry.date}: {entry.text}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}


export default DailyJournaling;