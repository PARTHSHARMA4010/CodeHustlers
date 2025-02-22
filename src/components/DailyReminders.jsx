// components/DailyReminders.jsx

import React from 'react';

const reminders = [
  {
    id: 1,
    label: 'Diet',
    actionText: 'Check Diet Plan',
    link: 'https://www.mhanational.org/sites/default/files/MHM%202014%20Fact%20Sheet-%20Healthy%20Diet%20Eating%20with%20Mental%20Health%20in%20Mind.pdf',
  },
  {
    id: 2,
    label: 'Exercise',
    actionText: 'Begin Now',
    link: 'https://www.youtube.com/watch?v=COp7BR_Dvps',
  },
  {
    id: 3,
    label: 'Meditation',
    actionText: 'Begin Now',
    link: 'https://www.youtube.com/watch?v=xRxT9cOKiM8',
  },
];

const DailyReminders = () => {
  return (
    <div className="bg-primary text-white p-8 rounded-lg shadow-md">
      <h2 className="text-lg font-bold mb-2 text-white">Daily Reminders</h2>
      {reminders.map((item) => (
        <div key={item.id} style={styles.reminderItem}>
          <input type="checkbox" id={`reminder-${item.id}`} style={styles.checkbox} />
          <label htmlFor={`reminder-${item.id}`} style={styles.label}>
            {item.label}
          </label>
          <button
            style={styles.button}
            onClick={() => window.open(item.link, '_blank')}
            onMouseEnter={(e) => e.target.style.backgroundColor = styles.buttonHover.backgroundColor}
            onMouseLeave={(e) => e.target.style.backgroundColor = styles.button.backgroundColor}
          >
            {item.actionText}
          </button>
        </div>
      ))}
    </div>
  );
};

const styles = {
    container: {
        padding: '20px',
        borderRadius: '8px',
        backgroundColor: '#9AC8B5', // Matches the design of other components
        boxShadow: '2px 2px 5px rgba(0, 0, 0, 0.1)',
        maxWidth: '400px',
        margin: '0 auto',
      },
      heading: {
        textAlign: 'center',
        color: '#2A5D67', // Consistent color for headings
        fontSize: '1.5rem',
        fontWeight: 'bold',
        marginBottom: '20px',
      },
  reminderItem: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '15px',
  },
  checkbox: {
    marginRight: '15px',
    fontSize:  '1.5rem',
  },
  label: {
    flex: '1',
    fontSize: '1.1rem',
    color: '#fff',
  },
  button: {
    padding: '8px 12px',
    backgroundColor: '#4CAF50',
    color: '#fff',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer',
    boxShadow: '1px 1px 3px rgba(0, 0, 0, 0.1)',
    fontSize: '0.9em',
    transition: 'background-color 0.3s',
    maxWidth: 200,
    minWidth: 200,
  },
  buttonHover: {
    backgroundColor: '#45a049',
  },
};

export default DailyReminders;
    