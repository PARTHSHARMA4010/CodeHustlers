// Notification.jsx
import React from 'react';

const Notification = ({ message, type, onClose }) => {
  return (
    <div className={`fixed top-4 right-4 z-50 p-4 rounded-md shadow-lg text-white ${type === 'success' ? 'bg-green-500' : 'bg-red-500'}`}>
      <div className="flex justify-between items-center">
        <span>{message}</span>
        <button onClick={onClose} className="ml-4 text-xl">&times;</button>
      </div>
    </div>
  );
};

export default Notification;
