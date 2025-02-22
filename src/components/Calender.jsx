import { useState, useEffect } from "react";

const CURR_YEAR = 2024;

const monthMap = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const yearMap = [2024, 2025, 2026, 2027];
const dayNames = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

const doctors = [
  { name: "Dr. Smith", specialty: "Psychologist", image: "https://via.placeholder.com/100" },
  { name: "Dr. Lee", specialty: "Therapist", image: "https://via.placeholder.com/100" },
  { name: "Dr. Patel", specialty: "Counselor", image: "https://via.placeholder.com/100" },
];

export default function Calendar() {
  const [month, setMonth] = useState(0);
  const [year, setYear] = useState(CURR_YEAR);
  const [daysArray, setDaysArray] = useState([]);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [showDoctorPopup, setShowDoctorPopup] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [showPointsPopup, setShowPointsPopup] = useState(false);
  const [points, setPoints] = useState(240);

  useEffect(() => {
    const daysInAMonth = daysInMonth(month + 1, year);
    const firstDayOffset = getFirstDayOfMonth(month + 1, year);
    const days = Array(firstDayOffset).fill(null);
    for (let i = 1; i <= daysInAMonth; i++) {
      days.push(i);
    }
    setDaysArray(days);
  }, [month, year]);

  function daysInMonth(month, year) {
    return new Date(year, month, 0).getDate();
  }

  function getFirstDayOfMonth(month, year) {
    return new Date(year, month - 1, 1).getDay();
  }

  const handleSelectDate = (day) => {
    setSelectedDate({ day, month, year });
  };

  const handleSelectDoctor = (doctor) => {
    setSelectedDoctor(doctor);
    setShowDoctorPopup(false);
  };

  const handleBookConsultation = () => {
    if (selectedDoctor && selectedDate) {
      setShowPointsPopup(true);
    } else {
      alert("Please select both a doctor and a date.");
    }
  };

  const handleConfirmBooking = () => {
    setPoints(points - 200);
    setShowPointsPopup(false);
    setShowConfirmation(true);
  };

  return (
    <div className="flex justify-center items-center h-screen w-screen bg-[#F7F6F2] text-[#006953]">
      <div className="max-w-lg p-6 bg-white rounded-lg shadow-lg w-full">
        <h1 className="text-3xl font-bold mb-6 text-center">Virtual Health Camp</h1>
        
        {/* Month and Year Selection */}
        <div className="flex justify-center space-x-4 mb-4">
          <select
            onChange={(e) => setMonth(Number(e.target.value))}
            className="p-2 bg-white border rounded"
            style={{ color: '#006953' }}
          >
            <option value="">--Select Month--</option>
            {monthMap.map((m, i) => (
              <option key={i} value={i}>{m}</option>
            ))}
          </select>
          
          <select
            onChange={(e) => setYear(Number(e.target.value))}
            className="p-2 bg-white border rounded"
            style={{ color: '#006953' }}
          >
            <option value="">--Select Year--</option>
            {yearMap.map((y, i) => (
              <option key={i} value={y}>{y}</option>
            ))}
          </select>
        </div>

        {/* Calendar Header */}
        <div
          className="grid grid-cols-7 gap-1 w-full max-w-sm mb-2 text-center font-bold"
          style={{ color: '#A9C7B8' }}
        >
          {dayNames.map((day, i) => (
            <div key={i}>{day}</div>
          ))}
        </div>

        {/* Calendar Days */}
        <div className="grid grid-cols-7 gap-1 w-full max-w-sm">
          {daysArray.map((day, index) => (
            <div
              key={index}
              onClick={() => day && handleSelectDate(day)}
              className={`h-14 w-14 flex items-center justify-center border rounded cursor-pointer transition-all duration-200 ${
                day ? "hover:bg-gray-200" : "bg-transparent"
              }`}
              style={{
                color:
                  selectedDate?.day === day &&
                  selectedDate.month === month &&
                  selectedDate.year === year
                    ? '#ffffff'
                    : '#006953',
                borderColor: '#A9C7B8',
                backgroundColor:
                  selectedDate?.day === day &&
                  selectedDate.month === month &&
                  selectedDate.year === year
                    ? '#71b48d'
                    : day
                    ? '#f7f6f2'
                    : 'transparent',
              }}
            >
              {day}
            </div>
          ))}
        </div>

        {/* Doctor Selection */}
        <div className="mt-6 text-center">
          <button
            onClick={() => setShowDoctorPopup(true)}
            className="bg-primary text-white font-bold py-2 px-4 rounded transition-all duration-200 hover:bg-opacity-80"
            style={{ backgroundColor: '#006953' }}
          >
            Select Doctor
          </button>
        </div>

        {/* Doctor Selection Popup */}
        {showDoctorPopup && (
          <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-10">
            <div className="bg-white rounded-lg shadow-lg p-6 max-w-md w-full">
              <h2 className="text-2xl font-bold mb-4" style={{ color: '#006953' }}>Select a Doctor</h2>
              <div className="grid gap-4">
                {doctors.map((doctor, i) => (
                  <div key={i} className="flex items-center p-2 rounded-md border transition-all duration-200 hover:bg-gray-200 cursor-pointer" onClick={() => handleSelectDoctor(doctor)}>
                    <img src={doctor.image} alt={doctor.name} className="rounded-full w-16 h-16 mr-4" />
                    <div>
                      <p className="font-bold" style={{ color: '#006953' }}>{doctor.name}</p>
                      <p className="text-gray-700">{doctor.specialty}</p>
                    </div>
                  </div>
                ))}
              </div>
              <button
                onClick={() => setShowDoctorPopup(false)}
                className="mt-4 text-primary font-bold underline"
                style={{ color: '#A9C7B8' }}
              >
                Close
              </button>
            </div>
          </div>
        )}

        {/* Book Consultation Button */}
        {selectedDoctor && selectedDate && (
          <div className="mt-4 text-center">
            <button
              onClick={handleBookConsultation}
              className="bg-primary text-white font-bold py-2 px-4 rounded transition-all duration-200 hover:bg-opacity-80"
              style={{ backgroundColor: '#006953' }}
            >
              Book Consultation with {selectedDoctor.name}
            </button>
          </div>
        )}

        {/* Points Deduction Popup */}
        {showPointsPopup && (
          <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-20">
            <div className="bg-white rounded-lg shadow-lg p-6 max-w-md w-full">
              <h2 className="text-2xl font-bold mb-4" style={{ color: '#006953' }}>Points Deduction</h2>
              <p className="mb-4">Booking this consultation will deduct 200 points from your balance of 440 points. You will have 240 points remaining after this booking. For future consultations, you will need to recharge or pay.</p>
              <div className="flex justify-center space-x-4">
                <button
                  onClick={handleConfirmBooking}
                  className="bg-primary text-white font-bold py-2 px-4 rounded hover:bg-opacity-80"
                  style={{ backgroundColor: '#006953' }}
                >
                  Confirm
                </button>
                <button
                  onClick={() => setShowPointsPopup(false)}
                  className="text-primary font-bold underline"
                  style={{ color: '#A9C7B8' }}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Confirmation Popup */}
        {showConfirmation && (
          <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-10">
            <div className="bg-white rounded-lg shadow-lg p-6 max-w-md w-full">
              <h2 className="text-2xl font-bold mb-4" style={{ color: '#006953' }}>Confirm Your Appointment</h2>
              <p>Doctor: <strong>{selectedDoctor.name}</strong></p>
              <p>Date: <strong>{`${selectedDate.day} ${monthMap[selectedDate.month]} ${selectedDate.year}`}</strong></p>
              <button
                onClick={() => setShowConfirmation(false)}
                className="mt-4 bg-primary text-white font-bold py-2 px-4 rounded hover:bg-opacity-80"
                style={{ backgroundColor: '#006953' }}
              >
                Confirm
              </button>
              <button
                onClick={() => setShowConfirmation(false)}
                className="mt-4 text-primary font-bold underline ml-4"
                style={{ color: '#A9C7B8' }}
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
  