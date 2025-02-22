import React from "react";
import "../assets/part3.css";

function Part3() {
  return (
    <div style={{ backgroundColor: '#F7F6F2' }}>
    <div className="app-container "  style={{ backgroundColor: '#F7F6F2' }}>
      <h1 className="heading">Take this simple test and measure the stress in your life</h1>
      <div className="content-container">
        <div className="image-container">
          <img
            src="./test_score.png"
            alt="Placeholder"
            className="image"
          />
        </div>
        <div className="quote-container">
          <blockquote className="quote">
          “Understanding your stress is the first step to taking control. Know where you stand and reclaim your peace – empower yourself today!”
          </blockquote>
        </div>
      </div>
      <div className="button-container">
        <button className="action-button">Get your Stress Score️ Now!!</button>
        <p className="button-text">Take action now to move forward!</p>
      </div>
    </div>
    </div>
  );
}

export default Part3;