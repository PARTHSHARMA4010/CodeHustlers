import React, { useState } from 'react';
import '../assets/comfort.css';

function Parth() {
  const [mood, setMood] = useState(null);
  const recommendations = {
    stressful: [
      { title: 'Meditation Video', link: 'https://youtu.be/iFpCeuA1mRs', image: '/images/guided.png', text: 'A calming meditation video to relieve stress.' },
      { title: 'Relaxing Music', link: 'https://open.spotify.com/playlist/4oblctaZrOuwS4YCeRG6wO?si=46bd6528656e4178', image: '/images/relaxingmusic.png', text: 'Soothing music for a peaceful mind.' },
      { title: 'Breathing Exercise', link: 'https://youtu.be/lPJAtHWq08k?si=sKwYrIfMg6GJkI9Z', image: '/images/breathing.png', text: 'Breathing exercises to help reduce anxiety.' },
      { title: 'Self-Help books', link: 'https://www.goodreads.com/shelf/show/self-help', image: '/images/selfhelpbook.png', text: 'Breathing exercises to help reduce anxiety.' },
    ],
    happy: [
      { title: 'Upbeat Playlist', link: 'https://open.spotify.com/playlist/37i9dQZF1EVJHK7Q1TBABQ?si=55c37a6cc4214ebb', image: 'playlist.jpg', text: 'A playlist to keep you in a joyful mood.' },
      { title: 'Feel Good Book', link: '#', image: '/images/feelgood.png', text: 'A book recommendation to amplify your happiness.' },
      { title: 'Creative Projects', link: '#', image: '/images/creative2.png', text: 'Ideas for fun and creative projects.' },
      { title: 'Breathing Exercise', link: 'https://youtu.be/lPJAtHWq08k?si=sKwYrIfMg6GJkI9Z', image: '/images/breathing.png', text: 'Breathing exercises to help reduce anxiety.' },
    ],
    nervous: [
      { title: 'Calm Down Techniques', link: '#', image: '/images/calm.jpg', text: 'Techniques to ease nervousness.' },
      { title: 'Guided Breathing', link: 'https://youtu.be/pAEioF7FaWY', image: "", text: 'Guided breathing for anxiety relief.' },
      { title: 'Soothing Sounds', link: 'https://open.spotify.com/playlist/78lWPOkG1nzQOWt5ScnQCD?si=c929e61e16d54301', image: 'sounds.jpg', text: 'Natural sounds to calm your nerves.' },
      { title: 'Breathing Exercise', link: 'https://youtu.be/lPJAtHWq08k?si=sKwYrIfMg6GJkI9Z', image: '/images/breathing.png', text: 'Breathing exercises to help reduce anxiety.' },
    ],
    excitement: [
      { title: 'Adventure Planning', link: '#', image: 'adventure.jpg', text: 'Plan your next big adventure.' },
      { title: 'Energy Boost Playlist', link: 'https://open.spotify.com/playlist/3mSm688yR6UeaAJNf93Ydr?si=11f49937cc6d4f4a', image: 'boost.jpg', text: 'A playlist to match your excitement.' },
      { title: 'Creative Activities', link: '#', image: 'activities.jpg', text: 'Engage in exciting creative activities.' },
      { title: 'Breathing Exercise', link: 'https://youtu.be/lPJAtHWq08k?si=sKwYrIfMg6GJkI9Z', image: '/images/breathing.png', text: 'Breathing exercises to help reduce anxiety.' },
    ],
  };

  return (
    <div className="app">
   <div className="parts"><div className="left-part"><h1>Interactive Healing</h1>
   <h2 style={{ maxWidth: '600px' }}>
  "Interactive healing is the bridge between mindful connection and inner peace, 
  where personalized care meets the transformative power of self-discovery."
</h2>
<div className="mood-selection-container">
            <p className="input">How are you feeling today?</p>
            <p className="mood-instruction">Choose a mood to receive tailored recommendations:</p>
      <div className="mood-options">
      <button onClick={() => setMood('stressful')} aria-label="Select Stressful Mood">
                <span role="img" aria-label="stressful">😣</span> Stressful
              </button>
              <button onClick={() => setMood('happy')} aria-label="Select Happy Mood">
                <span role="img" aria-label="happy">😊</span> Happy
              </button>
              <button onClick={() => setMood('nervous')} aria-label="Select Nervous Mood">
                <span role="img" aria-label="nervous">😰</span> Nervous
              </button>
              <button onClick={() => setMood('excitement')} aria-label="Select Excitement Mood">
                <span role="img" aria-label="excitement">🤩</span> Excitement
              </button>
      </div> 
      </div>
      </div>
      <div className="right-section">
    <img src="/images/bg.png" alt="Healing Illustration" />
  </div> </div>


      {mood && (
        <div className="recommendations">
          <h2>Recommendations for {mood.charAt(0).toUpperCase() + mood.slice(1)} Mood</h2>
          <div className="cards">
            {recommendations[mood].map((rec, index) => (
              <div key={index} className="card">
                <img src={rec.image} alt={rec.title} />
                <h3>{rec.title}</h3>
                <p>{rec.text}</p>
                <a href={rec.link}>Learn More</a>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default Parth;