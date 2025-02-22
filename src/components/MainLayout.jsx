import React from 'react';
import DailyJournaling from './DailyJourneling';
import CustomHealthGoals from './CustomHealth';
import ProgressTracker from './ProgressTracker';
import DailyReminders from './DailyReminders';


function MainLayout() {
  return (
    <div className="bg-background min-h-screen p-6">
      <header className="text-center text-primary text-3xl font-bold mb-4">
        Personalized Planner and Daily Targets
      </header>
      <p className="text-center text-secondary font-semibold mb-8">
        "Success is the sum of small efforts, repeated day in and day out." - Robert Collier
      </p>
      <div className="grid gap-4 md:grid-cols-2">
        {/* <DailyReminders />
        <CustomHealthGoals />
        <ProgressTracker />
        <DailyJournaling /> */}

        <DailyReminders/>
        <CustomHealthGoals/>
        <ProgressTracker/>
        <DailyJournaling/>
        
      </div>
    </div>
  );
}

export default MainLayout;