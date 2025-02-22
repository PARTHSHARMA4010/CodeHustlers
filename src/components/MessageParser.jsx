import React, { useState } from 'react';

const MessageParser = ({ children, actions }) => {
  const parse = async (message) => {
    await actions.geminiAction(message);
  };

  return (
    <>
    <div>
      {React.Children.map(children, (child) => {
        return React.cloneElement(child, {
          parse: parse,
          actions: actions,
        });
      })}
    </div>
    </> 
  );
};

export default MessageParser;
