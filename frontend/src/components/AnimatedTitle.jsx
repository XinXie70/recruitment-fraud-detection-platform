import React from 'react';

export default function AnimatedTitle({ text }) {
  let letterIndex = 0;

  return (
    <h1 className="animated-title" aria-label={text}>
      {text.split(' ').map((word, wordIndex) => (
        <span className="animated-word" key={`${word}-${wordIndex}`} aria-hidden="true">
          {Array.from(word).map((character) => {
            const currentIndex = letterIndex;
            letterIndex += 1;

            return (
              <span
                className="animated-letter"
                key={`${character}-${wordIndex}-${currentIndex}`}
                style={{ '--letter-index': currentIndex }}
              >
                {character}
              </span>
            );
          })}
        </span>
      ))}
    </h1>
  );
}
