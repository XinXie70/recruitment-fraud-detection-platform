import React, { useEffect, useRef } from 'react';

export default function MeteorBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animationFrame;
    let meteors = [];
    let stars = [];
    let nebulae = [];

    const createMeteor = (width, height) => ({
      x: Math.random() * width,
      y: Math.random() * height - height,
      length: 90 + Math.random() * 160,
      speed: 0.55 + Math.random() * 1.25,
      drift: -0.12 + Math.random() * 0.32,
      alpha: 0.06 + Math.random() * 0.16,
      width: Math.random() > 0.78 ? 2 : 1,
    });

    const createStar = (width, height) => ({
      x: Math.random() * width,
      y: Math.random() * height,
      radius: 0.7 + Math.random() * 1.9,
      alpha: 0.18 + Math.random() * 0.5,
      pulse: Math.random() * Math.PI * 2,
    });

    const createNebula = (width, height, index) => ({
      x: width * (index === 0 ? 0.18 : 0.78),
      y: height * (index === 0 ? 0.22 : 0.72),
      radius: Math.max(width, height) * (index === 0 ? 0.36 : 0.42),
      color: index === 0 ? '232, 168, 111' : '111, 128, 103',
      alpha: index === 0 ? 0.18 : 0.12,
    });

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = window.innerWidth;
      const height = window.innerHeight;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      meteors = Array.from({ length: Math.max(18, Math.floor(width / 70)) }, () =>
        createMeteor(width, height),
      );
      stars = Array.from({ length: Math.max(90, Math.floor((width * height) / 11000)) }, () =>
        createStar(width, height),
      );
      nebulae = [createNebula(width, height, 0), createNebula(width, height, 1)];
    };

    const draw = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      ctx.clearRect(0, 0, width, height);

      nebulae.forEach((nebula) => {
        const glow = ctx.createRadialGradient(
          nebula.x, nebula.y, 0, nebula.x, nebula.y, nebula.radius,
        );
        glow.addColorStop(0, `rgba(${nebula.color}, ${nebula.alpha})`);
        glow.addColorStop(0.42, `rgba(${nebula.color}, ${nebula.alpha * 0.35})`);
        glow.addColorStop(1, `rgba(${nebula.color}, 0)`);
        ctx.fillStyle = glow;
        ctx.fillRect(0, 0, width, height);
      });

      const now = Date.now() / 900;
      stars.forEach((star) => {
        const twinkle = star.alpha + Math.sin(now + star.pulse) * 0.08;
        ctx.fillStyle = `rgba(201, 127, 61, ${Math.max(0.08, twinkle)})`;
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
        ctx.fill();
      });

      meteors.forEach((meteor, index) => {
        const endX = meteor.x + meteor.drift * meteor.length;
        const endY = meteor.y + meteor.length;
        const gradient = ctx.createLinearGradient(meteor.x, meteor.y, endX, endY);
        gradient.addColorStop(0, `rgba(201, 127, 61, ${meteor.alpha * 0.28})`);
        gradient.addColorStop(0.42, `rgba(201, 127, 61, ${meteor.alpha * 0.62})`);
        gradient.addColorStop(0.82, `rgba(201, 127, 61, ${meteor.alpha})`);
        gradient.addColorStop(1, 'rgba(201, 127, 61, 0)');

        ctx.strokeStyle = gradient;
        ctx.lineWidth = meteor.width;
        ctx.beginPath();
        ctx.moveTo(meteor.x, meteor.y);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        meteor.x += meteor.drift;
        meteor.y += meteor.speed;

        if (meteor.y > height + meteor.length) {
          meteors[index] = createMeteor(width, height);
          meteors[index].y = -meteor.length;
        }
      });

      animationFrame = window.requestAnimationFrame(draw);
    };

    resize();
    draw();
    window.addEventListener('resize', resize);

    return () => {
      window.removeEventListener('resize', resize);
      window.cancelAnimationFrame(animationFrame);
    };
  }, []);

  return <canvas className="meteor-canvas" ref={canvasRef} aria-hidden="true" />;
}
