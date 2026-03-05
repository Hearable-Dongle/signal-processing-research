export function speakerColor(speakerId: number): string {
  const hue = (Math.abs(speakerId * 73) + 41) % 360;
  return `hsl(${hue} 75% 52%)`;
}
