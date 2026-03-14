export type MicArrayProfile = "respeaker_v3_0457" | "respeaker_xvf3800_0650";

function normalizeDirectionDeg(value: number): number {
  const wrapped = value % 360;
  return wrapped < 0 ? wrapped + 360 : wrapped;
}

export function backendArrivalToUiSourceBearingDeg(directionDeg: number, micArrayProfile: MicArrayProfile): number {
  const angle = normalizeDirectionDeg(directionDeg);
  if (micArrayProfile === "respeaker_xvf3800_0650") {
    return normalizeDirectionDeg(270 - angle);
  }
  return angle;
}
