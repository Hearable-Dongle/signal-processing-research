


import numpy as np

from localization.algo import SRPPHATLocalization



def run_test(f_min, f_max, desc):

    # Mock data: 8 mics in circle, radius 0.1m

    M = 8

    R = 0.1

    fs = 16000

    duration = 0.5

    t = np.linspace(0, duration, int(fs*duration))

    

    # Mic positions

    phi = np.linspace(0, 2 * np.pi, M, endpoint=False)

    mic_pos = np.array([

        R * np.cos(phi),

        R * np.sin(phi),

        np.zeros(M)

    ])

    

    signals = []

    # Source at 135 degrees

    src_deg = 135

    src_angle = np.radians(src_deg)

    src_u = np.array([np.cos(src_angle), np.sin(src_angle), 0])

    

    for m in range(M):

        # delay t_i = - p_i . u / c

        delay = -np.dot(mic_pos[:, m], src_u) / 343.0

        sig = np.sin(2 * np.pi * 1000 * (t - delay)) 

        # Add some broadband noise to source to fill spectrum

        sig += 0.5 * np.random.normal(0, 1, len(sig))

        

        # Add silence to first half

        sig[:len(sig)//2] = np.random.normal(0, 0.001, len(sig)//2) # Noise floor

        

        signals.append(sig)

        

    audio = np.array(signals)

    

    loc = SRPPHATLocalization(mic_pos=mic_pos, fs=fs, nfft=512, overlap=0.5, freq_range=(f_min, f_max), max_sources=1)

    

    doas, P, _ = loc.process(audio)

    

    deg_angles = np.linspace(0, 360, len(P), endpoint=False)

    

    # Find indices

    idx_src = np.argmin(np.abs(deg_angles - src_deg))

    idx_ghost = np.argmin(np.abs(deg_angles - ((src_deg + 180) % 360)))

    

    val_src = P[idx_src]

    val_ghost = P[idx_ghost]

    

    print(f"--- {desc} ({f_min}-{f_max} Hz) ---")

    print(f"Estimated DOA: {np.degrees(doas)}")

    print(f"Power at {src_deg} deg: {val_src:.4f}")

    print(f"Power at {(src_deg + 180) % 360} deg: {val_ghost:.4f}")

    print(f"Ratio Ghost/Source: {val_ghost/val_src:.4f}")



def test_srp_phat():

    run_test(200, 3000, "Low Freq Inclusion")

    run_test(500, 3000, "Medium Freq Start")

    run_test(800, 3000, "High Freq Start")



if __name__ == "__main__":

    test_srp_phat()
