update my localization algorithm in @localization/ to use a power-focused flow rather than a simple histogram of time delays. The goal is to improve the accuracy of Direction of Arrival (DOA) estimation by incorporating frequency-dependent weighting into the GCC-PHAT method.


To get a robust implementation, ask your agent to focus on these three specific components:GCC-PHAT for all Pairs: For your 4 mics, you have 6 unique pairs $( \binom{4}{2} = 6 )$.$$GCC_{PHAT}(\tau) = \mathcal{F}^{-1} \left( \frac{X_i(f) X_j^*(f)}{|X_i(f) X_j^*(f)|} \right)$$The Functional Form: Instead of a simple histogram, the "Power" at a candidate DOA $(\theta)$ is:$$P(\theta) = \sum_{pair=1}^{6} \sum_{f=f_{min}}^{f_{max}} W(f) \cdot \text{Real}(GCC_{PHAT}(f) \cdot e^{j 2\pi f \tau_{pair}(\theta)})$$The Weighting $W(f)$: Suggest that the agent use SNR-based weighting. If a frequency bin has very little energy compared to the noise floor (which you can estimate during "silence" frames), its contribution to the spatial sum should be zeroed out.

allow an option to use the normal histogram method or the power-weighted method via parameterization in the function call or if this flow is significantly diffferent, just treat it as a different algo_type altogether. this should fit seemlessly into the existing localization pipeline in @localization/ (called in @localization/main.py with python -m localization.main ...)





