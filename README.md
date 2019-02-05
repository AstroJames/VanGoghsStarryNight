# Van Gogh's Starry Night Power Spectrum Analysis

This is the code for the aXiv article: add hyperlink

## Abstract:

Vincent van Gogh's painting, The Starry Night, is an iconic piece of art and cultural history. The painting portrays a night sky full of stars, with eddies (spirals) both large and small. \cite{Kolmogorov1941}'s description of subsonic, incompressible turbulence gives a model for turbulence that involves eddies interacting on many length scales, and so the question has been asked: is The Starry Night turbulent? To answer this question, we calculate the azimuthally averaged power spectrum of a square region ($1165 \times 1165$ pixels) of night sky in The Starry Night. We find a power spectrum, $\mathcal{P}(k)$, where $k$ is the wavevector, that shares the same features as supersonic turbulence. It has a power-law $\mathcal{P}(k) \propto k^{2.1\pm0.3}$ in the scaling range, $34 \leq k \leq 80$. We identify a driving scale, $k_\text{D} = 3$, dissipation scale, $k_\nu = 220$ and a bottleneck. This leads us to believe that van Gogh's depiction of the starry night closely resembles the turbulence found in real molecular clouds, the birthplace of stars in the Universe.

## Example Command Line Argument in iPython:
```
run PowerSpectrumVanGogh -file "StarryNightSky.png"
```
where -file is the argument for feeding in the Starry Night data.

## Example data from Starry Night:

The .png file that I pass to my PowerSpectrumVanGogh code is:

![Example Starry Night data.](https://github.com/AstroJames/VanGoghsStarryNight/blob/Master/StarryNightSky.png)
