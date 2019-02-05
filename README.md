# Van Gogh's Starry Night Power Spectrum Analysis

This is the code for the aXiv article: add hyperlink

## Abstract:

Vincent van Gogh's painting, The Starry Night, is an iconic piece of art and cultural history. The painting portrays a night sky full of stars, with eddies (spirals) both large and small. \cite{Kolmogorov1941}'s description of subsonic, incompressible turbulence gives a model for turbulence that involves eddies interacting on many length scales, and so the question has been asked: is The Starry Night turbulent? To answer this question, we calculate the azimuthally averaged power spectrum of a square region (<img src="https://rawgit.com/AstroJames/VanGoghsStarryNight/Master/svgs/e47ef54470941a8726c44eaa6c2ef513.svg?invert_in_darkmode" align=middle width=85.8448668pt height=21.1872144pt/> pixels) of night sky in The Starry Night. We find a power spectrum, <img src="https://rawgit.com/AstroJames/VanGoghsStarryNight/Master/svgs/4dd602e3d752c85e1fb5cae1ddb3402b.svg?invert_in_darkmode" align=middle width=34.6462446pt height=24.657534pt/>, where <img src="https://rawgit.com/AstroJames/VanGoghsStarryNight/Master/svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.07536795pt height=22.8310566pt/> is the wavevector, that shares the same features as supersonic turbulence. It has a power-law <img src="https://rawgit.com/AstroJames/VanGoghsStarryNight/Master/svgs/f619cd520ae41ca02a2a8a3ffb675c1d.svg?invert_in_darkmode" align=middle width=109.931712pt height=26.7617526pt/> in the scaling range, <img src="https://rawgit.com/AstroJames/VanGoghsStarryNight/Master/svgs/7ba1a94dc0cb8aa4ea667946b71082c8.svg?invert_in_darkmode" align=middle width=85.78746165pt height=22.8310566pt/>. We identify a driving scale, <img src="https://rawgit.com/AstroJames/VanGoghsStarryNight/Master/svgs/0fcb48c0b1179c304a585eadf97c5a74.svg?invert_in_darkmode" align=middle width=49.413936pt height=22.8310566pt/>, dissipation scale, <img src="https://rawgit.com/AstroJames/VanGoghsStarryNight/Master/svgs/3cdb85d1e1bacfe1eeab2fd8c0789b7d.svg?invert_in_darkmode" align=middle width=63.31240575pt height=22.8310566pt/> and a bottleneck. This leads us to believe that van Gogh's depiction of the starry night closely resembles the turbulence found in real molecular clouds, the birthplace of stars in the Universe.

## Example Command Line Argument in iPython:

run PowerSpectrumVanGogh -file "StarryNightSky.png"

where -file is the argument for feeding in the Starry Night data.

## Example data from Starry Night:

The .png file that I pass to my PowerSpectrumVanGogh code is:

![Example Starry Night data.](https://github.com/AstroJames/VanGoghsStarryNight/blob/Master/StarryNightSky.png)
