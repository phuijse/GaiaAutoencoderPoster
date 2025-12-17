gaia_intro = """
ESA's *Gaia* (2013-2025) is a space astrometry mission designed to deliver a precise census of the Milky Way. Among other data products, *Gaia*'s third data release (DR3) provides:

- **Astrometry:** Positions and parallaxes for 1.5 billion astronomical sources
- **Photometry:** Broadband light curves for 11.7 million sources
- **Spectrophotometry:** Low-resolution spectra for 219 million sources 

With its all-sky, multi-epoch observations, *Gaia* provides an exceptional dataset for variability studies.
"""

tda_intro = """
We say that an astronomical source is **variable**, when its brightness, as observed from Earth, changes over time. 

Variability is driven by a broad range of physical processes, including **pulsations, eclipses, rotation and magnetic activity, eruptions, and accretion**.

The examples below illustrate idealized **light curves (brightness time series)** of a pulsating star (top) and an eclipsing binary system (bottom). 
"""

ae_intro = """
An autoencoder is an **artificial neural network** with two components:

- An **encoder** network that maps the input data to a compressed **latent representation**
- A **decoder** network that expands this representation to reconstruct the original data

By minimizing the error between the input data and its reconstruction, the model learns a latent representation that holds the most important features of the data. 

This representation can then be used for classification, clustering, anomaly detection and visualization purposes.
"""

encoding_md = """
- Each point in the scatter plot corresponds to a source (star/galaxy) observed by *Gaia*
- The encoder network receives the low-resolution spectra and phase-folded light curve and returns the latent coordinates
- The color of the point represents the variability class according to Rimoldini et al. 2023
"""

decoding_md = """
- During training the decoder "reconstructs" the input data from their latent space position
- Each slider corresponds to one latent variable
- We used five latent variables each for the low-resolution spectra and phase-folded time series, respectively
"""

classification_md = """
- The latent variables are used to train a linear classifier.
- The confusion matrix to the right visualizes the classifier’s performance.
- The i, j cell represents the percentage of sources from class i that are classified as j.
- Misclassifications are mostly confined to closely related subtypes.
"""

references = """
1. [Gaia collaboration, "The Gaia mission", A&A, 2016](https://www.aanda.org/articles/aa/full_html/2016/11/aa29272-16/aa29272-16.html) 
1. [Gaia collaboration, "Gaia Data Release 3. Summary of the content and survey properties", A&A, 2023](https://www.aanda.org/articles/aa/full_html/2023/06/aa43940-22/aa43940-22.html)
1. [L. Rimoldini, B. Holl, P. Gavras, et al., "All-sky classification of 12.4 million variable sources into 25 classes", A&A, 2023](https://www.aanda.org/articles/aa/full_html/2023/06/aa45591-22/aa45591-22.html)
1. [P. Huijse, J. De Ridder, L. Eyer, et al., "Learning novel representations of variable sources from multi-modal Gaia data via autoencoders", A&A, 2025](https://www.aanda.org/articles/aa/full_html/2025/09/aa54025-25/aa54025-25.html)
"""

conclusions = """
- We trained a neural network model that combines *Gaia* time series and low-resolution spectra into a single representation
- The class of the object is not given to the model. Yet, the different variability classes organize cleanly in the new representation
- These results were obtained using data from the *Gaia* DR3. **Scan the QR code below to access see our article**
- We are currently working on incorporating data from additional instruments and implementing the models for DR4
"""
