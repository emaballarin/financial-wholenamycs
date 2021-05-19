# Whole-system multidimensional financial time-series modeling from only-homogeneous data

### Davide Roznowicz and [Emanuele Ballarin](https://ballarin.cc)

#### [University of Trieste](https://units.it) and [SISSA](https://sissa.it)

</br>

## *Long abstract*

### Context

Since their very inception, financial markets and the data they produce have always drawn the interest and efforts of mathematicians, physicists, statisticians and - generically - *modeling professionals* due to the obvious economic appeal they exhibit. Soon, in addition to *material goals*, their more abstract and academic study began being driven by the (still ongoing) endavour to prove (or disprove) their direct generalizable learnability via statistically-learning machines.  

The many different scales (micro- and macro-economic, and everything in between), items (time series of prices, volumes, etc... of different commodities and financial instruments, or economic agents themselves) and approaches (plain prediction, causal modeling, credit-assignment based, agent-based) of such modeling - and the numerosity and generally strong financial involvement of modelers themselves - contributed to a great abundance of models being proposed, of which - though - it is often difficult to grasp clear applicability boundaries, limits, preprocessing required and actual performance (let alone the availability of their parameters, or weights) - and a relative scarcity of published scholarly articles or papers detailing their inner workings.  

Of those many models proposed to date, the most common category attacks the problem of predicting the time-dependent evolution of prices (or functions thereof) of tradeable items (e.g. currencies, commodities, stock shares) phenomenologically, via both classical statistical tools tailored toward time series modeling and *deep learning based* approaches. In any case, being the system *borderline-chaotic*, given the lack of widely established *physical laws* governing it and the etherogeneity of their agents' motives, its very nature calls for following the most *holistic* and *data-driven* modeling approach possible.

### Proposal (a.k.a. *short abstract*)

In our work - whose development is ongoing - we propose to assess the feasibility, and produce a *proof of concept* implementation, of a *deep-learning-based* *fully end-to-end* system to predict and/or eventually simulate *ex novo* multidimensional (i.e. composed of many, potentially correlated items) homogeneous (i.e. of the same type, from the domain-specific point of view) financial time-series data with the associated uncertainty quantification information, starting from a homogeneous dataset of the same kind and no additional or alternate data sources. The main focus will be on US stock shares prices.  

Given the *borderline-chaoticity* and *noisiness* of the system, but also the possibility of long-ranging dependencies, nontrivial cross-correlations among items, and the eventuality of sharply localized *phase transitions* in its dynamics, no *single-architecture* system seems right-off adequate for the task of interest.  

Neither:
    - ++Recurrent sequence models++ (*RNNs, Bi-RNNs, LSTMs, GRUs and variations*), the usual choice for the modeling of medium-length, 1D, time-dependent but generally noiseless sequences,
    - ++Hierarchical convolutional cascades++ (*or fully-CNN architectures in general*), a relatively new - but extremely well-performing - approach to short-term pattern or highly structured time-dependent modeling,
    - ++Token transformers++ (*or dense attentional architectures in general*) which, though appealing from the modeling standpoint, strongly suffer (quadratically) from the time series multidimensionality, and have been originally developed for sequences of *discrete, dictionarized tokens* (as in language models, task in which they attained unrivalled success).

More promising *hybrid* or *evolved architectures*, such as *(hierarchical) CNN-RNN*, *CNN-Transformers* with sparse attention or extremely more recent and radical proposals which blend classical (but differentiable) time-dependent analysis blocks with more *typical* *differentiable programming blocks* (e.g. *FT-Transformers*, *Wavelet-CNN*) match much more closely the ideal requirements, but lack extensive testing in time series modeling (being them usually developed for language modeling or machine translation tasks) - let alone in the financial domain - if not altogether.

Driven by genuine curiosity in *the new*, and in an attempt to slightly tighten the gap between promising novel proposals and real-world battle-testing, on this last class of models our work will be centered around - being us also aware of the limits and potential pitfalls it may carry. Among those, priority will be given to architectures trying to decouple *local, short patterns* (effectively modeled via convolution) and *global, more far-reaching trends* (modeled with sparse self-attention). Furthermore, additional *variationalization* of predictions may provide the uncertainty quantification looked for.  

Given the relative novelty of such experimentation, expected results are hard to figure at this time, being open to the widest variability in outcomes: from moderate success to complete failure. Our optimistic hope is that - at least - such work could help to shed some fleeble light on the nature of the learnability of (a restricted part of) the financial system and to suggest potential paths to tread (or not to!) for its modeling with cutting edge deep learning technologies.

### Minimal Bibliography

(Sketch of links just to have an idea of the main papers at our disposal)

- Transformer for time-series:  https://papers.nips.cc/paper/2019/file/6775a0635c302542da2c32aa19d86be0-Paper.pdf
- FFT speed-up:  https://arxiv.org/abs/2105.03824
- Reformer:  https://arxiv.org/abs/2001.04451
- Linformer:  https://arxiv.org/abs/2006.04768
