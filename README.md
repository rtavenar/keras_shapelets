Efficient implementation of Learning Time-Series Shapelets using `keras`
---

This code offers a Python implementation of the work presented in:

> Josif Grabocka, Nicolas Schilling, Martin Wistuba, Lars Schmidt-Thieme (2014): _Learning Time-Series Shapelets._
> In Proceedings of the 20th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2014

This implementation builds upon the `keras` library (basically, you will need `keras`, `tensorflow` and `numpy` to be 
installed) for efficient optimization of the Shapelet coefficients.

As an example, it takes roughly 1 minute (on a standard MacBook Pro laptop) for training on the Trace dataset from 
UCR/UEA repository.

This code is now integrated into the `tslearn` toolkit.
Have a look [there](https://github.com/rtavenar/tslearn) if you are interested.
