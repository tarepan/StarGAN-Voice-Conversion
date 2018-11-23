* IN as an alternatives of BN
* mean as alternatives of product in D/C last layer
* (in only C) max-pooling as alternatives of strided-Conv

## last layer of D & C takes "mean" of the probabilities, not "product"
In original article, probabilities (probability of each patches) is multiplied (== product).  

> the final output D(y, c) is given by the product of all these probabilities.  
> ...
> “Product” denote ... product pooling layers,

But in this implementation, in last layer, probabilities is taken average.  

> c1_red = tf.reduce_mean(c1, keepdims=True) [code](https://github.com/hujinsen/StarGAN-Voice-Conversion/blob/00abc81acddda348188574355f9d4e5b0284b32c/module.py#L239)

Is this intended implementation based on your experiments, or some other reasons?

# resolved
* concat of class => 1-channel : 1 hot (e.g. 4 class => 4 additional channels)
