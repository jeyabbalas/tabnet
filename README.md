# TensorFlow 2 Keras implementation of TabNet
A TensorFlow 2 Keras implementation of TabNet from the paper: [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442). The authors propose TabNet, a neural network architecture capable of learning a canonical representation of tabular data. This architecture is shown to perform competitively or better than the state-of-the-art tabular data learning methods like XGBoost, CatBoost, and LightGBM. TabNet is also interpretable i.e., they can generate both global and individualized feature importance.

<u>Citation</u>: Arık, S. O., & Pfister, T. (2020). Tabnet: Attentive interpretable tabular learning. *arXiv*.

This implementation closely follows [the TabNet implementation in PyTorch linked here](https://github.com/dreamquark-ai/tabnet/tree/b6e1ebaf694f37ad40a6ba525aa016fd3cec15da). The description of that implementation is [explained in this helpful video by Sebastian Fischman](https://www.youtube.com/watch?v=ysBaZO8YmX8). In my opinion, this is the most reliable and flexible implementation of TabNet that I could find. I was unable to find any good, reliable, and flexible implementation of TabNet in TensorFlow.

I re-implement TabNet in TensorFlow 2 Keras here mainly to enable the re-use and experimentation with this architecture from within the TensorFlow ecosystem and to be able to take advantage of the Keras API. 

## Usage









