# TensorFlow 2 Keras implementation of TabNet
A TensorFlow 2 Keras implementation of TabNets in the paper: [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442). The authors propose TabNet architecture as a neural network that learns a canonical representation of tabular data. This architecture performs competitively with the state-of-the-art tabular data learning methods like XGBoost, CatBoost, and LightGBM. TabNet is also interpretable that generates individualized feature importance.

<u>Citation</u>: Arık, S. O., & Pfister, T. (2020). Tabnet: Attentive interpretable tabular learning. *arXiv*.

This implementation closely follows [the TabNet implementation in PyTorch here](https://github.com/dreamquark-ai/tabnet/tree/b6e1ebaf694f37ad40a6ba525aa016fd3cec15da). The description of that implementation is [explained in this helpful video by Sebastian Fischman](https://www.youtube.com/watch?v=ysBaZO8YmX8). In my opinion, this is the most reliable and flexible implementation of TabNets that I could find.

Yet I re-implement TabNets here, in TensorFlow 2 Keras, mainly to allow the re-use and experiment with this architecture from within the TensorFlow ecosystem and enable the users to take advantage of the Keras API.



