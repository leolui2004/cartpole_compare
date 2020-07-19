# Compare between Dense and LSTM using CartPole
My sixth project on github. Compare efficiency and effectiveness between simple Dense layer and LSTM layer using Cart Pole

Feel free to provide comments, I just started learning Python for 3 monnths and I am now concentrating on data anylysis and web presentation.

## Reason to Compare
I started to learn Reinforcement Learning last month and the simplest way to build the model is just using Dense layer(sometimes with Convolutional Layer for images), however it should be only possible to learn one input(state) followed by one output(action), if we are going to solve more difficult problem which requires an input of series, something like Taxi-v3, MontezumaRevenge-v0 which a series of consecutive actions are needed, traditional method should not be possible to solve.

To see if LSTM is really able to implement in Reinforcement Learning, I started with a simple game CartPole-v0. 

## Methodology
In short, both of them use Tensorflow, a modified A2C model, Adam optimizer and same hyperparameters(e.g. Learning Rate, etc.).
Memory are cleared every 10 timesteps because I do not want the LSTM model to process with a too long series, and model is only trained(apply_gradients) when the length of the memory is larger than 8 as a kind of keeping only performances with good result.

The traditional one uses 1 Dense layer with 128 units connected to 1 Dense layer with 32 units.
The LSTM one uses 3 LSTM layer with 32, 64, 128 units and 0.2 Dropout respectively, followed by 3 Dense layer with 64, 32, 16 units.
The LSTM model has much more layers as to hold the stability. This may not be rigorous enough but I tried to keep other parts same as possible as I can and below you will find the result which should not have too much influences on the number of layers.

## Result
The result is pretty impressing, the traditional one, trained with only 1000 episodes shows an average timestep of 120, where the LSTM one, trained with 30000 episodes shows an average timestep of 33.

There are 2 implications I want to point out:
1. LSTM model is able to implement in Reinforcement Learning
2. LSTM model is extremely difficult to train to the point that one may think it is not work at all if trained for 1000 or 2000 episodes even for a simple Cart Pole game.

Upper one is traditional, lower one is LSTM

![image](https://github.com/leolui2004/cartpole_compare/blob/master/cartpole_timestep.png)

## One More Thing
Further look at the graph above, the LSTM model is quite stable in terms of average performance, where the traditional one has some big fluctuation. Indeed when I tried to train the traditional model, I encountered a situation of Catastrophic Forgetting once, I do not know the exact reason behind but I believe it is some kind of similar to the big fluctuation saw on above, so the stability of LSTM model may help although dozens of times of training is needed.

![image](https://github.com/leolui2004/cartpole_compare/blob/master/cartpole_catastrophic.png)

Actually before trying to compare, I already searched on the net and knew that LSTM is very difficult to train, and turnout the result is much more serious than I think. However I will still try to implement LSTM into other project as I think using only traditional one would not be enough to solve a more complex problem.
