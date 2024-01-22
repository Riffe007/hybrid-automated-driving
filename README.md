# Advanced Hybrid AI Model for Automated Driving

## Overview
This repository presents an advanced hybrid AI model for automated driving, integrating several cutting-edge technologies including CNN-VAE (Convolutional Neural Networks and Variational Autoencoders), MDN-RNN (Mixture Density Network-Recurrent Neural Network), Reinforcement Learning (RL), Neuroevolution, and the Full World Model.

## Features
- **CNN-VAE**: Processes and encodes visual input from the vehicle's environment.
- **MDN-RNN**: Models the temporal dependencies within the sequential data.
- **Reinforcement Learning**: Facilitates decision-making and control.
- **Neuroevolution**: Optimizes the neural network architecture for enhanced performance.
- **Full World Model**: Integrates all components into a comprehensive simulation environment.

## Installation
To set up the project environment:

```bash
git clone https://github.com/Riffe007/hybrid-automated-driving.git
cd automated-driving-ai
pip install -r requirements.txt

## Configuration

Fine-tune the model parameters and settings by editing the config.py file. This allows customizing the model behavior to fit specific driving scenarios and datasets.

## Usage

Import the necessary components and initialize them with the desired configurations:

from cnn_vae import CNNVAE
from mdnrnn import MDNRNN
from rl_agent import RLAgent
from neuroevolution import Neuroevolution
from full_world_model import FullWorldModel
from config import Config

# Initialize components
cnn_vae = CNNVAE(Config)
mdn_rnn = MDNRNN(Config)
rl_agent = RLAgent(Config)
neuroevolution = Neuroevolution(Config)
world_model = FullWorldModel(Config)

# Model training, evaluation, and driving simulation
# world_model.train(...)
# world_model.evaluate(...)
# world_model.simulate(...)

## Contributing

Your contributions are welcome! For major changes, please open an issue first to discuss what you would like to change. Ensure to update tests as appropriate.

## License

Distributed under the MIT License. See LICENSE for more information.

## Contact

For any queries or collaboration, feel free to contact Tim Riffe.
