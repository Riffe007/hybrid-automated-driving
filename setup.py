from setuptools import setup, find_packages

setup(
    name='mdn_rnn',
    version='1.0.0',
    description='A TensorFlow implementation of MDN-RNN for complex sequential data modeling.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=['tensorflow>=2.0', 'numpy'],
)
