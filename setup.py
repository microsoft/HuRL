from distutils.core import setup

setup(
    name='HuRL',
    version='0.1.0',
    author='Ching-An Cheng, Andrey Kolobov, Adith Swaminathan',
    author_email='chinganc@microsoft.com',
    packages=['hurl'],
    url='https://github.com/microsoft/HuRL',
    license='LICENSE',
    description='Codes to reproduce the experimental results of the Heuristic Guided Reinforcement Learning paper published in NeurIPS 2021.',
    long_description=open('README.md').read(),
    install_requires=[
        "garage==2021.3.0",
        "mujoco_py==2.0.2.8",
        "gym==0.17.2",
    ],
)