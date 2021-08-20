# Welcome to Parameter Identification of Ship Dynamics

This [JupyterBook](https://jupyterbook.org/) has been created by me ([Martin Alexandersson](https://www.linkedin.com/in/martin-alexandersson-77823312/)) to document the research for my Phd. studies.

This book is about ship dynamics: how the motion of ships can be described with physics and math to be used in computer simulations. Expressing the dynamics of a ship as an Ordinary Differential Equation (ODE) is a well established technique. The ODE can be solved as an initial value problem with integration of accelerations and velocities to obtain a ship's trajectory. This is commonly known as a simulation. The workflow of a simulation is to first establish a force model that can estimate the hydrodynamic forces as function of the current state. Accelerations can then be calculated from these forces together with the mass. The velocities and positions can then be determined with time integration of the acceleration.

This research is however about [Inverse dynamics](https://en.wikipedia.org/wiki/Inverse_dynamics), which is reversing the problem above. Instead of estimating a ships trajectory with a force model, we want to identify this force model, by using a measured ship trajectory. This is very useful when you want to fit a mathematical model to the measured ship motion, either obtained from ship model test or the real ship in full scale operation. The latter is something that today is becoming more and more relevant as more and more operational data is measure and recorded onboard the ships. 

So why is it convenient to have a mathematical model of your ship? 

