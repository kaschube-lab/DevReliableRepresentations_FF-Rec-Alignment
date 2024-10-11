# DevReliableRepresentations_FF-Rec_Alignment
Code accompanying paper 'The developmental emergence of reliable cortical representations'  (Trägenap et al, 2024)

The folder 'FeedforwardRecurrentAlignment_linear' is set up as a python project to reproduce the computational model figures. To run the code, the local python path needs to contain this folder - this can be achieved either via creating a 'project' (e.g. in Spyder) or manually setting the path.

Wihtin the project, the folder 'tools' contains a Class defining a Linear Recurrent neural network and functions to create random, symmetric interaction matrices (networks_definitions.py), diverse helper functions to constrsuct low-dimensional, Gaussian covariance matrices, as well as plotting definition functionalities (plotting_funcs.py).
The folder 'Analysis' contains various functions to quantify and illustrate Feedforward-Recurrent Alignment (Alignment_funcs.py).

The folder 'Alignment_response_properties' illustrates the influence of FF-Rec-Alignment on response properties and reproduces Figs. 7b-f.
Code in folder 'Modelpredictions' illustrates the predictions made by the model on the relationship between response properties and reproduces Figs. 7g and Extended Data Fig. 9.
The folder 'NetworkChanges_Alignment' explores what network changes could underlie increased FF-Rec-Alignment and contains code to reproduce Fig. 8. Here, please run the files {onlyFF, onlyRec, Both}Change_optimal.py and plot the results with plot_Fig8b.py .



