> Code for the paper _EnsembleSplice: Ensemble Deep Learning Model for Splice Site Prediction_
# EnsembleSplice: Ensemble Deep Learning Model for Splice Site Prediction


**OluwadareLab,**
**University of Colorado, Colorado Springs**

----------------------------------------------------------------------
**Developers:** <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Trevor Martin<br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Department of Mathematics <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Oberlin College, Oberlin, OH <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Email: trevormartin4321@gmail.com <br /><br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Victor Akpokiro<br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Department of Computer Science <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;University of Colorado, Colorado Springs <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Email: vakpokir@uccs.edu <br /><br />

**Contact:** <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Oluwatosin Oluwadare, PhD <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Department of Computer Science <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;University of Colorado, Colorado Springs <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Email: ooluwada@uccs.edu 
    
--------------------------------------------------------------------


**1.	Dependencies:**
-----------------------------------------------------------
We have attached the requirement file for the list of dependencies. For local install of dependencies from the <i>requirement.txt</i> file for virtual environment usage, use command `pip install -r requirement.txt` from the current working directory. 

This project is compatible with Anaconda environments.

When EnsembleSplice is run for Validation, Training, or Testing two things occur. First, a file with the sub-networks, splice sites, dataset, and other relevant information in its name is created. This is a text file containing dictionaries of output results. To have the results printed in the terminal or on Colab, move ENS_Temp_Run.py into ./Logs/ and run it. You can also create a new folder and move the log files and ENS_Temp_Run.py into this folder and then run ENS_Temp_Run.py. Second, the trained sub-networks and their weights are added to ./Models/TrainedModels/. 

To run the actual ensemble, make sure the argument `--esplice` is used. The specified sub-networks are the only models that run when this argument is not used, and results outputs are produced for each submodel. 


**2.	Validation :**
-----------------------------------------------------------
> _EnsembleSplice Validation: To perform validation training_
Usage: To train, type in the terminal `python3 exec.py [--train] [--donor, --acceptor] [--cnn1, --cnn2, --cnn3, --cnn4, --dnn1, --dnn2, --dnn3, --dnn4] [--hs3d_bal, --ar, --hs2] [--esplice] ` <br />
For Example: `python exec.py -validate --donor --dnn1 --dnn2 --dnn3 --dnn4 --cnn1 --cnn2 --cnn3 --cnn4 --hs3d_bal --esplice" ` <br />

* **Outputs**: <br />
The outputs of training includes: <br />
	* .h5: The deepslicer model file.
	* .txt: The output files (.txt) containig the evaluation metrics results is stored in the log directory.


**3.	Training :**
----------------------------------------------------------- 
> _EnsembleSplice Training: To perform training and saving_
Usage: To train, type in the terminal `python3 exec.py [--train] [--donor, --acceptor] [--cnn1, --cnn2, --cnn3, --cnn4, --dnn1, --dnn2, --dnn3, --dnn4] [--hs3d_bal, --ar, --hs2] [--esplice] ` <br><br>
For Example: `python exec.py -train --donor --dnn1 --dnn2 --dnn3 --dnn4 --cnn1 --cnn2 --cnn3 --cnn4 --hs2 --esplice" ` <br />

See `exec.py` for more details. 

* **Outputs**: <br />
The outputs of training includes: <br />
	* .h5: The deepslicer model file contained in ./Models/TrainedModels/
	* .txt: The output files (.txt) containig the evaluation metrics results is stored in the log directory.	
                          		
                           
**4.	Testing :**
-----------------------------------------------------------
> _EnsembleSplice Testing: To perform testing_
For Testing, use `python3 exec.py [--test,] [--donor, --acceptor] [--cnn1, --cnn2, --cnn3, --cnn4, --dnn1, --dnn2, --dnn3, --dnn4] [--hs3d_bal, --ar, --hs2] [--esplice] ` <br />
For Example: `python exec.py -test --donor --dnn1 --dnn2 --dnn3 --dnn4 --cnn1 --cnn2 --cnn3 --cnn4 --ar --esplice` <br />
ither balanced or imbalanced input dataset, i.e ("balanced" or "imbalanced")<br />

* **Outputs**: <br />
The outputs of testing includes: <br />
	* .txt: The output files (.txt) containig the evaluation metrics results is stored in the log directory.

**5.	Note:**
-----------------------------------------------------------
* Ensure you have a log directory for text file storage
