> Code for the paper _EnsembleSplice: Ensemble Deep Learning for Splice Site Prediction_
# EnsembleSplice: Ensemble Deep Learning for Splice Site Prediction


**OluwadareLab,**
**University of Colorado, Colorado Springs**

----------------------------------------------------------------------
**Developers:** <br />
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Trevor Martins<br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Department of Mathematics <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Oberlin College, Oberlin, OH <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Email: trevormartin4321@gmail.com <br /><br />
		 
		 <br />
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Victor Akpokiro<br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Department of Computer Science <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;University of Colorado, Colorado Springs<br />
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


**2.	Validation :**
-----------------------------------------------------------
> _EnsembleSplice Validation: To perforn validation training_
Usage: To train, type in the terminal `python3 exec.py [--train] [--donor, --acceptor] [--cnn1, --cnn2, --cnn3, --cnn4, --dnn1, --dnn2, --dnn3, --dnn4] [--hs3d_bal, --ar, --hs2] [esplice] ` <br />
For Example: `python exec.py -validate --donor --dnn1 --dnn2 --dnn3 --dnn4 --cnn1 --cnn2 --cnn3 --cnn4 --hs3d_bal --esplice" ` <br />

* **Outputs**: <br />
The outputs of training includes: <br />
	* .h5: The deepslicer model file.
	* .txt: The output files (.txt) containig the evaluation metrics results is stored in the log directory.


**3.	Training :**
----------------------------------------------------------- 
> _EnsembleSplice Training: To perforn training and saving_
Usage: To train, type in the terminal `python3 exec.py [--train] [--donor, --acceptor] [--cnn1, --cnn2, --cnn3, --cnn4, --dnn1, --dnn2, --dnn3, --dnn4] [--hs3d_bal, --ar, --hs2] [esplice] ` <br />
For Example: `python exec.py -train --donor --dnn1 --dnn2 --dnn3 --dnn4 --cnn1 --cnn2 --cnn3 --cnn4 --hs2 --esplice" ` <br />

* **Outputs**: <br />
The outputs of training includes: <br />
	* .h5: The deepslicer model file.
	* .txt: The output files (.txt) containig the evaluation metrics results is stored in the log directory.	
                          		
                           
**4.	Testing :**
-----------------------------------------------------------
> _EnsembleSplice Testing: To perforn testing_
For Testing, use `python3 exec.py [--test,] [--donor, --acceptor] [--cnn1, --cnn2, --cnn3, --cnn4, --dnn1, --dnn2, --dnn3, --dnn4] [--hs3d_bal, --ar, --hs2] [esplice] ` <br />
For Example: `python exec.py -test --donor --dnn1 --dnn2 --dnn3 --dnn4 --cnn1 --cnn2 --cnn3 --cnn4 --ar --esplice` <br />
ither balanced or imbalanced input dataset, i.e ("balanced" or "imbalanced")<br />

* **Outputs**: <br />
The outputs of testing includes: <br />
	* .txt: The output files (.txt) containig the evaluation metrics results is stored in the log directory.


**5.	Note:**
-----------------------------------------------------------
* Ensure you have a log directory for text file storage



