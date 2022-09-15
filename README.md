# DeepSimulator

DeepSimulator is a hybrid tool between DDS and DL techniques to simulate business processes
Prerequisites
To execute this code, you need to install Anaconda in your system, and create an environment using the environment.yml specification provided in the repository.
Running the script
1.	Once created the environment, you can perform each one of the tasks, specifying the following parameters in the pipeline.py module, or by command line as is described below:

parms['t_gen']['emb_method']" = "emb_dot_product" # emb_dot_product, emb_w2vec

parms['t_gen']['max_eval'] = 12 # 12,1 for test

parms['t_gen']['epochs'] = 200 # 200,2 for test

This will trigger the type of embedding that wants to be executed, for testing is recommended to reduce the number of tests to 1 max_eval and 2 epochs.

2.	Now that the pipeline.py is executed with all the parameters it triggers the deep_simulator module. This module is used as a master class to run the rest of the program.

When executing the instruction “Generate instances times the module” Line 60 it calls to the following method DeepSimulator/core_modules/times_allocator/times_generator 

3.	If the model doesn´t exist (line67), then the time optimizer needs to be runned. The timesModelOptimizer needs an embedded model to be discovered to be executed (line 224) 

4.	The embedder method DeepSimulator/core_modules/times_allocator/embedder reads the parameter set on step number one to execute the embedder method

5.	DeepSimulator/core_modules/times_allocator/embedding_trainer uses dot product embedding method to generate a model that uses activities sequences as an input to gather information about the process

6.	DeepSimulator/core_modules/times_allocator/embedding_word2vec is the second embedding method and uses three different inputs to learn: times, activities and roles:

To read these process attributes a different approach is taken:

7.	for times: The DeepSimulator/core_modules/support_modules/writers/log_Reader  module uses the method master duration to read the times in a trace, calculate the number of bins and finally discretize the sequence sending the information to embedding_word2vec 

8.	For Activities DeepSimulator/core_modules/support_modules/writers/log_Reader  module uses the method get_sentences_XES to extract all the 

9.	For Roles: DeepSimulator/core_modules/support_modules/writers/role_discovery is a specialized method to get all the roles from a process trace

10.	In the file WordEmbedding.ipynb additional implementation can be found:

11.	Tests to get the times and discretization

12.	Distance matrix g plots from a process information matrix

13.	Transformers implementation
