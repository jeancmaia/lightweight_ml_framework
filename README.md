A lightweight framework for machine learning experiments and productization. The framework aids in applying good practices of model evaluation for fast prototyping and deployment. Any model wrapped as a Sklearn pipeline might easily be deployed with this project.     
Caveat: It only covers Binary Classification.    

----------------

#### Setting up

A ML Project starts with EDA and Model Evaluation. The ML practioner must raise a jupyter notebook server and create on-demand notebook on directory `nbks`. Poetry helps on starting up a server.

```make jupyter```

The regular data pulling step might be executed with the support of script: `script/data_pulling.py`, with simple parametrization of the origin URL.  To downloas the data:

```make data-pulling```

Having the complex process of EDA and model evaluation knocked off, the model can be deployed in two simples steps:

1 - To implement the raw preprocessing on the script base: `script/data_pulling.py`.    
2 - To define the model pipeline in the method `Model` of class `ml_experiments.model.py`.     
    
Then, the entire model training flow is ready to proceed.

----------------

#### Triggering the model

1 - To train a model   
```make data-full-preprocessing```   
```make train-model```   

2 - To analyze the final results interactively

As long as the model training has succeded, the results might be analyzed on notebook: `nbks/DEFAULT_log_analysis.ipynb`

3 - To deploy a model  

The API can be raised either as a docker container or local env with poetry. The commands are:

local:     
```make api-server-poetry```    


docker:   
```make docker-build-server```
```make docker-run-server```

4 - To request the model   
   
The notebook `nbks/API_requests_test.ipynb` has a simple way to request the API.    


----------------
Index:

> scripts:
    - Outside scripts for general tasks.   
> API:
    - Flask API server for the endpoint.   
> assets:
    - Path for outputs of training.   
> docker:
    - Path for docker implementations.   
> nbks:
    - Path for notebook directories.  
> ml_experiments:
    - ML lightweight framework.   

----------------

TODO: To extend this framework for multi-classification and regression.