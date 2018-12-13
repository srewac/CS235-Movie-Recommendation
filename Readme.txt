# Code structure explained

We implemented three approaches of making a movie recommendation system, as described in the project report

The first one being the LFM. All the computation concerning LFM is done in LFM.py

The second one being the user-based model. All the computation concerning the user-based model is done in UserBasedModel.py

The third one is the hybrid approach. The creating user agent part is in the main.py and then the computation is hand over to the LFM. 

# How to use
Run main.py

Each approach is called in main.py. Where rate_engine is used for the LFM and rate_engine2 is used for the user-based model. 
To use the hybrid approach, set enable_hybrid = True. that would use a user agent for the given user instead of a randomly generated matrix.
Note: when using the hybrid approach, the number of features used in the LFM must be the same as the number of genres.

# Generated file explained
feature.txt is the user preference toward a given genre. 
user_info_based.txt is the user agent for jobs, age and gender
neighbour.txt is the neighbor for a given user i. File is written in append mode, so that we can map the user neighbor after predicting enough users.