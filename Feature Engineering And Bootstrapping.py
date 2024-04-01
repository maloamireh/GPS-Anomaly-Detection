import numpy as np
import pandas as pd

filetoUse = 'avg_allAttribute_allLocation_agg.csv'  #I used the dataset I made before which had all trajectories as one row with all attributes to start, I included this csv in the github uplaod just in case
#data init
df = pd.read_csv(filetoUse) 
columnstokeep = ['vz_sd', 'z_sd', 've', 'vu', 'pitch', 'roll', 'yaw', 'qbn_3', 'attacked']  #culling features I've selected based on feature importance from xgboost on above dataset(feel free to change)
df = df[columnstokeep]

#Function that just outputs the dataset with the chosen columns above; no bootstrapping or augmentation done.
def CreateCulledDataset():
    df.to_csv('avg_top10_allLocation_agg.csv')

#Function to bootstrap a number of new ground truth trajectories via resampling with replacement from existing ones, set augment = True if you want to augment instead(generate new samples via gaussian noise addition)
def bootstrapGrounds(n_additional_rows, augment):
    original_ground_truth_trajectories = df[df['attacked'] == 0]

    # Sample from subset where 'attacked' is 0 with replacement
    bootstrapped_attacked_0 = original_ground_truth_trajectories.sample(n=n_additional_rows, replace=True)

    # Concatenate sampled rows with original DataFrame
    bootstrapped_df = pd.concat([df, bootstrapped_attacked_0])

    # Reset index after concatenation
    bootstrapped_df.reset_index(drop=True, inplace=True)
    bootstrapped_df.to_csv('avg_top10_bootstrapped.csv', index=False)

    if augment:
        columnstoAugment = ['vz_sd', 'z_sd', 've', 'vu', 'pitch', 'roll', 'yaw', 'qbn_3']
        noise_level = .1
        for attr in columnstoAugment:
            bootstrapped_df[attr] = add_noise(bootstrapped_attacked_0[attr], noise_level)
        bootstrapped_df.to_csv('avg_top10_augmented.csv', index=False)



#function to augment the data via creating new ground truth trajectories via resampling + gaussian noise addition   ***OBSOLETE BUT I MIGHT USE IT LATER***
def Augment(num_to_bootstrap,  noise_level, num_to_augment):
    # Define noise level
    original_ground_truth_trajectories = df[df['attacked'] == 0]

    # Sample from subset where 'attacked' is 0 with replacement
    bootstrapped_attacked_0 = original_ground_truth_trajectories.sample(n=num_to_bootstrap, replace=True)
    bootstrapped_attacked_0.reset_index(drop=True, inplace=True)

    # Apply noise augmentation to selected attributes
    columnstoAugment = ['vz_sd', 'z_sd', 've', 'vu', 'pitch', 'roll', 'yaw', 'qbn_3']
    for attr in columnstoAugment:
        bootstrapped_attacked_0[attr] = add_noise(bootstrapped_attacked_0[attr].iloc[0:num_to_augment], noise_level)
    bootstrapped_df = pd.concat([df, bootstrapped_attacked_0])

    # Reset index after concatenation
    bootstrapped_df.reset_index(drop=True, inplace=True)
    bootstrapped_df.to_csv('avg_top10_augmented.csv',index=False)

def add_noise(x, noise_level):
    return x + np.random.normal(loc=0, scale=noise_level, size=len(x))



#Main
bootstrapGrounds(1000,augment=False)