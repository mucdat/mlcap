

''' Abstract class to allow users to apply transformations to the original data such as
- outlier imputation
- coordinate transformations

They can apply a transformation to a function by implementing this interface and invoking the transformation with the .transform() method of BBfcn'''

from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from .utils import fcn_as_df, format_query

class Transformation(ABC):
    @abstractmethod
    def apply(self,data):
        pass


import itertools

'''# Main data structure - BBfcn (Black Box Function)
- Contains all known data points of the black box function as read from the directory /measurements/latest
- Implements methods to fit surrogate functions, grid acquisition functions, make graphs, estimate correlation coefficients, etc.
'''

class BBfcn:

    latest_data_dir = os.path.join(".","measurements","latest")

    def __init__(self,function_number):
        self.function_number = function_number
        self.function_name = f"function_{function_number}"
        self.data_dir = os.path.join(self.latest_data_dir,self.function_name)
        self.__inputs = np.load(os.path.join(self.data_dir, 'inputs.npy'))
        self.__outputs = np.load(os.path.join(self.data_dir, 'outputs.npy'))
        self.data = fcn_as_df(self.__inputs,self.__outputs)
        self.lower_boundsX = np.zeros(self.__inputs.shape[1])
        self.upper_boundsX = np.ones(self.__inputs.shape[1])
        self.update()

    # display the function
    def __str__(self):
        print(f"------------ {self.function_name} --------------")
        print(self.data)
        print(f"\nMax of {self.max} realized at row")
        print(self.argmax)
        return ""
    
    # update internal statistics after data modification
    def update(self):
        if self.data is None or self.data.empty:
            self.max = None
            self.argmax = None
            self.argmax_as_np = None
            self.input_dimension = None
        else:
            self.max = self.data["y"].max()
            self.input_dimension = self.data.shape[1]-1
            if self.max is None or np.isnan(self.max):  
                self.argmax = None
                self.argmax_as_np = None
            else:
                self.argmax = self.data.loc[[self.data["y"].idxmax()]]
                self.argmax_as_np = self.argmax.iloc[0,:-1].to_numpy()  # exclude last column (y)



    # API to allow user to apply a transformation or otherwise modify the data
    # they must implement tr, an instance of Transformation, including implementing mandatory method Transformation.apply()
    def transform(self, tr: Transformation):
        self._orig_data = self.data.copy()
        tr.apply(self.data) # pass the user the data for in-place modification. This should be the only way to modify the data
        self.update()
        
    def plot_feature_correlations(self):
        """
        Plot correlations between features and with target.
        """
        # 3. Pairwise scatterplots
        sns.pairplot(self.data)
        plt.show()

        # 1. Correlation matrix of features
        X_df = self.data.drop("y",axis=1).copy()
        #corr_matrix = X_df.corr()
        #plt.figure(figsize=(10, 8))
        #sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        #plt.title("Feature Correlation Matrix")
        #plt.show()
    
        # 2. Correlation with target
        corr_with_target = X_df.corrwith(self.data["y"])
        plt.figure(figsize=(10, 5))
        sns.barplot(x=corr_with_target.index, y=corr_with_target.values)
        plt.xticks(rotation=45)
        plt.ylabel("Correlation with Target")
        plt.title("Feature Correlation with Target")
        plt.show()

    def plot_correlations_with_cross_terms(self):
        df = self.data.copy()
    
        # original feature columns
        feature_cols = df.columns[:-1]
    
        # Generate all pairwise cross terms
        for col1, col2 in itertools.combinations(feature_cols, 2):
            cross_col_name = f'{col1}*{col2}'
            df[cross_col_name] = df[col1] * df[col2]
    
        # Compute correlation of each column with the target
        corr_with_target = df.corr()["y"].sort_values(ascending=False)
    
        # Plot
        plt.figure(figsize=(10, max(4, len(corr_with_target)/2)))
        sns.barplot(x=corr_with_target.values, y=corr_with_target.index)
        plt.title(f'Correlation of features with target')
        plt.xlabel('Correlation')
        plt.ylabel('Feature')
        plt.show()

    def ucb(self,X_candidate, gp, kappa=2.0):
        mu, sigma = gp.predict(X_candidate, return_std=True)
        return mu + kappa * sigma

    def nextPointBayesianOptim(self,kernel=None,numgrid=20,acqFcn="UCB",beta=0.5):

        # if no kernel provided, use default, which is set up for 0-1 scaled inputs and outputs
        if kernel is None:
            length_scale = [0.5]*self.input_dimension   # repeat for each input dimension
            kernel = C(1, (0.00001, 10)) * Matern(length_scale=length_scale,nu=0.5) + WhiteKernel(noise_level=0.0)
        
         # Fit Gaussian Process to existing data
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, normalize_y=True)
        y_train = self.data.iloc[:,-1].to_numpy()  # output is in last column
        X_train = self.data.iloc[:,:-1].to_numpy() # other columns are inputs 

        # axes for grid
        x = []
        for i in range(self.input_dimension):
            x.append(np.linspace(self.lower_boundsX[i],self.upper_boundsX[i],numgrid))

        # convert list to an np.array of dimension (numgrid x input_dimension)
        x = np.vstack(x).T

        Y_scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaler = MinMaxScaler(feature_range=(0, 1))
        x_scale = X_scaler.fit_transform(x)
        X_train_scale = X_scaler.transform(X_train)
        y_train_scale = Y_scaler.fit_transform(y_train.reshape(-1,1)).ravel()  
        # store in the object for unscaling later
        self.scalerX = X_scaler
        self.scalerY = Y_scaler

        gp.fit(X_train_scale, y_train_scale)
        print("GP kernel used for fitting surrogate=")
        print(gp.kernel)
        #kernel_params = gp.kernel_.get_params()
        #print(kernel_params["k2__length_scale"])     # Access Matern length scale

        # Create grid of candidate points - meshgrid expects a list of 1D arrays, one per dimension, so unpack the columns of x_scale first
        grid_point_lists = [x_scale[:,i] for i in range(self.input_dimension)]
        xvalue_grids = np.meshgrid(*grid_point_lists,indexing="ij") # unpacks the list into arguments
        flat = [g.reshape(-1) for g in xvalue_grids]
        X_for_gp = np.vstack(flat).T
        X_for_gp_orig_scale = self.scalerX.inverse_transform(X_for_gp)
        X_value_grids_orig_scale = self.scalerX.inverse_transform(X_for_gp)
        x_for_graph = [X_value_grids_orig_scale[:,i].reshape(xvalue_grids[0].shape[0],-1) for i in range(self.input_dimension)]

        # Acquisition function      
        mu, sigma = gp.predict(X_for_gp, return_std=True)
        ucb = mu + beta * sigma

        # Select next acquisition point ---
        next_idx = np.argmax(ucb)
        X_next = X_for_gp[next_idx]
        X_next_original_scale = self.scalerX.inverse_transform(X_next.reshape(1,-1)).ravel()
        print("Next acquisition point (UCB):", X_next_original_scale)
        format_query(X_next_original_scale)

        # Interactive plot for 2D
        if self.input_dimension == 2:
            y_GP = mu.reshape(xvalue_grids[0].shape)
            y_GP = self.scalerY.inverse_transform(y_GP.reshape(-1,1)).reshape(y_GP.shape)
            dimension_names = self.data.columns[:-1].tolist()
            # Mean of Gaussian Process
            fig = go.Figure(data=[go.Surface(z=y_GP, x=x_for_graph[0], y=x_for_graph[1], opacity=0.4)])
            fig.update_layout(scene=dict(
                xaxis_title=dimension_names[0],
                yaxis_title=dimension_names[1],
                zaxis_title=f'f({dimension_names[0]},{dimension_names[1]})'))
            # original data points
            fig.add_trace(go.Scatter3d(
                x=X_train[:, 0],
                y=X_train[:, 1],
                z=y_train,
                mode='markers',
                marker=dict(color='red', size=5),
                name='Measured points'))
            fig.show()
        return X_next_original_scale

    # Farthest Point Sampling
    def FPS(self):
        X_train = self.data.iloc[:,:-1].to_numpy() # last column is the target
        tree = cKDTree(X_train)
        num_candidates = 10000
        num_candidates_per_dim = int(np.ceil(num_candidates ** (1 / self.input_dimension))) 
        x = []
        for i in range(self.input_dimension):
            x.append(np.linspace(0,1,num_candidates_per_dim))
        xvalue_grids = np.meshgrid(*x,indexing="ij")
        flat = [g.reshape(-1) for g in xvalue_grids]
        X_candidates = np.vstack(flat).T

        distances, _ = tree.query(X_candidates, k=1)
        # Sort the grid points by descending distance
        idx_sorted = np.argsort(-distances)
        N = 4
        unexplored_points = X_candidates[idx_sorted[:N]] #magic number 4 - pick top 4 most unexplored points
        farthest_idx = np.argmax(distances)
        X_next = X_candidates[farthest_idx]
        # Optional: print or visualize
        print("Most unexplored points (farthest from training data):")
        print(unexplored_points)
        print("point this week according to FPS:", unexplored_points[0,:])      

        # Interactive plot for 2D
        if self.input_dimension == 2:
            dimension_names = self.data.columns[:-1].tolist()
            fig = go.Figure()
            # original data points
            fig.add_trace(go.Scatter3d(
                x=X_train[:, 0],
                y=X_train[:, 1],
                z=self.data.iloc[:,-1].to_numpy(),
                mode='markers',
                marker=dict(color='red', size=3),
                name='Measured points'))
            # candidate points colored by distance to nearest existing point
            fig.add_trace(go.Scatter3d(
                x=X_candidates[:N, 0],
                y=X_candidates[:N, 1],
                z=np.zeros(N),
                mode='markers',
                marker=dict(
                    size=7,
                    color=distances,
                    colorscale='Viridis',
                    colorbar=dict(title='Distance to Nearest Point'),
                ),
                name='Candidate points'))
            fig.update_layout(scene=dict(
                xaxis_title=dimension_names[0],
                yaxis_title=dimension_names[1],
                zaxis_title='Distance to Nearest Point'))
            fig.show()
            
        return X_next
    

class TrustRegion(BBfcn):
    def __init__(self,fcn_number,center_point=None,cube_half_length=0.2):
        super().__init__(fcn_number)
        self.data_full = self.data.copy()  # keep a copy of the full original data
        self.cube_half_length = cube_half_length
        self.center_point = center_point
        self.update_trust_region()

    def set_center(self,center_point):
        self.center_point = center_point
        self.update_trust_region()

    def set_cube_half_length(self,half_length):
        self.cube_half_length = half_length
        self.update_trust_region()

    def set_data_manually(self,dataframe):
        self.data_full = dataframe.copy()
        self.data = dataframe.copy()
        self.update_trust_region()

    def update_trust_region(self):
        self.data = self.data_full.copy()  # reset data to full data
        if self.center_point is None:
            self.data = None
        else:
            lower_bounds = np.maximum(0.0, self.center_point - self.cube_half_length)
            upper_bounds = np.minimum(1.0, self.center_point + self.cube_half_length)
            # filter data to only include points within the trust region
            for i, col in enumerate(self.data.columns[:-1]):
                self.data = self.data[(self.data[col] >= lower_bounds[i]) & (self.data[col] <= upper_bounds[i])]
            self.data.reset_index(drop=True, inplace=True)
            self.lower_boundsX = lower_bounds
            self.upper_boundsX = upper_bounds
        self.update()   # parent class update method to refresh max, argmax, etc.



