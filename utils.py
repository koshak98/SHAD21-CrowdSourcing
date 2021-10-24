from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class CostMetrics():
    def __init__(self, df):
        self.df = df
    
        
    def get_cumlift(self, trip_col='exp_n_trips', discount_col='exp_discount_value', treatment_col='w',
                    random_seed=13, random_col_name='Random'):
        """
        Customized function of comlift from here:
        https://github.com/uber/causalml/blob/master/causalml/metrics/visualize.py#L51
        Args:
            df (pandas.DataFrame): a data frame with model estimates and actual data as columns
            trip_col (str, optional): the column name for the number of trips in experimet
            discount_col (str, optional): the column name for the sum of discount recieved
            treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
            random_seed (int, optional): random seed for numpy.random.rand()
        Returns:
            (pandas.DataFrame): average uplifts of model estimates in cumulative population
        """
        
        df = self.df.copy()
        assert (trip_col in df.columns) and (discount_col in df.columns) and (treatment_col in df.columns)

        np.random.seed(random_seed)
        random_cols = []
        for i in range(20):
            random_col = '__random_{}__'.format(i)
            df[random_col] = np.random.rand(df.shape[0])
            random_cols.append(random_col)

        model_names = [x for x in df.columns if x not in [trip_col, discount_col, treatment_col]]

        lift = []
        for i, col in enumerate(model_names):
            sorted_df = df.sort_values(col, ascending=False).reset_index(drop=True)
            sorted_df.index = sorted_df.index + 1

            sorted_df['cumsum_tr'] = sorted_df[treatment_col].cumsum()
            sorted_df['cumsum_ct'] = sorted_df.index.values - sorted_df['cumsum_tr']

            sorted_df['cumsum_trips_tr'] = (sorted_df[trip_col] * sorted_df[treatment_col]).cumsum()
            sorted_df['cumsum_discount_tr'] = (sorted_df[discount_col] * sorted_df[treatment_col]).cumsum()

            sorted_df['cumsum_trips_ct'] = (sorted_df[trip_col] * (1 - sorted_df[treatment_col])).cumsum()
            sorted_df['cumsum_discount_ct'] = (sorted_df[discount_col] * (1 - sorted_df[treatment_col])).cumsum()

            mean_discount_tr = sorted_df['cumsum_discount_tr'] / sorted_df['cumsum_tr']
            mean_discount_ct = sorted_df['cumsum_discount_ct'] / sorted_df['cumsum_ct']

            mean_trips_tr = sorted_df['cumsum_trips_tr'] / sorted_df['cumsum_tr']
            mean_trips_ct = sorted_df['cumsum_trips_ct'] / sorted_df['cumsum_ct']

            lift.append((mean_discount_tr - mean_discount_ct) / (mean_trips_tr - mean_trips_ct))

        lift = pd.concat(lift, join='inner', axis=1)
        lift.loc[0] = np.zeros((lift.shape[1], ))
        lift = lift.sort_index().interpolate()

        lift.columns = model_names
        lift[random_col_name] = lift[random_cols].mean(axis=1)
        lift.drop(random_cols, axis=1, inplace=True)

        self.df_lift = lift


    def plot(self, n_steps=10, figsize=(8, 8)):
        idx = np.quantile(self.df_lift.index, q = np.arange(0,1.1,0.1))
        self.df_lift.iloc[idx].plot(figsize=figsize, marker='o')
        plt.xlabel('Population')
        plt.ylabel('Cost of trip')
        
    def auuc_score(self):
        return self.df_lift.sum() / self.df_lift.shape[0]