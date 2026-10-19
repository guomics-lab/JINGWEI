import pandas as pd
from pimmslearn.sklearn.ae_transformer import AETransformer
import numpy as np


def impute(full_data, full_mask):
    full_data = full_data * full_mask
    full_data[full_mask == 0] = np.nan
    try:
        df = full_data.copy()
        df = pd.DataFrame(df)

        model = AETransformer(
            model='DAE',
            hidden_layers=[512,],
            latent_dim=50,
            batch_size=10,
           )

        model.fit(
            df,
            cuda=False,
            epochs_max=200
        )

        df_imputed= model.transform(df)

    except Exception as e:
        print(f"    Error in PIMMS: {e}")


    return df_imputed.values
