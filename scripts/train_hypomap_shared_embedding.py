import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import anndata as ad
import scanpy as sc
import scvi


def load_hypomap(fpath, annotation_column='C25_named'):
    """A function to loadthe hypomap data"""
    adata_ref = sc.read(fpath)
    adata_ref = adata_ref.raw.to_adata() # recover raw counts
    # assign the celltype from the cluster 25 column
    adata_ref.obs["celltype"] = adata_ref.obs[annotation_column] # note that there are other cluster results in .obs
    adata_ref.obs["batch"] = 'reference'
    adata_ref.var_names = adata_ref.var['feature_name'] # set gene names as column headers
    return adata_ref


def load_data(data_path, data={}):
    """A function to load our data 
    
    data may contain completmentary adata objects to be included 
    in the merge
    """
    for f in os.listdir(data_path):
        fullpath = f"{data_path}{f}"
        key = f.replace(".h5ad", "")
        batch_andata = sc.read(fullpath)
        batch_andata.obs['batch'] = key
        batch_andata.obs["celltype"] = 'Unknown'
        data[key] = batch_andata

    adata = ad.concat(data, index_unique="_") # combine all the experiments
    
    # set the counts as the main layer 
    adata.layers["counts"] = adata.X.copy()
    adata.raw = adata
    return adata


def get_predictions(adata):
    """A function to return the predicted cell types for our data"""
    df = adata.obs.copy()
    df = df[df['batch'] != 'reference']
    df = df.reset_index(drop=False)
    return df
    
    

if __name__ == "__main__":

    """ARGS"""
    SAMPLES_PER_EPOCH = 1000
    EPOCHS = 20
    SAMPLE_FRACTION = 1.0
    # ANNOTATION = 'C25_named'
    ANNOTATION = 'C7_named'
    
    
    """Define paths """
    hypomap_reference = "/nfs/turbo/umms-indikar/shared/projects/MC3R/hypomap/hypomap.h5ad"
    data_path = "/nfs/turbo/umms-indikar/shared/projects/MC3R/h5ad_files/" # path to our cleaned .h5ad files
    output_path = f"/nfs/turbo/umms-indikar/shared/projects/MC3R/hypomap/predictions_{ANNOTATION}.csv"

    """ Load data """
    adata_ref = load_hypomap(hypomap_reference, 
                             annotation_column=ANNOTATION)

    ## sample the reference data
    adata_ref = sc.pp.subsample(adata_ref, 
                                fraction=SAMPLE_FRACTION, 
                                copy=True)

    # pass reference data into the custom loader
    data = {
        'reference' : adata_ref
    }
    
    adata = load_data(data_path, data)

    print()
    print(adata.obs['celltype'].value_counts())
    print()

    """ Determine a combined embedding """
    scvi.model.SCVI.setup_anndata(adata, 
                                  layer="counts", 
                                  batch_key="batch",)
    
    model = scvi.model.SCVI(adata, 
                            n_layers=2, 
                            n_latent=30)

    """ TRAIN THE EMBEDDING """
    model.train(max_epochs=EPOCHS)

    # add latent representation to data object
    SCVI_LATENT_KEY = "X_scVI"
    adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

    # set up the label transfer
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        model,
        adata=adata,
        labels_key="celltype",
        unlabeled_category="Unknown",
    )

    """ TRAIN THE LABEL TRANSFER """
    scanvi_model.train(max_epochs=EPOCHS, 
                       n_samples_per_label=SAMPLES_PER_EPOCH)

    # add new latent representation and predictions to data object
    SCANVI_LATENT_KEY = "X_scANVI"
    adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
    adata.obs['predicted_celltype'] = scanvi_model.predict(adata)

    """Extract the predictions """
    pred = get_predictions(adata)

    pred.to_csv(output_path, index=False)
    print(f"saved: {output_path}")

    print()
    print(pred['predicted_celltype'].value_counts())
    print()
    
    

    
    


