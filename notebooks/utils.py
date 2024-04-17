import os
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
from Bio import SeqIO
from Bio.KEGG import REST
from Bio.KEGG.KGML import KGML_parser
from matplotlib import patheffects as pe
import io
from PIL import Image

def standardize(df, axis=1):
    """
    Standardizes a DataFrame (or array-like) along the specified axis, 
    returning a standardized DataFrame.

    Args:
        df (pd.DataFrame or array-like): The input data to standardize.
        axis (int, optional): The axis along which to standardize. 0 for rows, 1 for columns. Defaults to 1.

    Returns:
        pd.DataFrame: The standardized DataFrame with the same columns and indices as the input.
    """

    arr = np.asarray(df)  # Ensure it's a NumPy array

    # Calculate mean and standard deviation along the specified axis
    arr_mean = arr.mean(axis=axis, keepdims=True)
    arr_std = arr.std(axis=axis, keepdims=True)

    # Standardize
    standardized_arr = (arr - arr_mean) / arr_std

    # Convert back to DataFrame, preserving original structure
    return pd.DataFrame(standardized_arr, columns=df.columns, index=df.index)  
    

def plot_top_gene_heatmap(pdf, top_genes, cmap='viridis', fontsize=8):
    """A function to plot a heatmap of top genes """

    gene_list = top_genes['genes'].to_list()
    gene_list = [g.strip() for s in gene_list for g in s.split(', ') if not g == '']

    df = pdf.to_df()
    df = df[gene_list]
    df['cluster'] = pdf.obs['neuron_cluster_name']
    df = df.sort_values(by='cluster')

    df = df.groupby('cluster').mean()
    df = standardize(df, axis=0)

    sns.heatmap(df.T, lw=0.2, 
                cbar_kws={'shrink' : 0.5}, 
                cmap=cmap)
    plt.ylabel("")
    plt.xlabel("")
    plt.tick_params(axis='y', rotation=0)
    plt.tick_params(axis='both', labelsize=fontsize)


def plot_umap_with_labels(pdf, top_genes, x='NU1', y='NU2', hue_column='neuron_cluster_name',
                          color_map="nipy_spectral", title="Differentially Expressed Neuropeptides",
                          label_pad=0):
    """Creates a UMAP scatterplot with labels for neuron clusters, highlighting differentially expressed neuropeptides.

    Args:
        pdf (pd.DataFrame): DataFrame containing UMAP coordinates ('NU1', 'NU2'), cluster assignments 
                            (specified by 'hue_column'), and other metadata.
        top_genes (pd.DataFrame): DataFrame containing top differentially expressed genes with columns
                                  'cluster' and 'genes'.
        hue_column (str, optional): Name of the column in 'pdf' to use for coloring points. 
                                    Defaults to 'neuron_cluster_name'.
        color_map (str, optional): Name of colormap to use. Defaults to 'nipy_spectral'.
        title (str, optional): Title for the plot. Defaults to "Differentially Expressed Neuropeptides".
        label_pad (float): padding for label positions
    """
    # Sort DataFrame and determine hue order
    pdx = pdf.obs.copy().sort_values(by=hue_column).reset_index()
    hue_order = sorted(pdx[hue_column].unique())
    order = np.argsort(pdx[hue_column])[::-1]

    # Create the UMAP scatterplot
    sns.scatterplot(data=pdx.iloc[order],
                    x=x,
                    y=y,
                    hue=hue_column,
                    hue_order=hue_order,
                    palette=color_map, 
                    s=8,
                    alpha=0.85,
                    ec='none')

    # Add labels from top_genes 
    all_pos = pdx.groupby(hue_column)[['NU1', 'NU2']].median().sort_index()

    for label, x_pos, y_pos in all_pos.itertuples():
        new_label = top_genes[top_genes['cluster'] == label]['genes'].values[0]
        plt.text(
            x_pos + label_pad, 
            y_pos,  
            new_label,
            weight='bold',
            color='w',
            path_effects=[pe.withStroke(linewidth=1.5, foreground="k", alpha=0.95)],
            verticalalignment="center",
            horizontalalignment="center",  # Align left to prevent overlapping axis
            fontsize='5',
        )

    plt.gca().legend().remove()  # Remove default legend


def get_top_genes(deg, gene_list=None, n_genes=3, lft=1.5, alpha=0.01):
    """A function to get the top differentially expressed genes (DEGs) from each cluster, 
    optionally filtering by a provided list of genes.

    Args:
        deg (pd.DataFrame): DataFrame containing differential expression results.
        gene_list (list, optional):  A list of genes to consider. If provided, the function
                                     will only select top DEGs from this list. Defaults to None.
        n_genes (int, optional): The maximum number of top genes to return per cluster. Defaults to 3.
        lft (float, optional): Log fold change threshold for filtering DEGs. Defaults to 1.5.
        alpha (float, optional): Adjusted p-value threshold for filtering DEGs. Defaults to 0.01.

    Returns:
         pd.DataFrame: DataFrame with 'cluster' and 'genes' columns.
    """

    sig = deg[deg['logfoldchanges'] > lft]
    sig = sig[sig['pvals_adj'] <= alpha]

    # Filter out predicted genes
    sig = sig[~sig['names'].str.startswith("Gm")]
    sig = sig[~sig['names'].str.endswith("Rik")]

    # Filter based on optional gene_list
    if gene_list is not None:
        sig = sig[sig['names'].isin(gene_list)]  

    sig = sig.sort_values(by=['group', 'logfoldchanges'], ascending=[False, False])

    res = []
    for cluster, group in sig.groupby('group'):
        group = group.head(n_genes)
        genes = ", ".join(group['names'].to_list())
        row = {
            'cluster': cluster,
            'genes': genes,
        }
        res.append(row)

    result_df = pd.DataFrame(res)
    return result_df 
    

def plot_umap_scatter(pdf, x='U1', y='U2', color="Sun1", cmap='viridis', **kwargs):
    """
    Creates a scatterplot of UMAP data with custom color mapping and a colorbar.
    
    Args:
      pdf (pd.DataFrame): DataFrame containing UMAP data.
      x (str): column name of x coord.
      y (str): column name of y coord.
      color (str, optional): Column name for color mapping. Defaults to "Sun1".
      **kwargs (dict, optional): Additional keyword arguments for scatterplot customization.
    
    """
    
    # Extract color values from UMAP data
    v = pdf[:, [color]].X.todense()
    gdf = pdf.obs.copy()
    
    # Add color values as a new column and sort data
    gdf['exp'] = np.ravel(v)
    order = np.argsort(gdf['exp'])
    
    # Create the scatterplot with colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(v), max(v)))
    sm.set_array([])
    p = plt.scatter(gdf[x][order], 
                    gdf[y][order], 
                    c=gdf['exp'][order], 
                    **kwargs)  # Apply kwargs
    
    plt.colorbar(sm, shrink=0.4)
    
    # Additional plot formatting
    sns.despine()
    plt.yticks([])
    plt.xticks([])
    plt.ylabel("UMAP 2")
    plt.xlabel("UMAP 1")
    plt.title(color)
    
    return p  # Return the scatterplot object

def fig2img(fig, dpi=300):
    """Converts a Matplotlib figure to a PIL Image with a white background.

    Args:
        fig: The Matplotlib figure to convert.
        dpi (int, optional): The resolution of the output image in dots per inch.
            Defaults to 100.

    Returns:
        PIL.Image: The converted image with a white background.
    """

    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=dpi)  # Specify PNG format and DPI
        buf.seek(0)
        img = Image.open(buf)

        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Paste with alpha channel

        return background
        

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.05, 
                point['y'],
                str(point['val']),
                fontsize=4,
                fontweight='bold')

def ncolor(n, cmap='viridis'):
    cmap = matplotlib.cm.get_cmap(cmap)
    arr = np.linspace(0, 1, n)
    return [matplotlib.colors.rgb2hex(cmap(x)) for x in arr] 


def get_colors(data, cmap):
    """A function to return seaborn colormap
    dict from a colum """
    color_list = sns.palettes.color_palette(cmap,
                                            data.nunique(), 
                                            as_cmap=False)
    return color_list


def parseKEGG(pathId):
    genes = []
    results = REST.kegg_get(pathId).read()
    current_section = None
    for line in results.rstrip().split("\n"):
        section = line[:12].strip()  # section names are within 12 columns
        if not section == "":
            current_section = section

        if current_section == "GENE":
            linesplit = line[12:].split("; ")
            gene_identifiers = linesplit[0]
            gene_id, gene_symbol = gene_identifiers.split()
    
            if not gene_symbol in genes:
                genes.append(gene_symbol)
    return genes

def getPathname(pathId):
    """A function to return the legg pathname"""
    result = REST.kegg_list(pathId).read()
    return result.split("\t")[1].split("-")[0].strip()


def makeColorbar(cmap, width, hieght, title, orientation, tickLabels):
    a = np.array([[0,1]])
    plt.figure(figsize=(width, hieght))
    img = plt.imshow(a, cmap=cmap)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])
    ticks = np.linspace(0,1 , len(tickLabels))
    cbar = plt.colorbar(orientation=orientation, 
                        cax=cax, 
                        label=title,
                        ticks=ticks)

    if orientation == 'vertical':
        cbar.ax.set_yticklabels(tickLabels)
    else:
        cbar.ax.set_xticklabels(tickLabels)

        
def _normalize_data(X, counts, after=None, copy=False):
    X = X.copy() if copy else X
    if issubclass(X.dtype.type, (int, np.integer)):
        X = X.astype(np.float32)  # TODO: Check if float64 should be used
    else:
        counts_greater_than_zero = counts[counts > 0]

    after = np.median(counts_greater_than_zero, axis=0) if after is None else after
    counts += counts == 0
    counts = counts / after
    if scipy.sparse.issparse(X):
        sparsefuncs.inplace_row_scale(X, 1 / counts)
    elif isinstance(counts, np.ndarray):
        np.divide(X, counts[:, None], out=X)
    else:
        X = np.divide(X, counts[:, None])  # dask does not support kwarg "out"
    return X


def normalize(df, target_sum=1):
    """A function to normalize spots """
    index = df.index
    columns = df.columns
    X = df.to_numpy().copy()
    counts_per_cell = X.sum(1)
    counts_per_cell = np.ravel(counts_per_cell)
    cell_subset = counts_per_cell > 0
    Xnorm = _normalize_data(X, counts_per_cell, target_sum)
    
    ndf = pd.DataFrame(Xnorm, columns=columns, index=index)
    return ndf