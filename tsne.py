from sklearn.manifold import TSNE
import joblib
def tsne_func(data, label, no_components, perplexity_value, n_iter_value):
    start = datetime.now()
    print('TSNE with perplexity={} and no. of iterations={}'.format(perplexity_value, n_iter_value))
    tsne = TSNE(n_components=no_components, perplexity=perplexity_value, n_iter=n_iter_value)
    tsne_df1 = tsne.fit_transform(data)
    print(tsne_df1.shape)
    tsne_df1 = np.vstack((tsne_df1.T, Y)).T
    tsne_data1 = pd.DataFrame(data=tsne_df1, columns=['feature1', 'feature2', 'Output'])
    sns.FacetGrid(tsne_data1, hue='Output', size=6).map(plt.scatter, 'feature1', 'feature2').add_legend()
    print('Total time taken:',datetime.now()-start)
    plt.show()
    tsne_func(data=df, label=Y, no_components=2, perplexity_value=100, n_iter_value=500)
    tsne_func(data=df, label=Y, no_components=2, perplexity_value=50, n_iter_value=1000)