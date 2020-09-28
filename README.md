# Knowledge_Graph_Construction
**A coursework project for Knowledge Discovery class**

-----------------------------

### Dependencies

We need several libraries to run the entire project. On your terminal, run:

```
pip install -r requirements.txt
```

If you wish to visualize the graph, you need to install also [Graphviz](https://www.graphviz.org/download/) on your global environment. See the page for instruction.

### Data

Currently, the default data directory is `./data` with the `corpus.zipped` including the first 1000 sentences of the dataset. If you wish to use another data dir or simply want to perform the experiment on a larger dataset, please use the path and filename accordingly.


### Experiment

The main file of the entire project is `main.py`. The file accepts these following arguments:

```
--perform             which task to perform, default=extract. Choices=['read-corpus', 'extract', 'cluster', 'evaluate', 'visual', 'all']
--path_to_data_dir    path to data directory, default='./data'
--corpus_name         name of covid corpus to load, default='covid19.vert'
--max_sent            maximum number of sentences to retrieve from the corpus, default=1000
--mark_print          print features, including core and optional tokens, if any, of a certain sentence, default=None
--distance_metric     metric to compute distance metrix for clustering, default='cosine'
--linkage             which linkage criterion to use, default='average', choices=['average', 'single', 'complete', 'ward']
--distance_threshold  cutting threshold, above which the clusters won't be merged, default=0.999
--ranked_metric       metric for ranking patterns, default='count', choices=['count', 'tfidf']
--with_data           which dataset for visualization, default='ours', choices=['ours', 'cido']
--num_nodes           number of maximum nodes for drawing the graph, default=30
```

The most important argument is `--perform`, in which you need to specify which task to perform. Guide to each action is as follows:

For `--perform read-corpus`, relevant arguments are `path_to_data_dir`, `corpus_name`, and `max_sent`.

For `--perform extract`, relevant arguments are `path_to_data_dir` (if path is different from default), and `mark_print`.

For `--perform cluster`, relevant arguments are `path_to_data_dir` (if path is different from default), `distance_metric`, `linkage`, and `distance_threshold`.

For `--perform evaluate`, relevant arguments are `path_to_data_dir` (if path is different from default), and `ranked_metric`.

For `--perform visual`, relevant arguments are `path_to_data_dir` (if path is different from default), `with_data`, and `num_nodes`.

Or you can simply type `--perform all` to run everything from beginning to end. Be warned that a lot of information will be printed. Defaults are set up as specified in the project report.
