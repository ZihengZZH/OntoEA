# OntoEA: Ontology-guided Entity Alignment via Joint Knowledge Graph Embedding

## Code

The source code of OntoEA is implemented based on [OpenEA](https://github.com/nju-websoft/OpenEA) and we follow the same design features of OpenEA to achieve functionality while maintaining extensibility. We retain the minimal fraction of OpenEA in the implmentation of OntoEA, and we list the source code (along with comments) of as follows.

```
code
|__ run
|   |__ args
|   |   |__ ontoea_args_15K.json    // OntoEA config file for 15K benchmark
|   |   |__ ontoea_args_100K.json   // OntoEA config file for 100K benchmark
|   |__ main_from_args.py           // OntoEA interface with args input
|   |__ run_OntoEA.sh               // bash script to run experiments of OntoEA
|__ src
|   |__ openea
|   |   |__ approaches
|   |   |   |__ ontoea.py           // OntoEA core code
|   |   |__ models
|   |   |   |__ basic_model.py      // same as OpenEA
|   |   |__ modules
|   |   |   |__ args
|   |   |   |   |__ args_hander.py  // same as OpenEA
|   |   |   |__ base
|   |   |   |   |__ initializers.py // same as OpenEA
|   |   |   |   |__ losses.py       // same as OpenEA
|   |   |   |   |__ mapping.py      // same as OpenEA (with minor updates)
|   |   |   |   |__ optimizers.py   // same as OpenEA
|   |   |   |__ finding
|   |   |   |   |__ alignment.py    // same as OpenEA
|   |   |   |   |__ evaluation.py   // same as OpenEA (with minor updates)
|   |   |   |   |__ similarity.py   // same as OpenEA
|   |   |   |__ load
|   |   |   |   |__ kg.py           // same as OpenEA
|   |   |   |   |__ kgs.py          // same as OpenEA (with updates)
|   |   |   |   |__ read.py         // same as OpenEA (with updates)
|   |   |   |__ train
|   |   |   |   |__ batch.py        // same as OpenEA (with minor updates)
|   |   |   |   |__ sample.py       // same as OpenEA (with minor updates)
|   |   |   |__ utils
|   |   |   |   |__ check.py        // same as OpenEA (with updates)
|   |   |   |   |__ inference.py    // same as OpenEA (with updates)
|   |   |   |   |__ util.py         // same as OpenEA
```

### how to run OntoEA

There are two ways to run OntoEA:
1. cd to ```./code/run``` and run the bash script ```./run_OntoEA.sh``` (recommended)
2. cd to ```./code/run``` and run OntoEA as ```python main_from_args.py ./args/ontoea_args_15K.json <benchmark_name> 721_5fold/1/```

### configuration of OntoEA

Different settings of OntoEA can be configured in the [ontoea_args_15K.json](code/run/args/ontoea_args_15K.json) and [ontoea_args_100K.json](code/run/args/ontoea_args_100K.json). The key configuration terms are listed as follows.

```
{
	"training_data": "../../benchmarks/",       // where the benchmark is
	"output": "../../output/results/",          // where the output results go
	"dataset_division": "721_5fold/1/",         // where the train/valid/test splits are
	"word_embed": "../../benchmarks/wiki-news-300d-1M.vec",     // where the word embeddings are
	"use_word_embed_init": 0,                   // whether or not to use word embeddings in OntoEA (i.e., 1 == OntoEA w/ SI; 0 == OntoEA w/o SI)
	"use_word_embed_init_onto": 0,              // whether or not to use word embeddings in OntoEA (i.e., 1 == OntoEA w/ SI; 0 == OntoEA w/o SI)
	"word_embed_init_zh": 0,                    // whether or not Chinese word embeddings (only for MED_BBK_9K)
	"use_alter_label": 0,                       // whether or not to use alternative labels for entity names (only for D_W benchmarks)
}
```

## Benchmarks

The newly constructed benchmarks extends the original KG alignment benchmarks ([OpenEA](https://github.com/nju-websoft/OpenEA) and [MED_BBK_9K](https://github.com/ZihengZZH/industry-eval-EA)) with ontological information, including ontology triples (both disjointWith and subClassOf), membership links (or cross-view links), ontology hierarchy.

The data files are identical for each benchmark. All the benchmarks along with the MED_BBK_9K details are listed as follows.

```
benchmarks
|__ EN_FR_15K_V1    // EN_FR_15K_V1 from OpenEA with appended ontological information
|__ EN_FR_15K_V2    // EN_FR_15K_V2 from OpenEA with appended ontological information
|__ EN_DE_15K_V1    // EN_DE_15K_V1 from OpenEA with appended ontological information
|__ EN_DE_15K_V2    // EN_DE_15K_V2 from OpenEA with appended ontological information
|__ D_W_15K_V1_A    // D_W_15K_V1 from OpenEA with appended ontological information and ontologies pre-aligned with PARIS
|__ D_W_15K_V1_M    // D_W_15K_V1 from OpenEA with appended ontological information and ontologies pre-aligned with manual annotation
|__ D_W_15K_V2_A    // D_W_15K_V2 from OpenEA with appended ontological information and ontologies pre-aligned with PARIS
|__ D_W_15K_V2_M    // D_W_15K_V2 from OpenEA with appended ontological information and ontologies pre-aligned with  manual annotation
|__ MED_BBK_9K      // MED_BBK_9K with appended ontological information
|   |__ 721_5fold/1         // train/valid/test splits
|   |__ attr_triples_1      // attribute triples of KG1
|   |__ attr_triples_2      // attribute triples of KG2
|   |__ rel_triples_1       // relation triples of KG1
|   |__ rel_triples_2       // relation triples of KG2
|   |__ ent_links           // all entity mappings between KG1 and KG2
|   |__ crossview_link_1    // membership links between KG1 and its ontology
|   |__ crossview_link_2    // membership links between KG2 and its ontology
|   |__ onto_attr_triples   // attribute triples of the aligned/shared ontology
|   |__ onto_disjointWith_triples   // disjointWith relation triples of the aligned/shared ontology
|   |__ onto_subClassOf_triples     // subClassOf relation triples of the aligned/shared ontology
|   |__ class_path.json     // class hierarchy of the aligned/shared ontology
```

## Citation

If you have any difficulty or question in running code and reproducing experimental results, please email to zihengzhang1025@gmail.com

If you use this model or code, please cite it as follows:

```
```
