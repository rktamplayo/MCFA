# MCFA
Translations as Additional Contexts for Sentence Classification

This TensorFlow code was used in the experiments of the research paper

**Reinald Kim Amplayo**, Kyungjae Lee, Jinyoung Yeo, and Seung-won Hwang. **Translations as Additional Contexts for Sentence Classification**. _IJCAI_, 2018.

We provided the MR dataset and its translations to 10 other languages in the `data` folder to try the code.

Additionally, you will need two create three other folders which should contain the following:
- `pickles` folder: should contain nothing; the code automatically puts the pickled dataset files here
- `vectors` folder: should contain nothing; the code automatically puts the pickled word vector files here
- `pretrained` folder: should contain word vectors for all the languages that is used; these can be downloaded here: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

To run the code, generate the pickle files by executing the following:

`python src/CreateDataset.py`

To train and test the model, execute the following code:

`python src/mcfa.py`

To cite the paper/code, please use this BibTex:

```
@inproceedings{amplayo2017translations,
	Author = {Reinald Kim Amplayo and Kyungjae Lee and Jinyoung Yeo and Seung-won Hwang},
	Booktitle = {IJCAI},
	Location = {Stockholm, Sweden},
	Year = {2018},
	Title = {Translations as Additional Contexts for Sentence Classification},
}
```

If you have questions, send me an email: rktamplayo at yonsei dot ac dot kr
