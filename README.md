# IPS: Fine-grained Conversational Decoding via Isotropic and Proximal Search
This is the code of EMNLP 2023 Main Conference Paper: "Fine-grained Conversational Decoding via Isotropic and Proximal Search" 

[paper](https://arxiv.org/abs/2310.08130)
***
##  Introduction
General-purpose text decoding approaches are usually adopted for dialogue response generation. Although the quality of the generated responses can be improved with dialogue-specific encoding methods, conversational decoding methods are still under-explored. Inspired by   SimDRC(Wu et al.) that a good dialogue feature space should follow the rules of locality and isotropy, we present a fine-grained conversational decoding method, termed *isotropic and proximal search* (**IPS**). Our method is designed to generate the semantic-concentrated response, while still maintaining informativeness and discrimination against the context.
Experiments show that our approach outperforms existing decoding strategies in the dialogue field across both automatic and human evaluation metrics. More in-depth analyses further confirm the effectiveness of our approach.

IPS can be directly used with different models and achieve good performance, while the models trained with SimDRC are the best testbed for IPS.

For experiment settings, We follow [SimDRC](https://github.com/hahahawu/SimDRC).
***
## Citation
```
@misc{yao2023finegrained,
      title={Fine-grained Conversational Decoding via Isotropic and Proximal Search}, 
      author={Yuxuan Yao and Han Wu and Qiling Xu and Linqi Song},
      year={2023},
      eprint={2310.08130},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



