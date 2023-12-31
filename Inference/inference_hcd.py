import argparse
import os, warnings, logging, random
import json

import torch
from tqdm import tqdm
from processor import Processor
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from dataset import ResponseData
from torch.utils.data import DataLoader
from modeling_bart import BartForConditionalGeneration
from modeling_gpt2 import GPT2LMHeadModel

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def extract_response(len_history,outputs):
  output = []
  for i in range(len(outputs)):
    num = len_history[i]
    idx1 = 0
    idx2 = 0
    for j in range(len(outputs[i])):
      if outputs[i][j] == 21128 and num>1:
        num = num-1
      elif outputs[i][j] == 21128 and num==1:
        idx1 = j
        num = num-1
      elif outputs[i][j] == 21128 and num==0:
        idx2 = j
        output.append(outputs[i][idx1+1:idx2])
        break
  return output

def inference():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--model_dir", type=str, default="checkpoints")
    arg_parse.add_argument("--model_name", type=str, required=True)
    arg_parse.add_argument("--model_version", type=str, required=True)
    arg_parse.add_argument("--beam_size", type=int, default=None)
    arg_parse.add_argument("--alpha", type=float, default=1.0)
    arg_parse.add_argument("--first_k_steps", type=int, default=2)
    arg_parse.add_argument("--min_decode_length", type=int, default=10)
    arg_parse.add_argument("--max_decode_length", type=int, default=64)
    arg_parse.add_argument("--test_batch_size", type=int, default=4)
    arg_parse.add_argument("--test_path", type=str, default="/Users/yaoyuxuan/PycharmProjects/SimDRC/response_generation/data/dailydialog/dailydialog.debug.txt")
    arg_parse.add_argument("--no_cuda", action="store_true", default=False)
    arg_parse.add_argument("--seed", type=int, default=42)
    arg_parse.add_argument("--top_p", type=float, default=None)
    arg_parse.add_argument("--top_k", type=int, default=None)
    arg_parse.add_argument("--num_beams", type=int, default=None)
    arg_parse.add_argument("--do_sample", type=str, default=None)
    arg_parse.add_argument("--verbose", action="store_true", default=False)
    arg_parse.add_argument("--type", type=str, default="greedy")
    arg_parse.add_argument("--early_stopping", type=bool, default=True)

    args = arg_parse.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    n_gpu = torch.cuda.device_count()
    logger.info(" ** ** * device: {} n_gpu: {} * ** ** ".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed(args.seed)

    model_dir = os.path.join(args.model_dir, args.model_name)
    model_path = os.path.join(model_dir, args.model_version)
    training_config = json.load(open(os.path.join(model_dir, "training.config"), "r", encoding="utf-8"))
    logger.info(" ** ** * training configuration * ** ** ")
    logger.info(training_config)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_config = AutoConfig.from_pretrained(training_config['language_model'])
    model_config.vocab_size = training_config["vocab_size"]
    if "bart" in training_config['language_model']:
        model = BartForConditionalGeneration.from_pretrained(training_config['language_model'])
        padding_value = model.config.pad_token_id
    else:
        # model = AutoModelForCausalLM.from_pretrained(training_config['language_model'])
        model = GPT2LMHeadModel.from_pretrained(training_config['language_model'])
        # padding_value = 1.0
        padding_value = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    # load pretrained model
    logger.info(" ** ** * Loading model weights from {} * ** ** ".format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if hasattr(model, 'module'):
        model = model.module
    model.to(device)
    eou_token = training_config['eou_token']
    test_set = ResponseData(args.test_path, eou=eou_token)
    processor = Processor(tokenizer=tokenizer, max_len=training_config['max_seq_len'], eou=eou_token,
                          model_type=training_config['language_model'], is_training=False,
                          use_cross_loss=False, use_locality_loss=False)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False,
                             collate_fn=processor.batch_collate_fn_hcd)

    logger.info("******* Running training *******")
    logger.info("   NUM examples = {}   ".format(len(test_set)))
    logger.info("   MAX decode length = {}  ".format(args.max_decode_length))
    logger.info("   Test batch size= {}  ".format(args.test_batch_size))

    output_dir = os.path.join("output", args.model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model.eval()
    with torch.no_grad():
        with open(os.path.join(output_dir, "{}-max_len-{}-min_len-{}.txt".format(
                args.type, args.max_decode_length, args.min_decode_length)),
                  "w", encoding="utf-8") as fw:
            for step, batch in enumerate(tqdm(test_loader)):
                for k in batch.keys():
                    batch[k] = batch[k].to(device)
                    # print(batch[k])
                print("batch", batch["input_ids"].size())
                outputs = model.hierarchically_decoding(
                    tokenizer=tokenizer,
                    eos_position=batch["eos_position"],
                    attention_mask=batch["attention_mask"],
                    input_ids=batch["input_ids"],
                    beam_width=args.beam_size,
                    alpha=args.alpha,
                    top_k=args.top_k,
                    num_beams=args.num_beams,
                    max_decoding_length=args.max_decode_length if "bart" in args.model_dir else
                    args.max_decode_length+batch["input_ids"].shape[1],
                    eos_token_id=tokenizer.eos_token_id,
                    first_k_steps=args.first_k_steps,
                    do_sample=args.top_p is not None or args.top_k is not None,
                    top_p=args.top_p,
                    decoder_start_token_id=model.config.decoder_start_token_id if "bart" in args.model_dir else None,
                    hcd_padding_value=padding_value,
                    no_repeat_ngram_size=2,
                    early_stopping=args.early_stopping,
                )
                # len_history = list(batch["len_history"])
                # output = extract_response(len_history,outputs)
                generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                original_history = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                # assert len(generated) == len(original_history) == len(labels)
                for i, (word, gold, pred) in enumerate(zip(original_history, labels, generated)):
                    if args.verbose:
                        print(pred)
                    fw.write(json.dumps(
                        {"id": i, "history": word, "predict": pred, "label": gold}, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    inference()
